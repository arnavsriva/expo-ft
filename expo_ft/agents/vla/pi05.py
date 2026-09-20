"""Wrapper for pi05 agent to make it compatible with expo training framework."""

import dataclasses
import functools
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

from openpi_client import image_tools

from expo_ft.agents.vla.vla_base import Model
from expo_ft.data.dataset import DatasetDict


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Load and validate weights.
    
    Args:
        loader: Weight loader instance.
        params_shape: Expected parameter shape.
        
    Returns:
        Loaded subset of the weights.
    """
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape,
        got=loaded_params,
        check_shapes=True,
        check_dtypes=True,
    )

    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )

def pi05_get_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
    else:
        params = state.params
    return params


def pi05_init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
    is_target: bool = False,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict()
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    if is_target:
        model_params = pi05_get_params(train_state)
        return model_params
    return train_state, state_sharding


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # Get current learning rate from the schedule
    lr_schedule_fn = config.lr_schedule.create()
    current_lr = lr_schedule_fn(state.step)
    info = {
        "actor_loss": loss,
        "actor_grad_norm": optax.global_norm(grads),
        "actor_param_norm": optax.global_norm(kernel_params),
        "actor_lr": current_lr,
        "actor_state_step": state.step,
    }
    return new_state, info


def _bind_model(train_state: training_utils.TrainState, train: bool = False):
    """Bind model from train state."""
    model = nnx.merge(train_state.model_def, train_state.params)
    if train:
        model.train()
    return model


# wallclock-vla-rl: flow-matching denoising steps as an inference-latency knob.
# Unset -> openpi default (10). A Python int captured at trace time, so it is static under jit.
_WALLCLOCK_SAMPLE_KWARGS = (
    {"num_steps": int(os.environ["WALLCLOCK_FLOW_STEPS"])} if os.environ.get("WALLCLOCK_FLOW_STEPS") else {}
)


@functools.partial(jax.jit, static_argnames=['train', 'num_samples'])
def _jitted_infer(transformed_inputs, train_state, rng, policy_metadata, train, num_samples):
    model = _bind_model(train_state, train=False)
    sample_policy = _policy.Policy(
        model=model,
        rng=rng,
        transforms=[],  
        output_transforms=[],
        sample_kwargs=dict(train=train, num_samples=num_samples, **_WALLCLOCK_SAMPLE_KWARGS),
        metadata=policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )
    sample_info = sample_policy.infer(transformed_inputs, is_batch=True, for_training=True)
    return sample_info


def build_pi05(config, seed, mesh, data_sharding, replicated_sharding,
               resume, default_prompt):
    """Build Pi05 actor, train state, target params, and metadata from agent config.

    Returns (actor, actor_train_state, target_actor_params, agent_kwargs, metadata)
    where metadata is a dict with action_horizon, resize_size, freeze_encoder
    ready to pass into EXPOLearner/BCLearner.create().
    """
    from expo_ft.utils.train_utils import build_pi05_config
    agent_kwargs, pi05_train_config, pi05_resize_size, _ = build_pi05_config(config)
    freeze_encoder = agent_kwargs.pop("freeze_pi05_encoder", False)

    rng = jax.random.PRNGKey(seed)
    init_rng, rng = jax.random.split(rng)
    target_rng, rng = jax.random.split(rng)

    actor, actor_train_state, _ = Pi05Agent.initialize(
        pi05_train_config,
        mesh,
        init_rng,
        resume=resume,
        default_prompt=default_prompt,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        freeze_pi05_encoder=freeze_encoder,
        infer_device=jax.devices()[0],
        skip_repack_transforms=agent_kwargs.pop("skip_repack_transforms", False),
    )
    if resume:
        target_actor_params = actor.get_params(actor_train_state)
    else:
        target_actor_params = actor.init_target_params(target_rng, resume=resume)

    metadata = dict(
        action_horizon=pi05_train_config.model.action_horizon,
        resize_size=pi05_resize_size,
        freeze_encoder=freeze_encoder,
    )
    return actor, actor_train_state, target_actor_params, agent_kwargs, metadata


class Pi05Agent(Model):
    """Wrapper for pi05 model to make it compatible with expo training framework."""

    def __init__(
        self,
        *,
        train_config: Any,
        mesh: jax.sharding.Mesh,
        train_state_sharding: jax.sharding.NamedSharding,
        data_sharding: jax.sharding.NamedSharding,
        replicated_sharding: jax.sharding.NamedSharding,
        default_prompt: str,
        freeze_pi05_encoder: bool = False,
        infer_device: Optional[jax.Device] = None,
        action_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        skip_repack_transforms: bool = False,
    ):

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.train_config = train_config
        self.data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        self.model_config = train_config.model

        self.default_prompt = default_prompt
        self.freeze_pi05_encoder = freeze_pi05_encoder

        self.mesh = mesh
        self.train_state_sharding = train_state_sharding
        self.data_sharding = data_sharding
        self.replicated_sharding = replicated_sharding
        self.infer_device = infer_device or jax.devices()[0]
        self.infer_sharding = jax.sharding.SingleDeviceSharding(self.infer_device)
        self.skip_repack_transforms = skip_repack_transforms
        self.input_transforms = self._build_input_transform_pipeline()
        self.output_transforms = self._build_output_transform_pipeline()

        self.train_step = jax.jit(
            functools.partial(train_step, self.train_config),
            in_shardings=(self.replicated_sharding, self.train_state_sharding, self.data_sharding),
            out_shardings=(self.train_state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )

    def _build_input_transform_pipeline(self, normalize: bool = True):
        """Compose repack, data, normalize, and model transforms for raw inputs."""
        repack = [] if self.skip_repack_transforms else self.data_config.repack_transforms.inputs
        if normalize:
            transforms = [
                *repack,
                *self.data_config.data_transforms.inputs,
                _transforms.Normalize(
                    self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm
                ),
                *self.data_config.model_transforms.inputs,
            ]
        else:
            transforms = [
                *repack,
                *self.data_config.data_transforms.inputs,
                *self.data_config.model_transforms.inputs,
             ]
        return _transforms.compose(transforms)

    def _build_output_transform_pipeline(self, unnormalize: bool = True):
        """Compose model, unnormalize, and data transforms for model outputs."""
        if unnormalize:
            transforms = [
                *self.data_config.model_transforms.outputs,
                _transforms.Unnormalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.data_transforms.outputs,
                # *repack_transforms.outputs,
            ]
        else:
            transforms = [
                *self.data_config.model_transforms.outputs,
                *self.data_config.data_transforms.outputs,
            ]
        return _transforms.compose(transforms)

    @classmethod
    def load_pi05_config(cls, config_name: str):
        return _config.get_config(config_name)

    @classmethod
    def initialize(
        cls,
        train_config: _config.TrainConfig,
        mesh: jax.sharding.Mesh,
        init_rng: jax.random.PRNGKey,
        *,
        resume: bool = False,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        freeze_pi05_encoder: bool = False,
        infer_device: Optional[jax.Device] = None,
        skip_repack_transforms: bool = False,
    ) -> tuple["Pi05Agent", Any]:
        """Initialize a Pi05Agent instance using init_train_state."""
        train_state, train_state_sharding = pi05_init_train_state(
            train_config,
            init_rng,
            mesh,
            resume=resume,
            is_target=False,
        )

        agent = cls(
            train_config=train_config,
            mesh=mesh,
            train_state_sharding=train_state_sharding,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            default_prompt=default_prompt,
            freeze_pi05_encoder=freeze_pi05_encoder,
            infer_device=infer_device,
            skip_repack_transforms=skip_repack_transforms,
        )
        
        return agent, train_state, train_state_sharding

    def get_params(self, train_state):
        return pi05_get_params(train_state)

    def init_target_params(self, rng, *, resume=False):
        return pi05_init_train_state(
            self.train_config, rng, self.mesh, resume=resume, is_target=True
        )

    def process_raw_inputs(self, raw_observations, action_dim, resize_size, normalize=True):
        """Convert raw env observations into a batched model-ready Observation dict."""
        # create a dummy actions
        raw_observations["actions"] = np.zeros(action_dim)
        for key, value in raw_observations.items():
            raw_observations[key] = np.asarray(value)
            if "image" in key:
                # make image uint8
                raw_observations[key] = raw_observations[key].astype(np.uint8)
                assert np.max(raw_observations[key]) > 1

        transform_fns = self.input_transforms
        transformed_inputs = transform_fns(raw_observations)
        transformed_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed_inputs)
        processed_inputs = _model.Observation.from_dict(transformed_inputs).to_dict()
        return processed_inputs

    def process_transformed_outputs(self, transformed_actions, unnormalize=True):
        """Unnormalize unpadded actions back to the environment action space."""
        n = transformed_actions.shape[0]
        padded = self._pad_actions(transformed_actions.reshape(n, -1))
        dummy_state = np.zeros((n, self.model_config.action_dim), dtype=np.float32)
        output_dict = {
            "state": dummy_state,
            "actions": np.array(padded),
        }
        processed = [
            self.output_transforms(jax.tree.map(lambda x: x[i], output_dict))
            for i in range(n)
        ]
        return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *processed)["actions"]

    def prepare_batch_for_actor(self, batch):
        """Build (observation, padded actions) tuple for actor loss computation."""
        obs = _model.Observation.from_dict(batch.copy())
        actions = self._pad_actions(batch["full_actions"])
        return (obs, actions)

    def _unpad_actions(self, actions):
        """Reshape padded model actions to (batch, horizon, env action_dim)."""
        padded_dim = self.model_config.action_dim
        action_horizon = self.model_config.action_horizon
        actions = actions.reshape(actions.shape[0], action_horizon, padded_dim)
        return actions[..., :self.action_dim]

    def _pad_actions(self, actions):
        """Zero-pad env actions to the model's expected action dimension."""
        padded_dim = self.model_config.action_dim
        action_horizon = self.model_config.action_horizon
        actions = actions.reshape(actions.shape[0], action_horizon, self.action_dim)
        return jnp.concatenate([
            actions,
            jnp.zeros((actions.shape[0], action_horizon, padded_dim - self.action_dim))
        ], axis=-1)

    def sample_actions(
        self,
        transformed_inputs: Dict,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        train: Optional[bool] = False,
        num_samples: Optional[int] = 1,
    ) -> tuple:
        """Sample actions from pi05 model using already-transformed inputs."""
        infer_sharding = self.infer_sharding
        transformed_inputs = jax.tree.map(
            lambda x: jax.device_put(x, infer_sharding) if isinstance(x, (jnp.ndarray, np.ndarray)) else x,
            transformed_inputs,
        )
        if num_samples > 1 and not self.freeze_pi05_encoder:
            def repeat_value(v, n):
                if isinstance(v, dict):
                    return {k: repeat_value(vv, n) for k, vv in v.items()}
                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    return jnp.repeat(v, n, axis=0)
                return v
            transformed_inputs = repeat_value(transformed_inputs, num_samples)

        key, rng = jax.random.split(rng)
        key = jax.device_put(key, infer_sharding)
        infer_train_state = jax.device_put(train_state, infer_sharding)
        noise_samples = 1 if not self.freeze_pi05_encoder else num_samples
        sample_info = _jitted_infer(transformed_inputs, infer_train_state, key, self.train_config.policy_metadata, train, noise_samples)

        actions = self._unpad_actions(sample_info["actions"])
        return actions, sample_info["policy_timing"]["infer_ms"]


    def sample_training_actions(
        self,
        transformed_inputs,
        train_state: training_utils.TrainState,
        rng: jax.random.PRNGKey,
        train: Optional[bool] = True,
        num_samples: Optional[int] = 1,
    ) -> jnp.ndarray:
        """Sample actions from pi05 model."""
        key, rng = jax.random.split(rng)
        sample_info = _jitted_infer(transformed_inputs, train_state, key, self.train_config.policy_metadata, train, num_samples)
        actions = self._unpad_actions(sample_info["actions"])
        return actions, sample_info["policy_timing"]["infer_ms"]

