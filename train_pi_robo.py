#! /usr/bin/env python
import os
import gc
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir, save_replay_buffer_transition
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb
from expo_ft.utils.config_loader import load_task_config, resolve_run_dir


import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from torch.utils.tensorboard import SummaryWriter


FLAGS = flags.FLAGS

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")
flags.DEFINE_string("task_config", "configs/task/stack_cube.yaml", "Path to task YAML config.")


config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def get_or_create_episode_seeds(output_dir, n_episodes, master_seed):
    """Mirrors scripts/eval_curve.py's own get_or_create_episode_seeds()
    EXACTLY (same generation logic: np.random.default_rng(master_seed),
    same cache file name/location) so a periodic in-training eval here uses
    the IDENTICAL fixed seed list eval_curve.py would generate for the SAME
    (output_dir, n_episodes, master_seed) -- true apples-to-apples
    comparability with the SFT baseline's own evaluation, without needing to
    locate or copy any specific file: same inputs, same deterministic PRNG,
    same seeds, wherever this is called from. Kept as a separate copy here
    (not imported from scripts/eval_curve.py) to avoid a fragile
    cross-directory import; if eval_curve.py's own version ever changes,
    this one must be updated to match, or the two would silently diverge."""
    output_dir = Path(output_dir)
    seeds_path = output_dir / "episode_seeds.json"
    if seeds_path.exists():
        with open(seeds_path) as f:
            seeds = json.load(f)
        if len(seeds) != n_episodes:
            raise ValueError(
                f"Existing {seeds_path} has {len(seeds)} seeds but rl_eval_episodes={n_episodes}. "
                f"Delete the file to regenerate, or fix rl_eval_episodes to match."
            )
        return seeds
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_episodes).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f, indent=2)
    logging.info(f"[rigorous-eval] Generated {n_episodes} fixed episode seeds "
                 f"(master_seed={master_seed}): {seeds_path}")
    return seeds


def run_rigorous_eval(agent, eval_env, episode_seeds, cfg, only_base_actions=False):
    """Deterministic, fixed-seed evaluation -- Jesse's proposed protocol.
    Runs episode_seeds' episodes with agent.sample_actions(..., deterministic=True)
    (residual actor's mode, not a stochastic sample -- see expo_ft.py's
    sample_actions docstring) and returns (success_rate, stderr).

    only_base_actions: at step 0 (before any RL training), the residual and
    critic are freshly, randomly initialized -- deterministic=True only
    makes their OWN output deterministic, it does NOT mean "no residual
    contribution." Without only_base_actions=True, the step-0 number would
    actually measure SFT + an arbitrary untrained-residual offset + an
    untrained-critic argmax selection, not the clean frozen-SFT baseline
    Jesse's protocol calls for as the reference point. only_base_actions=True
    bypasses the whole residual/critic mechanism, matching exactly how
    eval_policy.py's own baseline mode works. Note this makes the eval
    genuinely non-deterministic at step 0 regardless of the deterministic=True
    passed to sample_actions internally -- the frozen base VLA's own
    sampling stays stochastic either way (no equivalent deterministic mode
    for a flow-matching model), which is expected and fine: there's no
    trained residual to speak of yet at step 0, so "deterministic" has
    nothing to apply to. Should be False for every later, trained-agent
    call (only_base_actions defaults to False, matching that).

    Deliberately does NOT touch the `agent` object passed in: uses its own
    local `eval_agent` variable, reassigned across this function's own
    rollout steps exactly like sample_actions() always requires (each call
    returns a new agent with an advanced .rng), but that variable is purely
    local and discarded when this function returns. The caller's own
    `agent` (and its .rng sequence) is completely unaffected by having run
    this evaluation -- training resumes exactly as if this detour never
    happened, only wandb gets the new eval_rigorous/* numbers."""
    from collections import deque
    eval_agent = agent
    successes = []
    for ep in tqdm.tqdm(range(len(episode_seeds)), desc="[rigorous-eval]", disable=not FLAGS.tqdm):
        obs = eval_env.reset(seed=int(episode_seeds[ep]))
        done = False
        steps = 0
        success = False
        action_plan = deque()
        while not done and steps < cfg.max_steps_per_episode:
            if not action_plan:
                action_chunk, eval_agent, _ = eval_agent.sample_actions(obs, only_base_actions=only_base_actions, deterministic=True)
                action_plan.extend(action_chunk[:cfg.replan_steps])
            action = action_plan.popleft()
            eval_env.step(action.tolist())
            done, success, _, _ = eval_env.get_info_for_step()
            obs = eval_env.get_observation()
            steps += 1
        successes.append(float(success))
    successes = np.asarray(successes)
    success_rate = float(successes.mean())
    stderr = float(successes.std(ddof=1) / np.sqrt(len(successes))) if len(successes) > 1 else 0.0
    return success_rate, stderr


def reconstruct_success_window(replay_buffer, success_rate_window):
    """On resume, when checkpoint_buffer=true has restored real past
    transitions (see restore_replay_buffer in expo_ft/data/replay_buffer.py
    -- files are named by step and reloaded via insert() in that same
    order, so the buffer's own index order IS true temporal order), rebuild
    eval/success_rate's rolling window directly from that restored history
    instead of leaving it empty. Without this, _success_window stays empty
    on every resume (it's a separate in-memory list, never populated by
    batch_processor.restore() itself) and eval/success_rate silently
    reports nothing until success_rate_window BRAND NEW episodes complete
    post-resume -- with a typical PushCube episode length, that's ~20K
    steps of pure silence on a metric checkpoint_buffer was specifically
    meant to make resume-safe.

    dones marks each episode's LAST transition; is_success at that same
    index is that episode's outcome (matches exactly how the live
    training loop appends to _success_window -- see the two call sites of
    training_log._success_window.append(...) in this file). Returns up to
    the last `success_rate_window` such outcomes, in original temporal
    order (oldest of the kept ones first) -- if fewer than
    success_rate_window episodes exist in the restored buffer, returns
    all of them (a partial pre-fill is still strictly better than none:
    it shrinks whatever real wait remains instead of eliminating it)."""
    size = replay_buffer._size
    if size == 0:
        return []
    dones = np.asarray(replay_buffer.dataset_dict["dones"][:size])
    is_success = np.asarray(replay_buffer.dataset_dict["is_success"][:size])
    episode_end_indices = np.flatnonzero(dones)
    if len(episode_end_indices) == 0:
        return []
    kept = episode_end_indices[-success_rate_window:]
    return is_success[kept].astype(float).tolist()


def main(_):
    init_logging()

    # Load task config from YAML
    cfg = load_task_config(FLAGS.task_config)
    # wallclock-vla-rl: env overrides + optional phase tracing (no-op unless WALLCLOCK_TRACE is set)
    from expo_ft.utils.wallclock_hooks import apply_env_overrides, make_wallclock
    apply_env_overrides(cfg)
    WC = make_wallclock({
        "role": "sync", "mode": "sync", "task": getattr(cfg, "env_id", None),
        "control_hz": getattr(cfg, "control_hz", None), "utd_ratio": getattr(cfg, "utd_ratio", None),
        "replan_steps": getattr(cfg, "replan_steps", None), "batch_size": getattr(cfg, "batch_size", None),
        "seed": getattr(cfg, "seed", None), "num_updates": getattr(cfg, "num_updates", None),
        "update_type": getattr(cfg, "update_type", None),
    })
    # Override pi05_config_name dynamically from task config
    from expo_ft.utils.config_loader import get_sft_config_name
    FLAGS.config.pi05_config_name = get_sft_config_name(cfg)
    FLAGS.config.skip_repack_transforms = cfg.skip_repack_transforms

    # Read once, early — used both for the hyperparameter overrides below and
    # for the learner-import dispatch further down. Reading it here (rather
    # than only later, where it used to be read) does not change any
    # behavior — FLAGS.config.model_cls is already fully populated at this
    # point since ml_collections loads --config before main() runs.
    model_cls = FLAGS.config.model_cls

    # Override FLAGS.config RL hyperparameters from the task YAML so everything
    # is configured in one place (the YAML) rather than split between YAML and
    # configs/model/*.py. Each learner reads its own, differently-prefixed
    # fields (rl_* for EXPOLearner, ppo_* for PPOLearner, grpo_* for
    # GRPOLearner) so a single task YAML can hold tuned overrides for every
    # algorithm at once without collisions.
    # NOTE: float() wrapping below is a deliberate defense against a PyYAML quirk —
    # bare scientific notation without a decimal point (e.g. "3e-4") is parsed as
    # a STRING, not a float (needs "3.0e-4" to parse correctly). ml_collections
    # then raises a TypeError trying to assign a str into a float-typed field.
    # float(x) is a no-op if x is already a float, and fixes it if x is a
    # not-quite-valid-YAML-float string — belt and suspenders alongside fixing
    # the YAML values themselves.
    if model_cls in ("EXPOLearner", "EXPOLearnerCategorical"):
        # --- unchanged from before this refactor: byte-for-byte identical ---
        FLAGS.config.actor_lr         = float(getattr(cfg, "rl_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr        = float(getattr(cfg, "rl_lr", FLAGS.config.critic_lr))
        # NOTE: previously orphaned -- temp_lr silently stayed at whatever
        # FLAGS.config.temp_lr (3e-4, from sac_config.py) already was,
        # decoupled from rl_lr above and with no YAML field to override it
        # independently. Wired here the same way as the other rl_* fields;
        # default is unchanged (falls back to FLAGS.config.temp_lr) for any
        # YAML that doesn't set rl_temp_lr yet.
        FLAGS.config.temp_lr          = float(getattr(cfg, "rl_temp_lr", FLAGS.config.temp_lr))
        FLAGS.config.discount         = float(getattr(cfg, "rl_discount", FLAGS.config.discount))
        FLAGS.config.tau              = float(getattr(cfg, "rl_tau", FLAGS.config.tau))
        FLAGS.config.init_temperature = float(getattr(cfg, "rl_init_temperature", FLAGS.config.init_temperature))
        FLAGS.config.adjust_target_entropy = getattr(cfg, "rl_adjust_target_entropy", FLAGS.config.adjust_target_entropy)
        _rl_fixed_temperature = getattr(cfg, "rl_fixed_temperature", FLAGS.config.fixed_temperature)
        FLAGS.config.fixed_temperature = float(_rl_fixed_temperature) if _rl_fixed_temperature is not None else None
        _rl_critic_weight_decay = getattr(cfg, "rl_critic_weight_decay", FLAGS.config.critic_weight_decay)
        FLAGS.config.critic_weight_decay = float(_rl_critic_weight_decay) if _rl_critic_weight_decay is not None else None
        _rl_critic_grad_clip_norm = getattr(cfg, "rl_critic_grad_clip_norm", FLAGS.config.critic_grad_clip_norm)
        FLAGS.config.critic_grad_clip_norm = float(_rl_critic_grad_clip_norm) if _rl_critic_grad_clip_norm is not None else None
        FLAGS.config.freeze_critic_encoder = getattr(cfg, "rl_freeze_critic_encoder", FLAGS.config.freeze_critic_encoder)
        if hasattr(cfg, "rl_hidden_dims"):
            FLAGS.config.hidden_dims  = tuple(cfg.rl_hidden_dims)
        FLAGS.config.edit_scale       = float(getattr(cfg, "rl_edit_scale", FLAGS.config.edit_scale))
        FLAGS.config.N = int(getattr(cfg, "rl_N", FLAGS.config.N))
        FLAGS.config.n_edit_samples = int(getattr(cfg, "rl_n_edit_samples", FLAGS.config.n_edit_samples))
        # --- end of original ExpoFT block; critic_pretrain_steps added below is
        # a new, default-off (0) field — behavior for existing configs that
        # don't set rl_critic_pretrain_steps is unchanged ---
        FLAGS.config.critic_pretrain_steps = int(getattr(cfg, "rl_critic_pretrain_steps", FLAGS.config.critic_pretrain_steps))
        FLAGS.config.actor_bc_pretrain_steps = int(getattr(cfg, "rl_actor_bc_pretrain_steps", FLAGS.config.actor_bc_pretrain_steps))
        FLAGS.config.num_atoms = int(getattr(cfg, "rl_num_atoms", FLAGS.config.num_atoms))
        FLAGS.config.v_min = float(getattr(cfg, "rl_v_min", FLAGS.config.v_min))
        FLAGS.config.v_max = float(getattr(cfg, "rl_v_max", FLAGS.config.v_max))
        FLAGS.config.reward_scale_decay = float(getattr(cfg, "rl_reward_scale_decay", FLAGS.config.reward_scale_decay))
        FLAGS.config.use_reward_normalization = bool(getattr(cfg, "rl_use_reward_normalization", FLAGS.config.use_reward_normalization))
        FLAGS.config.kl_coef = float(getattr(cfg, "rl_kl_coef", FLAGS.config.kl_coef))
        FLAGS.config.entropy_scale = float(getattr(cfg, "rl_entropy_scale", FLAGS.config.entropy_scale))
        FLAGS.config.kl_ref_std = float(getattr(cfg, "rl_kl_ref_std", FLAGS.config.kl_ref_std))
        FLAGS.config.use_hetstat_policy = bool(getattr(cfg, "rl_use_hetstat_policy", FLAGS.config.use_hetstat_policy))
        FLAGS.config.hetstat_num_rff_features = int(getattr(cfg, "rl_hetstat_num_rff_features", FLAGS.config.hetstat_num_rff_features))
        FLAGS.config.hetstat_var_lr_multiplier = float(getattr(cfg, "rl_hetstat_var_lr_multiplier", FLAGS.config.hetstat_var_lr_multiplier))
        FLAGS.config.use_double_q_selection = bool(getattr(cfg, "rl_use_double_q_selection", FLAGS.config.use_double_q_selection))
        FLAGS.config.use_clipped_double_q = bool(getattr(cfg, "rl_use_clipped_double_q", FLAGS.config.use_clipped_double_q))
    elif model_cls == "PPOLearner":
        FLAGS.config.actor_lr  = float(getattr(cfg, "ppo_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr = float(getattr(cfg, "ppo_lr", FLAGS.config.critic_lr))
        FLAGS.config.discount  = float(getattr(cfg, "ppo_discount", FLAGS.config.discount))
        FLAGS.config.gae_lambda        = float(getattr(cfg, "ppo_gae_lambda", FLAGS.config.gae_lambda))
        FLAGS.config.clip_eps          = float(getattr(cfg, "ppo_clip_eps", FLAGS.config.clip_eps))
        FLAGS.config.value_loss_coef   = float(getattr(cfg, "ppo_value_loss_coef", FLAGS.config.value_loss_coef))
        FLAGS.config.entropy_coef      = float(getattr(cfg, "ppo_entropy_coef", FLAGS.config.entropy_coef))
        _ppo_value_clip_eps = getattr(cfg, "ppo_value_clip_eps", FLAGS.config.value_clip_eps)
        FLAGS.config.value_clip_eps = float(_ppo_value_clip_eps) if _ppo_value_clip_eps is not None else None
        _ppo_max_grad_norm = getattr(cfg, "ppo_max_grad_norm", FLAGS.config.max_grad_norm)
        FLAGS.config.max_grad_norm = float(_ppo_max_grad_norm) if _ppo_max_grad_norm is not None else None
        FLAGS.config.num_minibatches = int(getattr(cfg, "ppo_num_minibatches", FLAGS.config.num_minibatches))
        if hasattr(cfg, "ppo_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.ppo_hidden_dims)
        FLAGS.config.actor_pretrain_steps = int(getattr(cfg, "ppo_actor_pretrain_steps", FLAGS.config.actor_pretrain_steps))
        FLAGS.config.rollout_length = int(getattr(cfg, "ppo_rollout_length", FLAGS.config.rollout_length))
        FLAGS.config.actor_log_std_min = float(getattr(cfg, "ppo_actor_log_std_min", FLAGS.config.actor_log_std_min))
    elif model_cls == "GRPOLearner":
        FLAGS.config.actor_lr     = float(getattr(cfg, "grpo_lr", FLAGS.config.actor_lr))
        FLAGS.config.group_size   = int(getattr(cfg, "grpo_group_size", FLAGS.config.group_size))
        FLAGS.config.clip_eps     = float(getattr(cfg, "grpo_clip_eps", FLAGS.config.clip_eps))
        FLAGS.config.kl_coef      = float(getattr(cfg, "grpo_kl_coef", FLAGS.config.kl_coef))
        FLAGS.config.entropy_coef = float(getattr(cfg, "grpo_entropy_coef", FLAGS.config.entropy_coef))
        _grpo_max_grad_norm = getattr(cfg, "grpo_max_grad_norm", FLAGS.config.max_grad_norm)
        FLAGS.config.max_grad_norm = float(_grpo_max_grad_norm) if _grpo_max_grad_norm is not None else None
        FLAGS.config.num_minibatches = int(getattr(cfg, "grpo_num_minibatches", FLAGS.config.num_minibatches))
        if hasattr(cfg, "grpo_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.grpo_hidden_dims)
        FLAGS.config.actor_pretrain_steps = int(getattr(cfg, "grpo_actor_pretrain_steps", FLAGS.config.actor_pretrain_steps))
        FLAGS.config.rollout_length = int(getattr(cfg, "grpo_rollout_length", FLAGS.config.rollout_length))
        FLAGS.config.actor_log_std_min = float(getattr(cfg, "grpo_actor_log_std_min", FLAGS.config.actor_log_std_min))
    elif model_cls == "PPOResidualLearner":
        # Same fields as PPOLearner's own override block above, EXCEPT
        # actor_pretrain_steps -- there is no pretrain_actor_bc on this
        # class (no random-init-actor problem to warm-start away, see
        # ppo_residual_pi_config.py's docstring), so that field does not
        # exist on this config at all; setting it would error. edit_scale
        # is new here (unused by plain PPOLearner).
        FLAGS.config.actor_lr  = float(getattr(cfg, "ppo_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr = float(getattr(cfg, "ppo_lr", FLAGS.config.critic_lr))
        FLAGS.config.discount  = float(getattr(cfg, "ppo_discount", FLAGS.config.discount))
        FLAGS.config.gae_lambda        = float(getattr(cfg, "ppo_gae_lambda", FLAGS.config.gae_lambda))
        FLAGS.config.clip_eps          = float(getattr(cfg, "ppo_clip_eps", FLAGS.config.clip_eps))
        FLAGS.config.value_loss_coef   = float(getattr(cfg, "ppo_value_loss_coef", FLAGS.config.value_loss_coef))
        FLAGS.config.entropy_coef      = float(getattr(cfg, "ppo_entropy_coef", FLAGS.config.entropy_coef))
        _ppo_value_clip_eps = getattr(cfg, "ppo_value_clip_eps", FLAGS.config.value_clip_eps)
        FLAGS.config.value_clip_eps = float(_ppo_value_clip_eps) if _ppo_value_clip_eps is not None else None
        _ppo_max_grad_norm = getattr(cfg, "ppo_max_grad_norm", FLAGS.config.max_grad_norm)
        FLAGS.config.max_grad_norm = float(_ppo_max_grad_norm) if _ppo_max_grad_norm is not None else None
        FLAGS.config.num_minibatches = int(getattr(cfg, "ppo_num_minibatches", FLAGS.config.num_minibatches))
        if hasattr(cfg, "ppo_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.ppo_hidden_dims)
        FLAGS.config.rollout_length = int(getattr(cfg, "ppo_rollout_length", FLAGS.config.rollout_length))
        FLAGS.config.actor_log_std_min = float(getattr(cfg, "ppo_actor_log_std_min", FLAGS.config.actor_log_std_min))
        FLAGS.config.edit_scale = float(getattr(cfg, "ppo_edit_scale", FLAGS.config.edit_scale))
    elif model_cls == "GRPOResidualLearner":
        # Same fields as GRPOLearner's own override block above, EXCEPT
        # actor_pretrain_steps (same rationale as PPOResidualLearner's
        # branch just above), plus edit_scale.
        FLAGS.config.actor_lr     = float(getattr(cfg, "grpo_lr", FLAGS.config.actor_lr))
        FLAGS.config.group_size   = int(getattr(cfg, "grpo_group_size", FLAGS.config.group_size))
        FLAGS.config.clip_eps     = float(getattr(cfg, "grpo_clip_eps", FLAGS.config.clip_eps))
        FLAGS.config.kl_coef      = float(getattr(cfg, "grpo_kl_coef", FLAGS.config.kl_coef))
        FLAGS.config.entropy_coef = float(getattr(cfg, "grpo_entropy_coef", FLAGS.config.entropy_coef))
        _grpo_max_grad_norm = getattr(cfg, "grpo_max_grad_norm", FLAGS.config.max_grad_norm)
        FLAGS.config.max_grad_norm = float(_grpo_max_grad_norm) if _grpo_max_grad_norm is not None else None
        FLAGS.config.num_minibatches = int(getattr(cfg, "grpo_num_minibatches", FLAGS.config.num_minibatches))
        if hasattr(cfg, "grpo_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.grpo_hidden_dims)
        FLAGS.config.rollout_length = int(getattr(cfg, "grpo_rollout_length", FLAGS.config.rollout_length))
        FLAGS.config.actor_log_std_min = float(getattr(cfg, "grpo_actor_log_std_min", FLAGS.config.actor_log_std_min))
        FLAGS.config.edit_scale = float(getattr(cfg, "grpo_edit_scale", FLAGS.config.edit_scale))
    elif model_cls == "SACLearner":
        FLAGS.config.actor_lr  = float(getattr(cfg, "sac_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr = float(getattr(cfg, "sac_lr", FLAGS.config.critic_lr))
        FLAGS.config.discount  = float(getattr(cfg, "sac_discount", FLAGS.config.discount))
        FLAGS.config.tau       = float(getattr(cfg, "sac_tau", FLAGS.config.tau))
        FLAGS.config.init_temperature = float(getattr(cfg, "sac_init_temperature", FLAGS.config.init_temperature))
        _sac_target_entropy = getattr(cfg, "sac_target_entropy", FLAGS.config.target_entropy)
        FLAGS.config.target_entropy = float(_sac_target_entropy) if _sac_target_entropy is not None else None
        _sac_critic_weight_decay = getattr(cfg, "sac_critic_weight_decay", FLAGS.config.critic_weight_decay)
        FLAGS.config.critic_weight_decay = float(_sac_critic_weight_decay) if _sac_critic_weight_decay is not None else None
        FLAGS.config.num_qs = int(getattr(cfg, "sac_num_qs", FLAGS.config.num_qs))
        if hasattr(cfg, "sac_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.sac_hidden_dims)
    # BCLearner: no RL hyperparameter overrides needed — imitation-only,
    # no critic/GAE/advantage machinery to tune here.

    # Sync actor_success_only from the task YAML into the model config too —
    # BatchProcessor already reads it from cfg (line below, via train_pi_robo's
    # own actor_success_only variable), but the EXPOLearner agent itself reads
    # its own copy from FLAGS.config, which defaults to True in
    # expo_ft_pi_config.py. Without this sync, BatchProcessor (correctly seeing
    # the YAML's actor_success_only=False) never builds an actor_batch, while
    # the agent (still seeing FLAGS.config's default True) expects one — crash.
    FLAGS.config.actor_success_only = getattr(cfg, "actor_success_only", False)
    # AssetsConfig DROID officielle deja bakee dans la config openpi nommee ci-dessus —
    # ne pas l'ecraser ici (meme bug corrige dans eval_policy.py : le SFT a ete
    # entraine avec ces stats officielles, pas des stats locales par repo_id).
    # Mark the run directory as an RL run (e.g. stack_cube_expo_ft_2026-07-05_01-06-12_rl)
    # so it's visually distinguishable from an SFT run directory at a glance.
    run_dir, resuming = resolve_run_dir(cfg, resume_dir=cfg.rl_resume_dir, suffix="rl")

    # getattr with a default here since offline_ratio no longer exists at all
    # in the PPO/GRPO task YAMLs (removed — it was dead for them regardless,
    # see is_on_policy_algo below) whereas EXPOLearner/SACLearner/BCLearner
    # YAMLs still define it and this assert still validates their real value.
    assert 0.0 <= getattr(cfg, "offline_ratio", 0.0) <= 1.0

    if cfg.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {cfg.batch_size} must be divisible by "
            f"the number of devices {jax.device_count()}"
        )
    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = run_dir
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer — logs saved alongside checkpoints
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=cfg.keep_period,
        max_to_keep=getattr(cfg, "max_to_keep", 100),
        # A fresh run gets its own brand-new timestamped directory (see
        # resolve_run_dir) — safe to "overwrite" since it's empty. Only skip
        # this when we're genuinely resuming (cfg.rl_resume_dir set), so we never
        # risk wiping real checkpoints being resumed into.
        overwrite=not resuming,
        resume=resuming,
    )

    init_wandb(checkpoint_dir_path, resuming, cfg.project_name, cfg.run_name)
    wandb.config.update(vars(cfg), allow_val_change=resuming)

    if cfg.env_type in ('droid', 'sim'):
        dataset = process_droid_dataset(
            cfg.droid_format_dir,
            cfg,
            num_data=cfg.num_data_rl if cfg.num_data_rl > 0 else None,
        )
        example_action = dataset[0]['actions'][np.newaxis]
    else:
        raise ValueError(f"Unsupported env_type: {cfg.env_type}")

    # Load env wrapper dynamically from task config
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }

    logging.info("Creating environment...")
    if cfg.env_wrapper == "droid":
        from expo_ft.env.env_client import EnvClientWrapper
        env = EnvClientWrapper(
            env_creation_request=train_env_creation_request,
            host="localhost",
            port=8102,
        )
    else:
        from expo_ft.env.env_factory import make_env_wrapper
        env = make_env_wrapper(env_creation_request=train_env_creation_request, cfg=cfg)
    env.reset()
    logging.info(f"Created training environment {env.env_id}")

    # Separate environment instance for periodic rigorous (deterministic,
    # fixed-seed) evaluation -- see run_rigorous_eval() below. Kept
    # completely separate from the training env so a mid-episode training
    # rollout is never disturbed by an eval detour. Only created when the
    # feature is actually enabled (rl_eval_interval > 0), matching how
    # rl_critic_pretrain_steps/rl_actor_pretrain_steps etc. are opt-in.
    rl_eval_interval = int(getattr(cfg, "rl_eval_interval", 0) or 0)
    rl_eval_episodes = int(getattr(cfg, "rl_eval_episodes", 200))
    rl_eval_seed = int(getattr(cfg, "rl_eval_seed", 42))
    eval_env = None
    if rl_eval_interval > 0:
        eval_env_creation_request = {
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": None,
        }
        if cfg.env_wrapper == "droid":
            eval_env = EnvClientWrapper(
                env_creation_request=eval_env_creation_request,
                host="localhost",
                port=8102,
            )
        else:
            eval_env = make_env_wrapper(env_creation_request=eval_env_creation_request, cfg=cfg)
        eval_env.reset()
        logging.info(f"Created separate rigorous-eval environment {eval_env.env_id} "
                     f"(every {rl_eval_interval} steps, {rl_eval_episodes} episodes, "
                     f"deterministic actor, fixed seeds from master_seed={rl_eval_seed})")

    # model_cls already read near the top of main() (see hyperparameter
    # override block above) — reused here for the learner-import dispatch.
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearnerCategorical":
        # The categorical/distributional (XQC/XQCfD-style, C51-bounded
        # support) critic rewrite, kept in expo_ft_categorical.py — for
        # direct A/B comparison against the MSE scalar critic/REDQ-ensemble
        # architecture now used by the default "EXPOLearner". Shares the
        # exact same config overrides above (expo_ft_categorical.create()'s
        # signature ends in **kwargs, so the MSE/REDQ-specific fields set
        # there — num_qs, num_min_qs, critic_layer_norm, etc. — are silently
        # absorbed and ignored, not an error).
        from expo_ft.agents.alg.expo_ft_categorical import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "PPOLearner":
        from expo_ft.agents.alg.ppo import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "GRPOLearner":
        from expo_ft.agents.alg.grpo import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "PPOResidualLearner":
        # Same on-policy PPO machinery as PPOLearner, but trained on a
        # residual policy conditioned on the frozen VLA's own sampled base
        # action (via ExpoFT's own residual actor construction), instead of
        # a standalone actor with zero connection to it — see
        # ppo_residual.py's module docstring for the full rationale.
        from expo_ft.agents.alg.ppo_residual import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "GRPOResidualLearner":
        # GRPO counterpart of PPOResidualLearner — same residual mechanism,
        # group-relative advantage instead of GAE. See grpo_residual.py.
        from expo_ft.agents.alg.grpo_residual import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "SACLearner":
        from expo_ft.agents.alg.sac import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # PPO/GRPO are genuinely on-policy — their update() math (importance ratio
    # against old_log_probs, GAE bootstrap, GRPO's group-relative advantage
    # over CONTIGUOUS same-group rollouts) assumes every transition in a batch
    # was actually sampled from a known, recent version of the CURRENT policy.
    # Demo transitions (scripted motion-planning actions) were never sampled
    # from any version of the trained policy, so mixing them in — whether via
    # offline_ratio>0 (a separate offline buffer blended into every batch) or
    # via offline_ratio==0 (which instead seeds them permanently into the
    # ONLINE replay buffer, where uniform sampling would keep resurfacing them
    # indefinitely) — silently breaks that assumption and any group structure.
    # Force zero demo contamination for these four model classes (the plain
    # and residual variants alike), regardless of whatever offline_ratio
    # happens to be set to in the task YAML.
    is_on_policy_algo = model_cls in ("PPOLearner", "GRPOLearner", "PPOResidualLearner", "GRPOResidualLearner")
    # Residual variants additionally need the frozen VLA's own sampled base
    # action recorded per-transition (see ppo_residual.py/grpo_residual.py's
    # module docstrings): update() must recompute the residual's log_prob
    # against the EXACT base action used at rollout time, not a fresh
    # resample (pi0.5 is a stochastic flow-matching model, so a fresh sample
    # would generally differ from the one actually executed).
    is_residual_on_policy_algo = model_cls in ("PPOResidualLearner", "GRPOResidualLearner")
    # Number of consecutive transitions per on-policy rollout. Fixed (rather
    # than "however many the last episode happened to last") so the batch
    # shape stays constant across updates — a varying shape would force JAX to
    # recompile _update_jit on every single update. Unused by off-policy
    # learners.
    rollout_length = int(getattr(FLAGS.config, "rollout_length", 0) or 0)
    if is_on_policy_algo:
        if rollout_length <= 0:
            raise ValueError(
                f"model_cls={model_cls} is on-policy and requires rollout_length > 0 "
                f"(set ppo_rollout_length / grpo_rollout_length in the task YAML)"
            )
        _nmb = int(getattr(FLAGS.config, "num_minibatches", 1) or 1)
        if rollout_length % _nmb != 0:
            # _update_jit asserts this too, but only once inside the traced
            # update — failing here instead surfaces it before a long rollout
            # collection phase has already been spent.
            raise ValueError(
                f"rollout_length ({rollout_length}) must be divisible by "
                f"num_minibatches ({_nmb})"
            )
    if is_on_policy_algo and getattr(cfg, "offline_ratio", 0.0) != 0:
        logging.warning(
            "model_cls=%s is on-policy — ignoring offline_ratio=%s from the task "
            "YAML and forcing zero demo contamination (no dataset inserted into "
            "either replay buffer for actual training sampling).",
            model_cls, getattr(cfg, "offline_ratio", 0.0),
        )

    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, cfg.seed, mesh, data_sharding, replicated_sharding,
        resuming, env.task_description,
    )

    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=cfg.max_steps,
        task_description=env.task_description,
        replan_steps=cfg.replan_steps,
        seed=cfg.seed,
        store_base_actions=is_residual_on_policy_algo,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

    actor_success_only = getattr(cfg, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=cfg.batch_size,
        utd_ratio=cfg.utd_ratio,
        offline_ratio=0.0 if is_on_policy_algo else cfg.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=None if is_on_policy_algo else dataset,
        # Independent of offline_ratio's value -- see BatchProcessor's own
        # docstring/comment. Off (default) means genuinely no demos anywhere
        # unless offline_ratio > 0 also puts them in the offline buffer;
        # explicitly opt in via rl_seed_demos_online in the task YAML to
        # match the EXPO paper's own single-buffer convention instead.
        seed_demos_online=False if is_on_policy_algo else bool(getattr(cfg, "rl_seed_demos_online", False)),
        # PPO/GRPO need a contiguous, time-ordered rollout collected under the
        # current policy, not a uniform random draw over the whole buffer
        # history — see BatchProcessor's docstring. Off-policy learners keep
        # the previous behavior exactly (on_policy=False).
        on_policy=is_on_policy_algo,
        rollout_length=rollout_length if is_on_policy_algo else 0,
        replan_steps=cfg.replan_steps,
    )

    if is_on_policy_algo:
        # offline_replay_buffer still needs at least one transition inserted
        # so the shape-inference call just below (convert_to_critic_format on
        # offline_replay_buffer.dataset_dict[...]) has something to read.
        # This is ONLY for shape inference — BatchProcessor above got
        # dataset=None and offline_ratio=0.0, so it never samples from this
        # buffer for actual training batches; replay_buffer (the online one)
        # stays completely free of demo data too, filled only by genuine
        # on-policy rollout transitions collected from here on.
        offline_replay_buffer.insert_dataset(dataset)

    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
    {
        "base_image": offline_replay_buffer.dataset_dict['base_image'][0][np.newaxis],
        "left_wrist_image": offline_replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
        "state": offline_replay_buffer.dataset_dict['state'][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict['actions'][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent = load_agent(
        seed=cfg.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=cfg.replan_steps,
        default_prompt=env.task_description,
        residual_action_xyzg=getattr(cfg, 'residual_action_xyzg', False),
    )
    
    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    # Computed here (right after restore, while the restored buffer is
    # freshest) but only actually USED later at training_log._success_window's
    # own initialization, since training_log doesn't exist yet at this point.
    # See reconstruct_success_window()'s own docstring for why this exists.
    _restored_success_window = (
        reconstruct_success_window(batch_processor.replay_buffer, int(getattr(cfg, "success_rate_window", 200)))
        if resuming and cfg.checkpoint_buffer else []
    )

    # Demos live in offline_replay_buffer when offline_ratio > 0 (the normal
    # ExpoFT setup); BatchProcessor's constructor instead seeds them into the
    # online replay_buffer when offline_ratio == 0 — follow whichever buffer
    # actually received the dataset. Shared by both pretraining stages below.
    # Guarded by model_cls: offline_ratio/this whole pretraining mechanism is
    # ExpoFT-specific (EXPOLearner/EXPOLearnerCategorical) -- PPO/GRPO/SAC task YAMLs
    # don't define offline_ratio at all, so this must not run unconditionally.
    critic_pretrain_steps = int(getattr(FLAGS.config, "critic_pretrain_steps", 0) or 0)
    actor_bc_pretrain_steps = int(getattr(FLAGS.config, "actor_bc_pretrain_steps", 0) or 0)
    if model_cls in ("EXPOLearner", "EXPOLearnerCategorical"):
        pretrain_buffer = offline_replay_buffer if cfg.offline_ratio > 0 else replay_buffer
    else:
        pretrain_buffer = None

    # ── Discarded compile-only warm-up, resumed runs only ───────────────────
    # Why: agent.update()'s first-ever call (and thus first-ever JIT
    # compilation of the whole update pipeline) is otherwise gated behind
    # `training_log.ep_count >= 10` in the main loop below, which resets to 0
    # on every resume regardless of prior training -- see that check. On a
    # resumed run this delays first compilation ~700+ steps in, by which
    # point whatever compile-time peak memory that first compilation needs
    # has to coexist with everything accumulated in the meantime, on top of
    # the checkpoint restore's own ~18GB. This repeatedly produced
    # RESOURCE_EXHAUSTED crashes at that exact point (never on a fresh run,
    # which never carries the extra restore memory).
    #
    # Triggering that same compilation HERE instead -- immediately after
    # restore, before rollout collection has a chance to add anything else
    # to the picture -- is a genuine attempt at reducing what's
    # simultaneously resident at the moment of peak demand, not just
    # relocating the same peak. Both static branches of _update_finalize_jit
    # (use_success_batch True and False) are warmed deliberately, so neither
    # is deferred to whenever a real success episode first appears online.
    #
    # Deliberately does NOT depend on pretrain_buffer/demo data being
    # available anywhere -- robust to offline_ratio=0.0 + rl_seed_demos_online
    # =false (the genuinely-no-demos-anywhere config), where pretrain_buffer
    # is correctly empty by design, not by bug. Instead this samples directly
    # from `replay_buffer` (the online buffer, which always exists, is always
    # pre-allocated to capacity=cfg.max_steps >> cfg.batch_size, and is safe
    # to read at literal indices [0, batch_size) via sample_jax(indices=...),
    # which bypasses the usual "only sample within _size" restriction).
    #
    # Those specific indices are explicitly zeroed first for the CONTROL
    # fields (dones/masks/is_hil/hil_chunk/rewards) before sampling --
    # PiReplayBuffer.__init__ allocates these via np.empty, not np.zeros
    # (only is_success is zero-initialized), so reading them unwritten would
    # be uninitialized garbage, which risks corrupting _finalize_sample's
    # "next" index bookkeeping (n-step reward packing) rather than just
    # producing meaningless-but-safe numbers the way garbage image/state/
    # action data would. Image/state/action fields are left as whatever
    # np.empty gave them -- irrelevant here since the result is discarded
    # and no control-flow logic depends on their values.
    #
    # This never touches _size/_insert_index, so it can never be sampled by
    # real training later (normal sampling only draws from [0, _size), which
    # stays 0 here) and real inserts will naturally overwrite these same
    # slots first anyway (_insert_index starts at 0) -- zero contamination
    # risk to the real training trajectory.
    #
    # Wrapped in try/except: this is a best-effort optimization, not a
    # required step. If it fails for any reason, the run falls back to the
    # original behavior (compilation deferred to the ep_count gate as
    # before) rather than failing a run that might not even have hit the
    # crash this exists for.
    if resuming:
        try:
            logging.info("[warmup] Triggering discarded warm-up compilation of agent.update() "
                         "right after restore (both use_success_batch branches), before rollout "
                         "collection accumulates further memory pressure...")
            # agent.update() internally splits its input batch into utd_ratio
            # minibatches of cfg.batch_size each (see _prepare_minibatches_jit's
            # own assert total_bs % utd_ratio == 0) -- so the TOTAL batch this
            # call needs is cfg.batch_size * cfg.utd_ratio, matching exactly
            # how BatchProcessor.__init__ computes its own total_bs for the
            # real training iterators. Passing cfg.batch_size alone here
            # (24, not divisible by utd_ratio=20) is what the previous
            # attempt got wrong.
            _total_bs = cfg.batch_size * cfg.utd_ratio
            _warmup_n = _total_bs + cfg.replan_steps + 1  # safety margin for "next" index lookups
            for _field in ("dones", "masks", "is_hil", "hil_chunk", "rewards"):
                if _field in replay_buffer.dataset_dict:
                    replay_buffer.dataset_dict[_field][:_warmup_n] = 0
            # sample_jax() asserts len(self) >= self._replan_steps UNCONDITIONALLY,
            # before it even looks at `indices` -- with _size genuinely 0 (nothing
            # inserted yet on a resumed run), that assertion fails regardless of
            # indices being explicitly given. Temporarily report a large-enough
            # size just for this call, then restore the true value immediately
            # after -- consistent with directly poking dataset_dict above; _size
            # is what real training's own insert()/sample_jax() calls rely on
            # being accurate, so it must not be left altered.
            _true_size = replay_buffer._size
            replay_buffer._size = max(_true_size, _warmup_n)
            try:
                _warmup_indices = np.arange(_total_bs)
                # sample_jax's raw output still uses the buffer's own field
                # names (base_image, left_wrist_image, ...) -- agent.update()
                # expects the assembled "image"/"next_image" dict structure
                # instead. _convert_to_openpi_format does exactly this
                # conversion; apply_data_sharding is applied AFTER it (not
                # via sample_jax's own data_sharding= param), matching
                # next_batch()'s own proven on-policy-branch ordering exactly
                # rather than inventing a different one here.
                _warmup_raw = replay_buffer.sample_jax(_total_bs, indices=_warmup_indices)
                _warmup_batch = replay_buffer._convert_to_openpi_format(_warmup_raw)
                _warmup_batch = replay_buffer.apply_data_sharding(_warmup_batch, data_sharding)
            finally:
                replay_buffer._size = _true_size

            # agent's buffers may be donated inside update()'s own internal
            # JIT calls (see expo_ft.py's donate_argnames=("agent",) on
            # _prepare_minibatches_jit/_critic_update_step_jit/
            # _update_finalize_jit) -- reusing the SAME agent reference for
            # a second call after donation would be unsafe (its buffers may
            # already be reused/invalid). Chaining instead: the second warm-up
            # call uses the first call's own (still-discarded) result as its
            # input, so the original `agent` reference is only ever used
            # once, by the first call.
            # CRITICAL: donation must never touch the REAL, persistent
            # `agent` object -- it's reused for the rest of training
            # (sample_actions, etc.) right after this block. The earlier
            # "chain the second call through the first's result" fix only
            # protected the SECOND warmup call; it missed that the FIRST
            # call itself passes the real `agent` as the donated argument,
            # letting JAX free ITS buffers too. That's exactly what caused
            # "RuntimeError: Array has been deleted with shape=uint32[2]"
            # later in sample_actions -- agent.rng's buffer had already
            # been donated away during warmup. A genuine, independent
            # .copy() of every array leaf here ensures donation only ever
            # touches a throwaway copy, never the real agent.
            _warmup_agent = jax.tree_util.tree_map(
                lambda x: x.copy() if isinstance(x, jax.Array) else x, agent
            )
            _discard_agent, _discard_info = _warmup_agent.update(_warmup_agent, _warmup_batch, cfg.utd_ratio, None)
            jax.block_until_ready(_discard_agent)
            del _discard_info, _warmup_agent

            # Second pass warms the use_success_batch=True branch too (see
            # comment block above). That branch's actor_batch goes through
            # its OWN separate prepare_critic_batch call (see
            # _update_finalize_jit), NOT the utd_ratio-based minibatch
            # splitting _prepare_minibatches_jit does -- so it expects
            # cfg.batch_size directly, not total_bs. Sampled separately here
            # rather than reusing _warmup_batch, which is the wrong size for
            # this specific argument.
            _actor_warmup_indices = np.arange(cfg.batch_size)
            replay_buffer._size = max(_true_size, cfg.batch_size + cfg.replan_steps + 1)
            try:
                _actor_warmup_raw = replay_buffer.sample_jax(cfg.batch_size, indices=_actor_warmup_indices)
                _actor_warmup_batch = replay_buffer._convert_to_openpi_format(_actor_warmup_raw)
                _actor_warmup_batch = replay_buffer.apply_data_sharding(_actor_warmup_batch, data_sharding)
            finally:
                replay_buffer._size = _true_size
            # _discard_agent is already a throwaway copy's descendant (never
            # shared any buffers with the real, persistent agent to begin
            # with), so chaining through it here remains safe.
            _discard_agent2, _discard_info2 = _discard_agent.update(_discard_agent, _warmup_batch, cfg.utd_ratio, _actor_warmup_batch)
            jax.block_until_ready(_discard_agent2)
            del _discard_agent, _discard_agent2, _discard_info2, _actor_warmup_batch

            del _warmup_batch
            gc.collect()
            logging.info("[warmup] Compile-only warm-up complete; agent unchanged "
                         "(warm-up result discarded, never assigned; only the "
                         "original `agent` reference passed to update() -- the "
                         "chained second call never reused it after donation).")
        except Exception as e:
            logging.warning("[warmup] Warm-up compilation attempt failed (%s) -- continuing "
                           "with the original deferred-compilation behavior instead.", e)

    # ── Actor BC warm-start (XQCfD-style "policy pretraining") ──────────────
    # Only on a fresh run, same rationale as critic pretraining below. Runs
    # BEFORE critic pretraining — see pretrain_actor_bc()'s docstring for why
    # the ordering matters. EXPOLearner (MSE) doesn't have pretrain_actor_bc
    # (legacy baseline, frozen on purpose) — only EXPOLearnerCategorical
    # (the categorical-critic architecture) does.
    if not resuming and model_cls == "EXPOLearnerCategorical" and actor_bc_pretrain_steps > 0:
        logging.info(
            "BC warm-start: pretraining residual actor for %d steps on demo data (%s buffer)...",
            actor_bc_pretrain_steps,
            "offline" if cfg.offline_ratio > 0 else "online (offline_ratio=0, demos live there instead)",
        )
        bc_iterator = pretrain_buffer.get_iterator(
            sample_args={"batch_size": cfg.batch_size},
            data_sharding=data_sharding,
        )
        wandb.define_metric("bc_pretrain_step")
        wandb.define_metric("bc_pretrain/*", step_metric="bc_pretrain_step")
        for bc_step in tqdm.tqdm(
            range(actor_bc_pretrain_steps), desc="Actor BC warm-start", disable=not FLAGS.tqdm
        ):
            bc_batch = next(bc_iterator)
            bc_batch = pretrain_buffer.apply_data_sharding(bc_batch, data_sharding)
            bc_batch = dict(bc_batch)
            rng, key1 = jax.random.split(agent.rng)
            bc_batch["image"] = agent.data_augmentation_fn(key1, bc_batch["image"])
            # next_image left unaugmented on purpose: pretrain_actor_bc never
            # reads next_observations, but prepare_critic_batch still builds
            # that key from whatever's in next_image — passing it through raw
            # is harmless and saves an augmentation call.
            bc_batch = prepare_critic_batch(
                bc_batch,
                agent.actor.model_config.action_dim,
                agent.action_dim,
                agent.state_dim,
                agent.action_horizon,
                agent.replan_steps,
            )
            agent = agent.replace(rng=jax.device_put(rng, replicated_sharding))
            agent, bc_info = agent.pretrain_actor_bc(bc_batch)
            for k, v in bc_info.items():
                try:
                    tb_writer.add_scalar(f"bc_pretrain/{k}", float(v), global_step=bc_step)
                except (TypeError, ValueError):
                    pass
            wandb.log({
                "bc_pretrain_step": bc_step,
                **{f"bc_pretrain/{k}": v for k, v in bc_info.items()},
            })
        logging.info("Actor BC warm-start complete.")

    # ── Critic pretraining (XQCfD-style critic/actor coherence warmup) ──────
    # Only on a fresh run — a resumed run's critic has already been through
    # this (or through real online training), so redoing it here would be
    # meaningless and could even undo real progress.
    #
    # Trains the critic ONLY (residual actor, base VLA, and temperature are
    # left untouched) on offline demo data, using the exact same
    # update_critic() logic as normal training — same argmax-over-candidates
    # target computation, same masking, nothing algorithmically different.
    # The point is purely to get the critic roughly "coherent" with the SFT
    # policy's own actions before the online loop starts pulling the residual
    # policy toward whatever a still-randomly-initialized critic prefers.
    if not resuming and model_cls in ("EXPOLearner", "EXPOLearnerCategorical") and critic_pretrain_steps > 0:
        logging.info(
            "Pretraining critic for %d steps on demo data (%s buffer)...",
            critic_pretrain_steps,
            "offline" if cfg.offline_ratio > 0 else "online (offline_ratio=0, demos live there instead)",
        )
        pretrain_iterator = pretrain_buffer.get_iterator(
            sample_args={"batch_size": cfg.batch_size},
            data_sharding=data_sharding,
        )
        # Give pretrain/* metrics their own independent step axis
        # ("pretrain_step"), decoupled from the default global step counter
        # that the main training loop below uses for training/* metrics (via
        # wandb.log(..., step=i)). Without this, wandb treats "step" as one
        # single global, monotonically-increasing counter shared by every
        # metric in the run regardless of name — logging pretraining at
        # steps 0..N-1 (or any other scheme) would collide with, and get
        # silently dropped against, whatever the main loop logs afterward
        # (this is what caused the "steps must be monotonically increasing"
        # warnings that silently dropped every single pretrain data point).
        # TensorBoard doesn't share this problem (each tag is an independent
        # scalar stream), so no equivalent change is needed for tb_writer.
        wandb.define_metric("pretrain_step")
        wandb.define_metric("pretrain/*", step_metric="pretrain_step")
        for pretrain_step in tqdm.tqdm(
            range(critic_pretrain_steps), desc="Critic pretraining", disable=not FLAGS.tqdm
        ):
            pretrain_batch = next(pretrain_iterator)
            pretrain_batch = pretrain_buffer.apply_data_sharding(pretrain_batch, data_sharding)
            # update_critic() expects a batch already run through the same two
            # steps _update_jit() normally applies before ever calling it —
            # augmentation, then prepare_critic_batch() (raw "image"/
            # "next_image" -> the structured "observations"/"next_observations"
            # format update_critic actually reads). Skipping these is what
            # crashed the first version of this loop with a bare
            # KeyError('next_observations').
            pretrain_batch = dict(pretrain_batch)
            rng, key1 = jax.random.split(agent.rng)
            rng, key2 = jax.random.split(rng)
            pretrain_batch["image"] = agent.data_augmentation_fn(key1, pretrain_batch["image"])
            pretrain_batch["next_image"] = agent.data_augmentation_fn(key2, pretrain_batch["next_image"])
            pretrain_batch = prepare_critic_batch(
                pretrain_batch,
                agent.actor.model_config.action_dim,
                agent.action_dim,
                agent.state_dim,
                agent.action_horizon,
                agent.replan_steps,
            )
            agent = agent.replace(rng=jax.device_put(rng, replicated_sharding))
            agent, pretrain_info = agent.update_critic(pretrain_batch)
            for k, v in pretrain_info.items():
                try:
                    tb_writer.add_scalar(f"pretrain/{k}", float(v), global_step=pretrain_step)
                except (TypeError, ValueError):
                    pass
            # No step= kwarg here on purpose — the custom step_metric wiring
            # above means wandb plots these against "pretrain_step" (logged
            # in the same call) instead of the shared global step counter.
            wandb.log({
                "pretrain_step": pretrain_step,
                **{f"pretrain/{k}": v for k, v in pretrain_info.items()},
            })
        logging.info("Critic pretraining complete.")

    # ── Actor behavior-cloning pretraining (PPO/GRPO only) ──────────────────
    # Same rationale as the critic pretraining above, but for a different
    # gap: PPOLearner/GRPOLearner's actor is a separate, randomly-initialized
    # TanhNormal network (+ its own batch_encoder) — the loaded VLA is used
    # only for input preprocessing / output denormalization, never to
    # initialize this network's own weights (see ppo.py/grpo.py's
    # load_agent() docstrings). Without this warm-start, on-policy PPO/GRPO
    # starts from a random policy and reproduces this project's own Phase 1
    # finding (on-policy from a random/pre-SFT start = 0% success, no
    # learning signal) despite intending to test on-policy FROM an SFT
    # checkpoint.
    #
    # Only on a fresh run, same as critic pretraining. Trains actor +
    # batch_encoder ONLY (never the value network — nothing SFT-relevant for
    # it to imitate) via maximum-likelihood behavior cloning on offline demo
    # data, using pretrain_actor_bc() — not part of the PPO/GRPO objective
    # itself, purely a warm-start executed before it.
    actor_pretrain_steps = int(getattr(FLAGS.config, "actor_pretrain_steps", 0) or 0)
    if not resuming and model_cls in ("PPOLearner", "GRPOLearner") and actor_pretrain_steps > 0:
        # For PPO/GRPO, offline_replay_buffer always holds the demo dataset
        # regardless of cfg.offline_ratio (which is forced to 0 for these
        # on-policy algos and doesn't reflect where the dataset was
        # inserted — see the is_on_policy_algo block above, which explicitly
        # seeds offline_replay_buffer for shape-inference purposes; that
        # same data serves this pretraining loop).
        actor_pretrain_buffer = offline_replay_buffer
        logging.info("Pretraining actor (behavior cloning) for %d steps on demo data...", actor_pretrain_steps)
        actor_pretrain_iterator = actor_pretrain_buffer.get_iterator(
            sample_args={"batch_size": cfg.batch_size},
            data_sharding=data_sharding,
        )
        wandb.define_metric("actor_pretrain_step")
        wandb.define_metric("actor_pretrain/*", step_metric="actor_pretrain_step")
        for actor_pretrain_step in tqdm.tqdm(
            range(actor_pretrain_steps), desc="Actor BC pretraining", disable=not FLAGS.tqdm
        ):
            actor_pretrain_batch = next(actor_pretrain_iterator)
            actor_pretrain_batch = actor_pretrain_buffer.apply_data_sharding(actor_pretrain_batch, data_sharding)
            actor_pretrain_batch = dict(actor_pretrain_batch)
            rng, key1 = jax.random.split(agent.rng)
            rng, key2 = jax.random.split(rng)
            actor_pretrain_batch["image"] = agent.data_augmentation_fn(key1, actor_pretrain_batch["image"])
            actor_pretrain_batch["next_image"] = agent.data_augmentation_fn(key2, actor_pretrain_batch["next_image"])
            actor_pretrain_batch = prepare_critic_batch(
                actor_pretrain_batch,
                agent.vla.model_config.action_dim,
                agent.action_dim,
                agent.state_dim,
                agent.action_horizon,
                agent.replan_steps,
            )
            agent = agent.replace(rng=jax.device_put(rng, replicated_sharding))
            agent, actor_pretrain_info = agent.pretrain_actor_bc(actor_pretrain_batch)
            for k, v in actor_pretrain_info.items():
                try:
                    tb_writer.add_scalar(f"actor_pretrain/{k}", float(v), global_step=actor_pretrain_step)
                except (TypeError, ValueError):
                    pass
            wandb.log({
                "actor_pretrain_step": actor_pretrain_step,
                **{f"actor_pretrain/{k}": v for k, v in actor_pretrain_info.items()},
            })
        logging.info("Actor BC pretraining complete.")

    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    batch_processor.on_episode_start()

    dt = 1.0 / cfg.control_hz
    done = False
    env.reset()
    start_step_time = time.time()
    env.step(example_action.squeeze().tolist())
    action_plan = deque()
    action_type = "policy"
    episodes_since_update = 0
    combine_rng = jax.random.PRNGKey(cfg.seed + 100)

    def run_agent_updates(num_updates: int, metrics: dict):
        nonlocal agent, combine_rng
        for _ in range(num_updates):
            update_start = time.time()
            batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
            metrics["batch_info"] = get_batch_info(batch)
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
            with WC.phase("learner_update", thread="learner", update=WC.n_updates,
                          extra={"utd_ratio": cfg.utd_ratio, "batch_size": cfg.batch_size}):
                agent, update_info = agent.update(agent, batch, cfg.utd_ratio, actor_batch)
            WC.n_updates += cfg.utd_ratio  # agent.update runs utd_ratio gradient steps
            training_log.record_update_time(time.time() - update_start, metrics)
            for k, v in update_info.items():
                metrics[f"training/{k}"] = v

    # =====================================================================
    # Priming phase: evaluate the STARTING model on success_rate_window real
    # episodes, BEFORE any weight updates. Deliberately kept OUTSIDE
    # cfg.max_steps' budget -- has its own step counter here, but `i` and
    # everything keyed off it below (checkpoint naming, cfg.max_steps, the
    # loop bound itself) are completely untouched by this phase.
    #
    # wandb rejects going backward on its own internal step counter -- but
    # ALSO, tested directly (offline run, raw datastore inspection), NEVER
    # passing an explicit step= at all and instead using per-metric custom
    # step metrics (wandb.define_metric(..., step_metric=...)) sidesteps
    # that entirely: the internal counter still auto-increments harmlessly
    # in the background (nothing reads it), while each metric's own chart
    # x-axis uses whatever value we log alongside it under its custom step
    # key. This means training/eval metrics can genuinely start at 0 for
    # every run/task -- clean, directly comparable curves, no offset
    # arithmetic needed -- while eval/init_progress (priming) keeps its own
    # separate, real-step-based x-axis. Checkpoint saving/naming and
    # cfg.max_steps below use `i` directly, unaffected either way.
    #
    # Skipped entirely when resuming: the model already has real training
    # history; _success_window is a plain in-memory list (not part of the
    # checkpoint) so it isn't "empty" in any meaningful sense on resume,
    # there's just nothing to reconstruct it from -- re-priming would
    # needlessly redo real training time.
    wandb.define_metric("eval/init_step")
    wandb.define_metric("eval/init_progress", step_metric="eval/init_step")
    wandb.define_metric("training/global_step")
    wandb.define_metric("training/*", step_metric="training/global_step")
    wandb.define_metric("eval/success_rate", step_metric="training/global_step")
    # Deterministic, fixed-seed rigorous eval (Jesse's proposed protocol) --
    # deliberately a DIFFERENT metric name from eval/success_rate above
    # (that one is the in-training rolling window over stochastic online
    # episodes) so the two are never confused on the same dashboard.
    wandb.define_metric("eval_rigorous/success_rate", step_metric="training/global_step")
    wandb.define_metric("eval_rigorous/success_rate_stderr", step_metric="training/global_step")
    # Same protocol, but with n_edit_samples=0: the trained critic still
    # picks (argmax) among the N raw/un-edited VLA candidate samples, but the
    # residual actor is never invoked. Isolates how much of eval_rigorous's
    # improvement over the frozen-SFT baseline comes from the critic's
    # candidate selection alone, vs. the residual correction on top of it.
    wandb.define_metric("eval_rigorous/success_rate_critic_only", step_metric="training/global_step")
    wandb.define_metric("eval_rigorous/success_rate_critic_only_stderr", step_metric="training/global_step")

    success_rate_window = getattr(cfg, "success_rate_window", 200)
    training_log._success_window = _restored_success_window
    if _restored_success_window:
        logging.info(f"[resume] Reconstructed eval/success_rate's rolling window from "
                     f"{len(_restored_success_window)} restored episode(s) in the checkpointed "
                     f"replay buffer ({'full' if len(_restored_success_window) >= success_rate_window else 'partial'} "
                     f"window of {success_rate_window}) -- no need to wait for that many brand-new "
                     f"episodes post-resume before this metric reports again.")

    # ── Unified step-0 baseline pass (fresh runs only) ──────────────────────
    # One set of success_rate_window fixed-seed episodes, run on the TRAINING
    # env with only_base_actions=True (the true, clean frozen-SFT baseline --
    # no trained residual/critic exists yet at step 0, so this is the only
    # way to avoid biasing every later "did training improve over baseline"
    # comparison with an arbitrary untrained-residual contribution). This one
    # pass now serves three purposes at once, instead of two separate
    # 200-episode passes as before: (1) seeds the replay buffer with real
    # transitions via batch_processor.insert_transition, exactly as the old
    # priming phase did: (2) fills training_log._success_window, giving
    # eval/success_rate its own step-0 value; (3) gives eval_rigorous/
    # success_rate its step-0 value too, computed from the SAME episode
    # outcomes.
    #
    # Robust to eval_env being None (rl_eval_interval=0, rigorous eval
    # disabled entirely): this whole pass, and eval/success_rate's own
    # initialization, do NOT depend on eval_env or rl_eval_interval at
    # all -- only whether eval_rigorous/* also gets computed and logged
    # from the same data is conditional on that.
    #
    # Uses FIXED episode seeds (get_or_create_episode_seeds, the same
    # mechanism run_rigorous_eval's periodic calls use) rather than the
    # env's own natural reset randomness -- makes this step-0 pass
    # reproducible across different fresh runs of the same task (e.g.
    # comparing hyperparameter configs, all starting from the identical
    # 200 conditions), and is what makes deriving eval_rigorous/success_rate
    # from this same pass valid at all.
    if not resuming and success_rate_window > 0:
        step0_seeds = get_or_create_episode_seeds(checkpoint_dir_path, success_rate_window, rl_eval_seed)

        # Large fixed offset for this pass's own saved-transition step
        # numbers: guaranteed far beyond any realistic cfg.max_steps (real
        # training steps top out in the 1e5-1e6 range for these tasks), so
        # step-0 saves can never collide with real training-step saves in
        # the same buffers/ directory. Must stay non-negative: restore_replay_buffer
        # filters filenames with str.isdigit(), which is False for a
        # leading "-", so a negative offset would be silently excluded on
        # restore -- this bit us once already while designing this.
        STEP0_STEP_OFFSET = 10**9

        # Detect a step-0 pass that was interrupted by preemption: no agent
        # checkpoint exists yet (real training never started, hence
        # `resuming=False` above), but if checkpoint_buffer=true, some of
        # THIS pass's own transitions may already be saved to disk from a
        # previous attempt. Restore them and skip the seeds already
        # covered, instead of redoing the whole pass from scratch -- this
        # phase never updates the agent's own weights ("No agent.update(...)
        # call anywhere in this phase, by design", below), so there is
        # nothing to restore about the agent itself: the SFT checkpoint
        # freshly loaded via pi05_weight_loader_path already IS the correct,
        # unchanged starting point either way. Only the already-collected
        # transitions and episode outcomes are worth recovering.
        step0_already_done = 0
        step0_transitions_restored = 0
        if cfg.checkpoint_buffer:
            try:
                batch_processor.restore(checkpoint_dir_path)
                step0_transitions_restored = batch_processor.replay_buffer._size
                _restored_step0_window = reconstruct_success_window(batch_processor.replay_buffer, success_rate_window)
                if _restored_step0_window:
                    training_log._success_window = _restored_step0_window
                    step0_already_done = len(_restored_step0_window)
                    logging.info(f"[step-0] Found {step0_already_done} already-completed episode(s) "
                                 f"({step0_transitions_restored} raw transitions) saved from an interrupted "
                                 f"previous attempt at this same pass -- resuming from episode "
                                 f"{step0_already_done + 1}/{success_rate_window} instead of restarting "
                                 f"from scratch.")
            except Exception as e:
                logging.warning(f"[step-0] Could not check for an interrupted previous attempt "
                                 f"({e}) -- starting this pass from episode 1 as normal.")

        logging.info(f"[step-0] Evaluating the frozen SFT base policy (only_base_actions=True) "
                     f"on {success_rate_window} fixed-seed episodes -- seeds the replay buffer, "
                     f"initializes eval/success_rate's rolling window, and (if rl_eval_interval>0) "
                     f"gives eval_rigorous/success_rate its own step-0 value, all from this one pass...")
        # Resume the RAW-TRANSITION count from wherever the restored buffer
        # left off (0 on a genuinely fresh attempt) -- this is a small,
        # human/wandb-readable number (at most a few thousand for a
        # realistic success_rate_window), completely SEPARATE from
        # step0_file_step below. Keeping these two counters distinct is the
        # important part: priming_step is what gets logged to wandb
        # (eval/init_step, and indirectly eval/init_progress's own x-axis,
        # since it's that metric's declared step_metric) and must stay
        # small; step0_file_step is only ever used for on-disk filenames and
        # is the one that needs STEP0_STEP_OFFSET, to guarantee no collision
        # with the main training loop's own step-numbered files. An earlier
        # version of this fix used ONE variable for both purposes, which
        # correctly fixed the file-naming collision risk but also
        # accidentally sent eval/init_step (and eval/init_progress's x-axis)
        # up into the billions -- this split is the actual fix.
        priming_step = step0_transitions_restored
        step0_file_step = STEP0_STEP_OFFSET + step0_transitions_restored
        step0_episode_idx = step0_already_done
        priming_pbar = tqdm.tqdm(total=success_rate_window, initial=step0_already_done, desc="step-0 eval", disable=not FLAGS.tqdm)
        env.reset(seed=int(step0_seeds[step0_episode_idx]))

        while len(training_log._success_window) < success_rate_window:
            observation = env.get_observation()

            if not action_plan and action_type != "human":
                # only_base_actions=True: see this block's own comment above
                # for why -- the clean frozen-SFT baseline, not SFT plus an
                # arbitrary untrained-residual perturbation.
                action_chunk, agent, new_si = agent.sample_actions(observation, only_base_actions=True)
                episode_log.sample_info_history.append(new_si)
                action_plan.extend(action_chunk[:cfg.replan_steps])
            else:
                episode_log.sample_info_history.append(
                    episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                )

            elapsed = time.time() - start_step_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

            has_action = bool(action_plan)
            action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
            with WC.phase("eval", thread="actor", step=priming_step, extra={"priming": True}):
                real_action, action_type = env.step(action.tolist())
            start_step_time = time.time()
            done, success, reward, mask = env.get_info_for_step()
            priming_step += 1
            step0_file_step += 1

            episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

            if action_type == "human":
                action_plan.clear()

            if has_action or action_type == "human":
                _last_si = episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                transition_dict = dict(
                    observations=observation, actions=real_action, rewards=reward,
                    masks=mask, dones=done, is_hil=(action_type == "human"),
                )
                if _last_si and "base_action" in _last_si:
                    transition_dict["base_actions"] = _last_si["base_action"]
                batch_processor.insert_transition(transition_dict)
                # Persist this pass's own transitions to disk too (offset
                # step numbers, see STEP0_STEP_OFFSET above) -- without
                # this, a preemption mid-pass had no way to resume short of
                # redoing the whole 200 episodes, even with
                # checkpoint_buffer=true, since that flag previously only
                # covered the MAIN training loop's own saves.
                if cfg.checkpoint_buffer:
                    try:
                        save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=step0_file_step)
                    except Exception as e:
                        logging.error(f"[step-0] Could not save replay buffer transition: {e}")
            # No agent.update(...) call anywhere in this phase, by design.

            if done:
                batch_processor.on_episode_done(success)
                training_log.on_episode_done(episode_log, success, {})
                training_log._success_window.append(float(success))
                priming_pbar.update(1)
                wandb.log({
                    "eval/init_progress": len(training_log._success_window) / success_rate_window,
                    "eval/init_step": priming_step,
                })

                episode_log.reset()
                batch_processor.on_episode_start()
                step0_episode_idx += 1
                done = False
                action_type = "policy"
                action_plan.clear()
                # Always reset (leaves env ready either for this pass's next
                # fixed-seed episode, or for the main training loop right
                # after this block, which expects a valid post-reset
                # observation) -- seeded only while there's a next episode
                # of THIS pass still to run.
                if step0_episode_idx < success_rate_window:
                    env.reset(seed=int(step0_seeds[step0_episode_idx]))
                else:
                    env.reset()
                observation = env.get_observation()

        priming_pbar.close()
        wandb.log({"eval/init_progress": 1.0, "eval/init_step": priming_step})

        step0_success_rate = float(np.mean(training_log._success_window))
        step0_stderr = float(np.std(training_log._success_window, ddof=1) / np.sqrt(success_rate_window)) if success_rate_window > 1 else 0.0
        wandb.log({"eval/success_rate": step0_success_rate, "training/global_step": 0})
        if eval_env is not None:
            wandb.log({
                "eval_rigorous/success_rate": step0_success_rate,
                "eval_rigorous/success_rate_stderr": step0_stderr,
                "training/global_step": 0,
            })
            logging.info(f"[step-0] eval_rigorous/success_rate={step0_success_rate:.3f} +/- {step0_stderr:.3f} "
                         f"(derived from this same pass -- no separate rigorous-eval-only rollout needed).")

        logging.info(f"[step-0] Done after {priming_step} raw steps -- {success_rate_window} episodes "
                     f"evaluated, starting success_rate={step0_success_rate:.3f}. Real "
                     f"training begins now, budget (cfg.max_steps={cfg.max_steps}) untouched by this phase. "
                     f"eval/success_rate and training/* now use training/global_step as their own x-axis, "
                     f"starting fresh at 0 -- checkpoint step "
                     f"numbers (i) are NOT offset and remain exactly comparable across runs.")

    WC.scalar("training_start_t", WC.elapsed(), step=start_step)
    for i in tqdm.tqdm(
        range(start_step, cfg.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        loop_start = time.time()
        step_metrics = {}

        observation = env.get_observation()
        # NOTE: done/success/reward/mask are deliberately NOT fetched here.
        # They must reflect the CONSEQUENCE of the action taken THIS
        # iteration (env.step() below), not the previous iteration's action
        # — fetching them here (before env.step()) was storing every
        # transition's reward/done/mask one step out of phase with its own
        # (observation, action) pair: transition i would get
        # (o_i, a_i, r_{i-1}, done_{i-1}) instead of (o_i, a_i, r_i, done_i).
        # This also delayed episode-boundary detection (the `if done:` reset
        # check below) by one step, letting one extra action execute past
        # the true terminal state before resetting.

        # Skip model inference while human is controlling.
        if not action_plan and action_type != "human":
            sample_start = time.time()
            with WC.phase("inference", thread="actor", step=i, extra={"replan_steps": cfg.replan_steps}):
                action_chunk, agent, new_si = agent.sample_actions(observation)
            episode_log.sample_info_history.append(new_si)
            training_log.record_sample_time(time.time() - sample_start, step_metrics)
            action_plan.extend(action_chunk[:cfg.replan_steps])
        else:
            episode_log.sample_info_history.append(episode_log.sample_info_history[-1] if episode_log.sample_info_history else None)

        elapsed = time.time() - start_step_time
        if elapsed < dt:
            with WC.phase("robot_idle", thread="actor", step=i):
                time.sleep(dt - elapsed)
        else:
            # Missed the control deadline: in sync mode this is the learner stalling the robot.
            WC.scalar("deadline_overrun_s", elapsed - dt, step=i)

        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
        with WC.phase("robot_step", thread="actor", step=i):
            real_action, action_type = env.step(action.tolist())
        start_step_time = time.time()
        # Fetch AFTER env.step(): now reflects the consequence of
        # `real_action` taken from `observation`, matching the (o_i, a_i,
        # r_i, done_i) convention the rest of the pipeline (Bellman backup,
        # GAE, ...) assumes.
        done, success, reward, mask = env.get_info_for_step()

        episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

        if action_type == "human":
            action_plan.clear()

        if has_action or action_type == "human":
            _last_si = episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
            transition_dict = dict(
                observations=observation,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),
            )
            if _last_si and "base_action" in _last_si:
                transition_dict["base_actions"] = _last_si["base_action"]
            batch_processor.insert_transition(transition_dict)
        
        can_update = training_log.ep_count >= 10 and i >= cfg.batch_size
        # On-policy (PPO/GRPO): update exactly when one full fresh rollout has
        # been collected, then discard it — NOT on the step/episode/batch
        # cadence below, which pairs with uniform random replay sampling and
        # would here either update on stale data or on a variable-length
        # (recompilation-triggering) batch. cfg.update_type/num_updates are
        # deliberately ignored in this branch.
        if is_on_policy_algo:
            if can_update and batch_processor.rollout_ready():
                run_agent_updates(1, step_metrics)
        elif cfg.update_type == "step" and can_update:
            run_agent_updates(cfg.num_updates, step_metrics)

        if done:
            batch_processor.on_episode_done(success)
            WC.scalar("episode_success", float(success), step=i)
            with WC.phase("reset", thread="actor", step=i):
                env.reset()

            if is_on_policy_algo:
                pass  # rollout-driven, handled above
            elif cfg.update_type == "episode" and can_update:
                for _ in tqdm.tqdm(range(cfg.num_updates)):
                    run_agent_updates(1, step_metrics)
            elif cfg.update_type == "batch" and can_update:
                episodes_since_update += 1
                if episodes_since_update >= cfg.num_batch:
                    for _ in tqdm.tqdm(range(cfg.num_updates)):
                        run_agent_updates(1, step_metrics)
                    episodes_since_update = 0

            training_log.on_episode_done(episode_log, success, step_metrics)
            step_metrics["training/episode_count"] = training_log.ep_count
            
            # Rolling success rate over the last success_rate_window episodes
            # (YAML-driven, computed once before the priming phase above --
            # shared with it so both stay in sync). Nothing is logged under
            # eval/success_rate at all until the window is genuinely full: a
            # partial window (e.g. after 1 episode) can ONLY read exactly
            # 0.0 or 1.0 regardless of the true underlying rate -- not a
            # calculation bug, just what a 1-sample mean is mathematically
            # forced to be. In practice the window is already full by the
            # time this loop starts (see the priming phase above), except
            # right when resuming a run whose in-memory window was never
            # reconstructed -- this guard still protects that case too.
            training_log._success_window.append(float(success))
            if len(training_log._success_window) > success_rate_window:
                training_log._success_window.pop(0)
            if len(training_log._success_window) >= success_rate_window:
                step_metrics["eval/success_rate"] = np.mean(training_log._success_window)
                WC.scalar("eval/success_rate", step_metrics["eval/success_rate"], step=i)
            
            episode_log.reset()
            batch_processor.on_episode_start()

            observation = env.get_observation()
            done = False
            action_type = "policy"
            action_plan.clear()

        if cfg.checkpoint_model and cfg.checkpoint_interval > 0 and i > 0 and i % cfg.checkpoint_interval == 0:
            try:
                with WC.phase("other", thread="learner", step=i, extra={"what": "checkpoint"}):
                    save_checkpoint(checkpoint_manager, agent, i)
                # save_checkpoint() only blocks for the synchronous portion
                # of the save (per Orbax's own docs: "Finished blocking save
                # in N seconds. Continuing to save asynchronously...") --
                # the background thread doing the actual host transfer can
                # still be running, and still using GPU memory for that
                # transfer, well after this call returns and this log line
                # prints. That transfer memory being unexpectedly still
                # resident is a plausible explanation for the OOM crashes
                # observed shortly after a checkpoint save reports complete.
                # wait_until_finished() is Orbax's own documented API for
                # genuinely blocking until that background thread is done
                # ("Blocks until any incomplete save operations are
                # completed... will wait until each of these checkpointers
                # is finished") -- calling it here, then gc.collect()
                # immediately after, gives that transfer memory a real
                # chance to be released before the next training step needs
                # its own peak memory.
                checkpoint_manager.wait_until_finished()
                gc.collect()
                logging.info(f"Saved agent checkpoint at step {i} (interval={cfg.checkpoint_interval})")
            except Exception as e:
                logging.error(f"Could not save model checkpoint: {e}")

        # Periodic rigorous deterministic eval (Jesse's proposed protocol) --
        # see run_rigorous_eval()'s own docstring for why this is safe to
        # call mid-training: it never touches the real `agent`/its .rng
        # sequence, only reads it. i>0 skips step 0 here since that's
        # already covered by the dedicated initialization eval above (fresh
        # runs) -- on a resumed run there was no such initialization call,
        # but the periodic boundary will still be hit at the next multiple
        # of rl_eval_interval regardless.
        if eval_env is not None and i > 0 and i % rl_eval_interval == 0:
            try:
                episode_seeds = get_or_create_episode_seeds(checkpoint_dir_path, rl_eval_episodes, rl_eval_seed)
                with WC.phase("eval", thread="actor", step=i, extra={"rigorous": True}):
                    success_rate, stderr = run_rigorous_eval(agent, eval_env, episode_seeds, cfg)
                WC.scalar("eval_rigorous/success_rate", success_rate, step=i)
                wandb.log({
                    "eval_rigorous/success_rate": success_rate,
                    "eval_rigorous/success_rate_stderr": stderr,
                    "training/global_step": i,
                })
                logging.info(f"[rigorous-eval] step {i}: success_rate={success_rate:.3f} +/- {stderr:.3f}")
            except Exception as e:
                logging.error(f"[rigorous-eval] Failed at step {i}: {e}")

            # Critic-only variant: agent.replace() is a cheap struct copy (n_edit_samples
            # is a static, non-pytree field) -- the real `agent` below is completely
            # unaffected, same as run_rigorous_eval's own non-mutation guarantee above.
            # Same episode_seeds as the call above, for a like-for-like comparison.
            try:
                critic_only_agent = agent.replace(n_edit_samples=0)
                critic_only_success_rate, critic_only_stderr = run_rigorous_eval(
                    critic_only_agent, eval_env, episode_seeds, cfg
                )
                wandb.log({
                    "eval_rigorous/success_rate_critic_only": critic_only_success_rate,
                    "eval_rigorous/success_rate_critic_only_stderr": critic_only_stderr,
                    "training/global_step": i,
                })
                logging.info(
                    f"[rigorous-eval-critic-only] step {i}: "
                    f"success_rate={critic_only_success_rate:.3f} +/- {critic_only_stderr:.3f}"
                )
            except Exception as e:
                logging.error(f"[rigorous-eval-critic-only] Failed at step {i}: {e}")

        if cfg.checkpoint_buffer and (has_action or action_type == "human"):
            try:
                save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=i)
            except Exception:
                logging.exception("Could not save agent buffer.")

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        step_metrics["training/global_step"] = i

        # TensorBoard logging — convert to float() to handle JAX scalars
        # (jnp.float32 etc.) which don't pass isinstance(v, (int, float))
        for k, v in step_metrics.items():
            try:
                tb_writer.add_scalar(k, float(v), global_step=i)
            except (TypeError, ValueError):
                pass  # skip non-scalar values (e.g. batch_info dicts)
        # No explicit step= here -- training/global_step (defined above via
        # wandb.define_metric) is what every training/eval chart actually
        # uses for its x-axis, starting fresh at 0 regardless of how long
        # priming took. See this file's priming-phase comment for why this
        # avoids the step-monotonicity data-loss issue entirely.
        wandb.log(step_metrics)
    
    if cfg.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, cfg.max_steps)
            logging.info(f"Saved final agent checkpoint at step {cfg.max_steps}")
        except Exception as e:
            logging.error(f"Could not save final checkpoint: {e}")
        logging.info("Waiting for checkpoint manager to finish")
        tb_writer.close()
        checkpoint_manager.wait_until_finished()
        # NOTE: the automatic end-of-training eval_curve.py sweep (writing a
        # handoff file for job_rl.sh to pick up) has been removed here --
        # eval/success_rate is now primed from step 1 (see the priming phase
        # above) and trustworthy throughout, so an automatic full 200-seed
        # sweep after every single run is no longer needed by default. The
        # scripts themselves (expo_ft/utils/eval_curve_runner.py,
        # scripts/run_eval_curve_from_handoff.py, scripts/eval_curve.py) are
        # untouched -- run scripts/eval_curve.py directly (see job_eval_curve.sh)
        # whenever a rigorous fixed-seed sweep is specifically wanted.


if __name__ == "__main__":
    app.run(main)
