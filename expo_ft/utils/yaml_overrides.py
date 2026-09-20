"""Apply task-YAML RL overrides to an ml_collections model config (EXPOLearner family).

Verbatim copy of the EXPOLearner branch in train_pi_robo.py::main so that
train_pi_robo_async.py can consume the same ManiSkill task YAMLs. If you edit
one, edit the other (or refactor the sync trainer to call this).
"""

from __future__ import annotations


def apply_expo_yaml_overrides(cfg, config) -> None:
    """cfg: SimpleNamespace from load_task_config; config: FLAGS.config (ml_collections)."""
    model_cls = config.model_cls
    if model_cls in ("EXPOLearner", "EXPOLearnerCategorical"):
        # --- unchanged from before this refactor: byte-for-byte identical ---
        config.actor_lr         = float(getattr(cfg, "rl_lr", config.actor_lr))
        config.critic_lr        = float(getattr(cfg, "rl_lr", config.critic_lr))
        # NOTE: previously orphaned -- temp_lr silently stayed at whatever
        # config.temp_lr (3e-4, from sac_config.py) already was,
        # decoupled from rl_lr above and with no YAML field to override it
        # independently. Wired here the same way as the other rl_* fields;
        # default is unchanged (falls back to config.temp_lr) for any
        # YAML that doesn't set rl_temp_lr yet.
        config.temp_lr          = float(getattr(cfg, "rl_temp_lr", config.temp_lr))
        config.discount         = float(getattr(cfg, "rl_discount", config.discount))
        config.tau              = float(getattr(cfg, "rl_tau", config.tau))
        config.init_temperature = float(getattr(cfg, "rl_init_temperature", config.init_temperature))
        config.adjust_target_entropy = getattr(cfg, "rl_adjust_target_entropy", config.adjust_target_entropy)
        _rl_fixed_temperature = getattr(cfg, "rl_fixed_temperature", config.fixed_temperature)
        config.fixed_temperature = float(_rl_fixed_temperature) if _rl_fixed_temperature is not None else None
        _rl_critic_weight_decay = getattr(cfg, "rl_critic_weight_decay", config.critic_weight_decay)
        config.critic_weight_decay = float(_rl_critic_weight_decay) if _rl_critic_weight_decay is not None else None
        _rl_critic_grad_clip_norm = getattr(cfg, "rl_critic_grad_clip_norm", config.critic_grad_clip_norm)
        config.critic_grad_clip_norm = float(_rl_critic_grad_clip_norm) if _rl_critic_grad_clip_norm is not None else None
        config.freeze_critic_encoder = getattr(cfg, "rl_freeze_critic_encoder", config.freeze_critic_encoder)
        if hasattr(cfg, "rl_hidden_dims"):
            config.hidden_dims  = tuple(cfg.rl_hidden_dims)
        config.edit_scale       = float(getattr(cfg, "rl_edit_scale", config.edit_scale))
        config.N = int(getattr(cfg, "rl_N", config.N))
        config.n_edit_samples = int(getattr(cfg, "rl_n_edit_samples", config.n_edit_samples))
        # --- end of original ExpoFT block; critic_pretrain_steps added below is
        # a new, default-off (0) field — behavior for existing configs that
        # don't set rl_critic_pretrain_steps is unchanged ---
        config.critic_pretrain_steps = int(getattr(cfg, "rl_critic_pretrain_steps", config.critic_pretrain_steps))
        config.actor_bc_pretrain_steps = int(getattr(cfg, "rl_actor_bc_pretrain_steps", config.actor_bc_pretrain_steps))
        config.num_atoms = int(getattr(cfg, "rl_num_atoms", config.num_atoms))
        config.v_min = float(getattr(cfg, "rl_v_min", config.v_min))
        config.v_max = float(getattr(cfg, "rl_v_max", config.v_max))
        config.reward_scale_decay = float(getattr(cfg, "rl_reward_scale_decay", config.reward_scale_decay))
        config.use_reward_normalization = bool(getattr(cfg, "rl_use_reward_normalization", config.use_reward_normalization))
        config.kl_coef = float(getattr(cfg, "rl_kl_coef", config.kl_coef))
        config.entropy_scale = float(getattr(cfg, "rl_entropy_scale", config.entropy_scale))
        config.kl_ref_std = float(getattr(cfg, "rl_kl_ref_std", config.kl_ref_std))
        config.use_hetstat_policy = bool(getattr(cfg, "rl_use_hetstat_policy", config.use_hetstat_policy))
        config.hetstat_num_rff_features = int(getattr(cfg, "rl_hetstat_num_rff_features", config.hetstat_num_rff_features))
        config.hetstat_var_lr_multiplier = float(getattr(cfg, "rl_hetstat_var_lr_multiplier", config.hetstat_var_lr_multiplier))
        config.use_double_q_selection = bool(getattr(cfg, "rl_use_double_q_selection", config.use_double_q_selection))
        config.use_clipped_double_q = bool(getattr(cfg, "rl_use_clipped_double_q", config.use_clipped_double_q))
    config.actor_success_only = getattr(cfg, "actor_success_only", False)
