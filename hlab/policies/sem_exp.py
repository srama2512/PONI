import gym
import numpy as np
import torch
import torch.nn as nn
from habitat.config import Config

# Relies on SemExp codebase
from semexp.model import RL_Policy

from policies.policy_registry import policy_registry


@policy_registry.register_policy
class SemExp(RL_Policy):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: Config):
        gcfg = cfg.GLOBAL_AGENT
        map_size = gcfg.map_size_cm // gcfg.map_resolution
        full_w, full_h = map_size, map_size
        local_w = int(full_w / gcfg.global_downscaling)
        local_h = int(full_h / gcfg.global_downscaling)

        ngc = 8 + cfg.GLOBAL_AGENT.num_sem_categories
        es = 2
        obs_space = gym.spaces.Box(0, 1, (ngc, local_w, local_h), dtype="uint8")
        act_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)

        return cls(
            cfg,
            obs_space.shape,
            act_space,
            model_type=1,
            base_kwargs={
                "recurrent": cfg.SEM_EXP_POLICY.use_recurrent_global,
                "hidden_size": cfg.SEM_EXP_POLICY.global_hidden_size,
                "num_sem_categories": cfg.GLOBAL_AGENT.num_sem_categories,
                "main_model": cfg.SEM_EXP_POLICY.main_model,
            },
        )

    def load_checkpoint(self):
        path = self.cfg.SEM_EXP_POLICY.pretrained_weights
        if path != "":
            print(f"=====> SemExp: Loading checkpoint from {path}")
            loaded_state = torch.load(path, map_location="cpu")
            self.load_state_dict(loaded_state)
        else:
            print(f"=====> SemExp: No pretrained weights available.")

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False, **kwargs):
        outputs = super().act(
            inputs, rnn_hxs, masks, extras=extras, deterministic=deterministic
        )
        value, action, action_log_probs, rnn_hxs = outputs
        action = nn.Sigmoid()(action)
        return value, action, action_log_probs, rnn_hxs

    @property
    def needs_dist_maps(self):
        return False

    @property
    def needs_unexplored_maps(self):
        return False

    @property
    def needs_egocentric_transform(self):
        return False

    @property
    def needs_frontier_maps(self):
        return False

    @property
    def has_action_output(self):
        return False
