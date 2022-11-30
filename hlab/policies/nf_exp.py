import copy

import torch.nn as nn
from habitat.config import Config

from policies.policy_registry import policy_registry


@policy_registry.register_policy
class NearestFrontierExp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: Config):
        gcfg = cfg.GLOBAL_AGENT
        return cls(cfg)

    def load_checkpoint(self):
        pass

    def act(
        self,
        inputs,
        rnn_hxs,
        masks,
        extras=None,
        deterministic=False,
        extra_maps=None,
    ):
        B, _, H, W = inputs.shape
        gcfg = self.cfg.GLOBAL_AGENT
        ncfg = self.cfg.NF_EXP_POLICY

        assert extra_maps is not None

        frontiers = extra_maps["fmap"].unsqueeze(1)  # (B, 1, H, W)
        action = frontiers
        agent_locs = extra_maps["agent_locations"]

        if ncfg.mask_nearest_locations:
            for i in range(B):
                ri, ci = agent_locs[i]
                size = int(ncfg.mask_size * 100.0 / gcfg.map_resolution)
                local_frontier = copy.deepcopy(
                    frontiers[
                        i, :, ri - size : ri + size + 1, ci - size : ci + size + 1
                    ]
                )
                frontiers[
                    i, :, ri - size : ri + size + 1, ci - size : ci + size + 1
                ] = 0
                # If masking leads to zero frontier sum, then restore
                if frontiers[i].sum() == 0:
                    frontiers[
                        i,
                        :,
                        ri - size : ri + size + 1,
                        ci - size : ci + size + 1,
                    ] = local_frontier

        return None, action, None, rnn_hxs

    @property
    def needs_dist_maps(self):
        return False

    @property
    def needs_frontier_maps(self):
        return True

    @property
    def needs_unexplored_maps(self):
        return False

    @property
    def needs_egocentric_transform(self):
        return False

    @property
    def has_action_output(self):
        return False
