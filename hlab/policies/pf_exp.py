import argparse

from habitat.config import Config

from semexp.model_pf import RL_Policy

from policies.policy_registry import policy_registry


@policy_registry.register_policy
class PFExp(RL_Policy):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.prev_maps = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: Config):
        gcfg = cfg.GLOBAL_AGENT
        pcfg = cfg.PF_EXP_POLICY
        # Sanity check
        assert pcfg.pf_model_path != ""
        # Create arguments
        args = argparse.Namespace()
        args.use_egocentric_transform = pcfg.use_egocentric_transform
        args.add_agent2loc_distance = pcfg.add_agent2loc_distance
        args.add_agent2loc_distance_v2 = pcfg.add_agent2loc_distance_v2
        args.pf_masking_opt = pcfg.pf_masking_opt
        args.mask_nearest_locations = pcfg.mask_nearest_locations
        args.mask_size = pcfg.mask_size
        args.visualize = gcfg.visualize
        args.print_images = gcfg.visualize
        args.map_resolution = gcfg.map_resolution
        args.area_weight_coef = pcfg.area_weight_coef
        args.dist_weight_coef = pcfg.dist_weight_coef
        args.map_size_cm = cfg.PLANNER.map_size_cm

        return cls(cfg, args, pcfg.pf_model_path)

    def load_checkpoint(self):
        path = self.cfg.PF_EXP_POLICY.pf_model_path
        if path != "":
            print(f"=====> PFExp: Loading checkpoint from {path}")
        else:
            print(f"=====> PFExp: No pretrained weights available.")

    def act(
        self,
        inputs,
        rnn_hxs,
        masks,
        extras=None,
        deterministic=False,
        extra_maps=None,
    ):
        outputs = super().act(
            inputs,
            rnn_hxs,
            masks,
            extras=extras,
            deterministic=deterministic,
            extra_maps=extra_maps,
        )
        value, action, action_log_probs, rnn_hxs, all_maps = outputs
        self.prev_maps = all_maps
        return value, action, action_log_probs, rnn_hxs

    @property
    def needs_dist_maps(self):
        return (
            self.cfg.PF_EXP_POLICY.add_agent2loc_distance
            or self.cfg.PF_EXP_POLICY.add_agent2loc_distance_v2
        )

    @property
    def needs_unexplored_maps(self):
        return self.cfg.PF_EXP_POLICY.pf_masking_opt == "unexplored"

    @property
    def needs_frontier_maps(self):
        return False

    # Note: RL_Policy already implements needs_egocentric_transform
