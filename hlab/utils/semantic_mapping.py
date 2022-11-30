from argparse import Namespace

# Relies on SemExp codebase
from semexp.model import Semantic_Mapping as Orig_Semantic_Mapping


class Semantic_Mapping(Orig_Semantic_Mapping):
    @classmethod
    def from_config(cls, cfg, device):
        args = Namespace()
        args.device = device
        args.frame_height = cfg.GLOBAL_AGENT.frame_height
        args.frame_width = cfg.GLOBAL_AGENT.frame_width
        args.map_resolution = cfg.GLOBAL_AGENT.map_resolution
        args.map_size_cm = cfg.GLOBAL_AGENT.map_size_cm
        args.camera_height = cfg.GLOBAL_AGENT.camera_height
        args.global_downscaling = cfg.GLOBAL_AGENT.global_downscaling
        args.vision_range = cfg.SEMANTIC_MAPPING.vision_range
        args.hfov = float(cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV)
        args.du_scale = cfg.SEMANTIC_MAPPING.du_scale
        args.cat_pred_threshold = cfg.SEMANTIC_MAPPING.cat_pred_threshold
        args.exp_pred_threshold = cfg.SEMANTIC_MAPPING.exp_pred_threshold
        args.map_pred_threshold = cfg.SEMANTIC_MAPPING.map_pred_threshold
        args.num_sem_categories = cfg.GLOBAL_AGENT.num_sem_categories
        args.num_processes = cfg.NUM_ENVIRONMENTS

        return cls(args)
