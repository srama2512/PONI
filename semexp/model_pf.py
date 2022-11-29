import cv2
import numpy as np

import poni.geometry as pgeo
import torch
import torch.nn as nn
from einops import asnumpy, rearrange, repeat
from poni.dataset import SemanticMapDataset as PFDataset
from poni.default import get_cfg
from torch.nn import functional as F
from train import SemanticMapperModule as PFModel


class Potential_Function_Semantic_Policy(nn.Module):
    def __init__(self, pf_model_path):
        super().__init__()

        loaded_state = torch.load(pf_model_path, map_location="cpu")
        pf_model_cfg = get_cfg()
        pf_model_cfg.merge_from_other_cfg(loaded_state["cfg"])
        self.pf_model = PFModel(pf_model_cfg)
        # Remove dataparallel modules
        state_dict = {
            k.replace(".module", ""): v for k, v in loaded_state["state_dict"].items()
        }
        self.pf_model.load_state_dict(state_dict)
        self.eval()

    def forward(self, inputs, rnn_hxs, masks, extras):
        # inputs - (bs, N, H, W)
        # x_pf - (bs, N, H, W), x_a - (bs, 1, H, W)
        x_pf, x_a = self.pf_model.infer(inputs, avg_preds=False)
        return x_pf, x_a

    def add_agent_dists_to_object_dists(self, pfs, agent_dists):
        # pfs - (B, N, H, W)
        # agent_dists - (B, H, W)
        object_dists = self.convert_object_pf_to_distance(pfs)
        agent2obj_dists = agent_dists.unsqueeze(1) + object_dists
        # Convert back to pf
        return self.pf_model.convert_distance_to_pf(agent2obj_dists)

    def convert_object_pf_to_distance(self, pfs):
        return self.pf_model.convert_object_pf_to_distance(pfs)

    def convert_distance_to_pf(self, dists):
        return self.pf_model.convert_distance_to_pf(dists)

    @property
    def cfg(self):
        return self.pf_model.cfg


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):
    def __init__(self, args, pf_model_path):

        super(RL_Policy, self).__init__()

        self.args = args
        self.network = Potential_Function_Semantic_Policy(pf_model_path)
        self._cached_visualizations = None

    @property
    def is_recurrent(self):
        return False

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return 10  # Some random value

    @property
    def needs_egocentric_transform(self):
        cfg = self.network.pf_model.cfg
        output_type = "map"
        if hasattr(cfg.MODEL, "output_type"):
            output_type = cfg.MODEL.output_type
        return (
            output_type in ["dirs", "locs", "acts"]
        ) or self.args.use_egocentric_transform

    @property
    def has_action_output(self):
        cfg = self.network.pf_model.cfg
        return cfg.MODEL.output_type == "acts"

    def get_pf_cfg(self):
        return self.network.pf_model.get_pf_cfg()

    def forward(self, inputs, rnn_hxs, masks, extras):
        raise NotImplementedError

    def act(
        self, inputs, rnn_hxs, masks, extras=None, extra_maps=None, deterministic=False
    ):

        assert extra_maps is not None
        value = torch.zeros(inputs.shape[0], device=inputs.device)
        action_log_probs = torch.zeros(inputs.shape[0], device=inputs.device)
        # Convert inputs to appropriate format for self.network
        proc_inputs = self.do_proc(inputs)  # (B, N, H, W)

        # Perform egocentric transform if needed
        B, _, H, W = proc_inputs.shape
        t_ego_agent_poses = None
        t_proc_inputs = proc_inputs
        if self.needs_egocentric_transform:
            # Input conventions:
            # X is down, Y is right, origin is top-left
            # theta in radians from Y to X
            ego_agent_poses = extra_maps["ego_agent_poses"]  # (B, 3)
            # Convert to conventions appropriate for spatial_transform_map
            # Required conventions:
            # X is right, Y is down, origin is map center
            # theta in radians from new X to new Y (no changes in effect)
            t_ego_agent_poses = torch.stack(
                [
                    ego_agent_poses[:, 1] - W / 2.0,
                    ego_agent_poses[:, 0] - H / 2.0,
                    ego_agent_poses[:, 2],
                ],
                dim=1,
            )  # (B, 3)
            t_proc_inputs = pgeo.spatial_transform_map(t_proc_inputs, t_ego_agent_poses)

        with torch.no_grad():
            t_pfs, t_area_pfs = self.network(t_proc_inputs, rnn_hxs, masks, extras)

        if self.has_action_output:
            goal_cat_id = extras[:, 1].long()  # (bs, )
            out_actions = [
                t_pfs[e, gcat.item() + 2].argmax().item()
                for e, gcat in enumerate(goal_cat_id)
            ]
            return value, out_actions, action_log_probs, rnn_hxs, {}

        # Transform back the prediction if needed
        pfs = t_pfs
        area_pfs = t_area_pfs
        if self.needs_egocentric_transform:
            # Compute transform from t_ego_agent_poses -> origin
            origin_pose = torch.Tensor([[0.0, 0.0, 0.0]]).to(inputs.device)
            rev_ego_agent_poses = pgeo.subtract_poses(t_ego_agent_poses, origin_pose)
            pfs = pgeo.spatial_transform_map(pfs, rev_ego_agent_poses)  # (B, N, H, W)
            if area_pfs is not None:
                area_pfs = pgeo.spatial_transform_map(
                    area_pfs, rev_ego_agent_poses
                )  # (B, 1, H, W)

        # Add agent to location distance if needed
        if self.args.add_agent2loc_distance:
            agent_dists = extra_maps["dmap"]  # (B, H, W)
            pfs_dists = self.network.convert_object_pf_to_distance(pfs)  # (B, N, H, W)
            pfs_dists = pfs_dists + agent_dists.unsqueeze(1)
            # Convert back to a pf
            pfs = self.network.convert_distance_to_pf(pfs_dists)

        dist_pfs = None
        if self.args.add_agent2loc_distance_v2:
            agent_dists = extra_maps["dmap"].unsqueeze(1)  # (B, 1, H, W)
            dist_pfs = self.network.convert_distance_to_pf(agent_dists)

        # Take the mean with area_pfs
        init_pfs = pfs
        if area_pfs is not None:
            if dist_pfs is None:
                awc = self.args.area_weight_coef
                pfs = (1 - awc) * pfs + awc * area_pfs
            else:
                awc = self.args.area_weight_coef
                dwc = self.args.dist_weight_coef
                assert (awc + dwc <= 1) and (awc + dwc >= 0)
                pfs = (1 - awc - dwc) * pfs + awc * area_pfs + dwc * dist_pfs

        # Get action
        goal_cat_id = extras[:, 1].long()
        action = self.get_action(
            pfs,
            goal_cat_id,
            extra_maps["umap"],
            extra_maps["dmap"],
            extra_maps["agent_locations"],
        )
        pred_maps = {
            "pfs": pfs,
            "raw_pfs": init_pfs,
            "area_pfs": area_pfs,
        }
        pred_maps = {
            k: asnumpy(v) if v is not None else v for k, v in pred_maps.items()
        }
        if self.args.visualize or self.args.print_images:
            # Visualize the transformed PFs
            self._cached_visualizations = RL_Policy.generate_pf_vis(
                t_proc_inputs,
                pred_maps,
                goal_cat_id,
                dset=self.network.cfg.DATASET.dset_name,
            )
        return value, action, action_log_probs, rnn_hxs, pred_maps

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        raise NotImplementedError

    def do_proc(self, inputs):
        """
        Map consists of multiple channels containing the following:
        ----------- For local map -----------------
        1. Obstacle Map
        2. Explored Area
        3. Current Agent Location
        4. Past Agent Locations
        ----------- For global map -----------------
        5. Obstacle Map
        6. Explored Area
        7. Current Agent Location
        8. Past Agent Locations
        ----------- For semantic local map -----------------
        9,10,11,.. : Semantic Categories
        """
        # The input to PF model consists of Free map, Obstacle Map, Semantic Categories
        # The last semantic map channel is ignored since it belongs to unknown categories.
        obstacle_map = inputs[:, 0:1]
        explored_map = inputs[:, 1:2]
        semantic_map = inputs[:, 8:-1]
        free_map = ((obstacle_map < 0.5) & (explored_map >= 0.5)).float()
        outputs = torch.cat([free_map, obstacle_map, semantic_map], dim=1)
        return outputs

    def get_action(self, pfs, goal_cat_id, umap, dmap, agent_locs):
        """
        Computes distance from (agent -> location) + (location -> goal)
        based on PF predictions. It then selects goal as location with
        least distance.

        Args:
            pfs = (B, N + 2, H, W) potential fields
            goal_cat_id = (B, ) goal category
            umap = (B, H, W) unexplored map
            dmap = (B, H, W) geodesic distance from agent map
            agent_locs = B x 2 list of agent positions
        """
        B, N, H, W = pfs.shape[0], pfs.shape[1] - 2, pfs.shape[2], pfs.shape[3]
        goal_pfs = []
        for b in range(B):
            goal_pf = pfs[b, goal_cat_id[b].item() + 2, :]
            goal_pfs.append(goal_pf)
        goal_pfs = torch.stack(goal_pfs, dim=0)
        agt2loc_dist = dmap
        if self.args.pf_masking_opt == "unexplored":
            # Filter out explored locations
            goal_pfs = goal_pfs * umap
        # Filter out locations very close to the agent
        if self.args.mask_nearest_locations:
            for i in range(B):
                ri, ci = agent_locs[i]
                size = int(self.args.mask_size * 100.0 / self.args.map_resolution)
                goal_pfs[i, ri - size : ri + size + 1, ci - size : ci + size + 1] = 0

        act_ixs = goal_pfs.view(B, -1).max(dim=1).indices
        # Convert action to (0, 1) values for x and y coors
        actions = []
        for b in range(B):
            act_ix = act_ixs[b].item()
            # Convert action to (0, 1) values for x and y coors
            act_x = float(act_ix % W) / W
            act_y = float(act_ix // W) / H
            actions.append((act_y, act_x))
        actions = torch.Tensor(actions).to(pfs.device)

        return actions

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        raise NotImplementedError

    @staticmethod
    def generate_pf_vis(semantic_maps, pred_maps, goal_cat_ids, dset):
        vis_maps = []
        for i in range(semantic_maps.shape[0]):
            vis_maps_i = {}
            semmap = semantic_maps[i]
            pfs = pred_maps["pfs"][i]
            cat_id = goal_cat_ids[i].cpu().item()
            pfs_rgb = PFDataset.visualize_object_category_pf(semmap, pfs, cat_id, dset)
            vis_maps_i["pfs"] = pfs_rgb
            if "raw_pfs" in pred_maps and pred_maps["raw_pfs"] is not None:
                raw_pfs = pred_maps["raw_pfs"][i]
                raw_pfs_rgb = PFDataset.visualize_object_category_pf(
                    semmap,
                    raw_pfs,
                    cat_id,
                    dset,
                )
                vis_maps_i["raw_pfs"] = raw_pfs_rgb
            if "area_pfs" in pred_maps and pred_maps["area_pfs"] is not None:
                area_pfs = pred_maps["area_pfs"][i]
                area_pfs_rgb = PFDataset.visualize_area_pf(semmap, area_pfs, dset=dset)
                vis_maps_i["area_pfs"] = area_pfs_rgb
            vis_maps.append(vis_maps_i)
        return vis_maps

    def visualize_inputs_and_outputs(self, semantic_maps, object_pfs):
        for semmap, opfs in zip(semantic_maps, object_pfs):
            semmap = semmap.cpu().numpy()  # (N, H, W)
            opfs = opfs.cpu().numpy()  # (N, H, W)
            semmap_rgb = PFDataset.visualize_map(semmap)
            opfs_rgb = PFDataset.visualize_object_pfs(semmap, opfs)
            vis_image = PFDataset.combine_image_grid(
                semmap_rgb,
                semmap_rgb,
                opfs_rgb,
                dset=self.network.cfg.DATASET.dset_name,
            )
            cv2.imshow("Image", vis_image[..., ::-1])
            cv2.waitKey(0)
            break

    @property
    def visualizations(self):
        return self._cached_visualizations
