import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import skimage.morphology

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.pose as pu
import utils.visualization as svu
from einops import rearrange

from habitat import logger
from PIL import Image

from policies import policy_registry
from poni.constants import d3_40_colors_rgb, gibson_palette
from utils.planners import PlannerActorSequential, PlannerActorVector
from utils.rednet_semantic_prediction import SemanticPredRedNet
from utils.semantic_mapping import Semantic_Mapping


class GlobalAgent(object):
    valid_segmentation_types = ["rednet"]

    def __init__(self, cfg, device):

        self.cfg = cfg
        self.device = device
        self.dataset = cfg.GLOBAL_AGENT.dataset
        assert cfg.IMAGE_SEGMENTATION.type in self.valid_segmentation_types
        # Create segmentation model
        if cfg.IMAGE_SEGMENTATION.type == "rednet":
            self.sem_seg_model = SemanticPredRedNet(cfg.IMAGE_SEGMENTATION)
        # Create semantic mapping model
        self.sem_map_model = Semantic_Mapping.from_config(cfg, self.device)
        self.sem_map_model.eval()
        # Create global policy
        self.g_policy = policy_registry.get_policy(cfg.GLOBAL_AGENT.name).from_config(
            cfg
        )
        self.g_policy.load_checkpoint()
        self.g_policy.to(device)
        self.g_policy.eval()
        # Efficient mapping
        self.seg_interval = cfg.GLOBAL_AGENT.seg_interval
        self.num_conseq_fwds = 0
        if self.seg_interval > 0:
            assert cfg.PLANNER.n_planners == 1
            n_classes = cfg.IMAGE_SEGMENTATION.n_classes
            H = cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
            W = cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
            self.zero_sem_seg = torch.zeros(1, n_classes + 1, H, W, device=device)
        self.stop_upon_replan = cfg.GLOBAL_AGENT.stop_upon_replan
        if self.stop_upon_replan:
            assert cfg.PLANNER.n_planners == 1
            self.replan_count = 0

        self.kornia = None
        if cfg.SEMANTIC_MAPPING.use_gt_segmentation:
            import kornia

            self.kornia = kornia
        # Create local policy
        if cfg.PLANNER.n_planners > 1:
            self.planners = PlannerActorVector(cfg.PLANNER)
        else:
            self.planners = PlannerActorSequential(cfg.PLANNER)
        # Useful pre-computation
        gcfg = self.cfg.GLOBAL_AGENT
        full_size = gcfg.map_size_cm // gcfg.map_resolution
        local_size = int(full_size / gcfg.global_downscaling)
        self._full_map_size = (full_size, full_size)
        self._local_map_size = (local_size, local_size)
        self.color_palette = None
        self.time_benchmarks = defaultdict(lambda: deque(maxlen=50))
        # Visualization
        if cfg.GLOBAL_AGENT.visualize:
            self.legend = cv2.imread(f"../docs/legend_{self.dataset}.png")
            if self.dataset == "gibson":
                self.color_palette = [int(x * 255.0) for x in gibson_palette]
            else:
                ncat = self.cfg.IMAGE_SEGMENTATION.n_classes
                self.color_palette = [
                    255,
                    255,
                    255,  # Out of bounds
                    77,
                    77,
                    77,  # Obstacles
                    230,
                    230,
                    230,  # Free space
                    245,
                    92,
                    63,  # Visible mask
                    31,
                    120,
                    180,  # Goal mask
                ]
                self.color_palette += [
                    c for color in d3_40_colors_rgb[:ncat] for c in color.tolist()
                ]

    def act(self, batched_obs, agent_states, steps, g_masks, l_masks):

        gcfg = self.cfg.GLOBAL_AGENT
        l_step = steps % gcfg.num_local_steps

        full_map = agent_states["full_map"]
        full_pose = agent_states["full_pose"]
        lmb = agent_states["lmb"]
        local_map = agent_states["local_map"]
        local_pose = agent_states["local_pose"]
        planner_pose_inputs = agent_states["planner_pose_inputs"]
        origins = agent_states["origins"]
        wait_env = agent_states["wait_env"]
        finished = agent_states["finished"]
        global_orientation = agent_states["global_orientation"]
        global_input = agent_states["global_input"]
        global_goals = agent_states["global_goals"]
        extras = agent_states["extras"]
        local_w, local_h = self.local_map_size
        full_w, full_h = self.full_map_size

        ########################################################################
        # Semantic Mapping
        ########################################################################
        start_time = time.time()
        poses = self._get_poses_from_obs(batched_obs, agent_states, g_masks)
        state = self.preprocess_obs(batched_obs)
        self.time_benchmarks["semantic_prediction"].append(time.time() - start_time)

        start_time = time.time()
        _, local_map, _, local_pose = self.sem_map_model(
            state, poses, local_map, local_pose
        )

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(self.cfg.NUM_ENVIRONMENTS):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / gcfg.map_resolution),
                int(c * 100.0 / gcfg.map_resolution),
            ]
            local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

        self.time_benchmarks["semantic_mapping"].append(time.time() - start_time)
        ########################################################################
        # Global policy
        ########################################################################
        if (steps == 0) or (l_step == gcfg.num_local_steps - 1):
            start_time = time.time()
            # For every global step, update the full and local maps
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.0

                full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ] = local_map[e]
                full_pose[e] = (
                    local_pose[e] + torch.from_numpy(origins[e]).to(self.device).float()
                )

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / gcfg.map_resolution),
                    int(c * 100.0 / gcfg.map_resolution),
                ]

                if not self.cfg.GLOBAL_AGENT.smart_local_boundaries:
                    update_local_boundaries = True
                else:
                    local_r, local_c = local_pose[e, 1].item(), local_pose[e, 0].item()
                    local_loc_r, local_loc_c = [
                        int(local_r * 100.0 / gcfg.map_resolution),
                        int(local_c * 100.0 / gcfg.map_resolution),
                    ]
                    if local_loc_r < (local_w * 0.2) or local_loc_r > (local_w * 0.8):
                        update_local_boundaries = True
                    elif local_loc_c < (local_h * 0.2) or local_loc_c > (local_h * 0.8):
                        update_local_boundaries = True
                    else:
                        update_local_boundaries = False

                if update_local_boundaries:
                    lmb[e] = self.get_local_map_boundaries(
                        (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
                    )

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [
                    lmb[e][2] * gcfg.map_resolution / 100.0,
                    lmb[e][0] * gcfg.map_resolution / 100.0,
                    0.0,
                ]

                local_map[e] = full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]
                local_pose[e] = (
                    full_pose[e] - torch.from_numpy(origins[e]).to(self.device).float()
                )

            locs = local_pose.cpu().numpy()
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
            global_input[:, 4:8, :, :] = nn.MaxPool2d(gcfg.global_downscaling)(
                full_map[:, 0:4, :, :]
            )
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = batched_obs["objectgoal"]
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id[:, 0]
            self.time_benchmarks["map_update"].append(time.time() - start_time)

            ####################################################################
            start_time = time.time()
            # Compute additional inputs if needed
            extra_maps = {
                "dmap": None,
                "umap": None,
                "fmap": None,
                "pfs": None,
                "agent_locations": None,
                "ego_agent_poses": None,
            }

            extra_maps["agent_locations"] = []
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                pose_pred = planner_pose_inputs[e]
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                map_r, map_c = start_y, start_x
                map_loc = [
                    int(map_r * 100.0 / gcfg.map_resolution - gx1),
                    int(map_c * 100.0 / gcfg.map_resolution - gy1),
                ]
                map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
                extra_maps["agent_locations"].append(map_loc)

            if self.g_policy.needs_dist_maps:
                planner_inputs = [{} for e in range(self.cfg.NUM_ENVIRONMENTS)]
                for e, p_input in enumerate(planner_inputs):
                    obs_map = local_map[e, 0, :, :].cpu().numpy()
                    exp_map = local_map[e, 1, :, :].cpu().numpy()
                    # set unexplored to navigable by default
                    p_input["map_pred"] = (obs_map * np.rint(exp_map)) > 0
                    p_input["pose_pred"] = planner_pose_inputs[e]
                masks = [1 for _ in range(self.cfg.NUM_ENVIRONMENTS)]
                _, dmap = self.planners.get_reachability_maps(planner_inputs, masks)
                dmap = torch.from_numpy(dmap).to(self.device)
                # Convert to float
                dmap = dmap.float().div_(100.0)  # cm -> m
                extra_maps["dmap"] = dmap

            if self.g_policy.needs_unexplored_maps:
                extra_maps["umap"] = 1.0 - local_map[:, 1, :, :]

            if self.g_policy.needs_frontier_maps:
                planner_inputs = [{} for e in range(self.cfg.NUM_ENVIRONMENTS)]
                for e, p_input in enumerate(planner_inputs):
                    obs_map = local_map[e, 0, :, :].cpu().numpy()
                    exp_map = local_map[e, 1, :, :].cpu().numpy()
                    p_input["obs_map"] = obs_map
                    p_input["exp_map"] = exp_map
                    p_input["pose_pred"] = planner_pose_inputs[e]
                masks = [1 for _ in range(self.cfg.NUM_ENVIRONMENTS)]
                fmap = self.planners.get_frontier_maps(planner_inputs, masks)
                extra_maps["fmap"] = torch.from_numpy(fmap).to(self.device)

            if self.g_policy.needs_egocentric_transform:
                ego_agent_poses = []
                for e in range(self.cfg.NUM_ENVIRONMENTS):
                    map_loc = extra_maps["agent_locations"][e]
                    # Crop map about a center
                    ego_agent_poses.append(
                        [map_loc[0], map_loc[1], math.radians(start_o)]
                    )
                ego_agent_poses = torch.Tensor(ego_agent_poses).to(self.device)
                extra_maps["ego_agent_poses"] = ego_agent_poses

            self.time_benchmarks["extra_inputs"].append(time.time() - start_time)
            ####################################################################
            start_time = time.time()
            # Sample long-term goal from global policy
            _, g_action, _, _ = self.g_policy.act(
                global_input,
                None,
                g_masks,
                extras=extras.long(),
                deterministic=False,
                extra_maps=extra_maps,
            )
            if not self.g_policy.has_action_output:
                cpu_actions = g_action.cpu().numpy()
                if len(cpu_actions.shape) == 4:
                    # Output action map
                    global_goals = cpu_actions[:, 0]  # (B, H, W)
                elif len(cpu_actions.shape) == 3:
                    # Output action map
                    global_goals = cpu_actions  # (B, H, W)
                else:
                    # Output action locations
                    assert len(cpu_actions.shape) == 2
                    global_goals = [
                        [int(action[0] * local_w), int(action[1] * local_h)]
                        for action in cpu_actions
                    ]
                    global_goals = [
                        [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                        for x, y in global_goals
                    ]
            g_masks.fill_(1.0)
            self.time_benchmarks["goal_sampling"].append(time.time() - start_time)

        ########################################################################
        # Define long-term goal map
        ########################################################################
        start_time = time.time()
        found_goal = [0 for _ in range(self.cfg.NUM_ENVIRONMENTS)]
        goal_maps = [
            np.zeros((local_w, local_h)) for _ in range(self.cfg.NUM_ENVIRONMENTS)
        ]

        if not self.g_policy.has_action_output:
            # Set goal to sampled location
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                if type(global_goals) == type([]):
                    goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                else:
                    assert len(global_goals.shape) == 3
                    goal_maps[e][:, :] = global_goals[e]

        # Re-set goal to object if already visible
        for e in range(self.cfg.NUM_ENVIRONMENTS):
            cn = int(batched_obs["objectgoal"][e, 0].item()) + 4
            cat_semantic_map = local_map[e, cn, :, :]
            if cat_semantic_map.sum() != 0.0:
                cat_semantic_map = cat_semantic_map.cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1

        self.time_benchmarks["goal_map_building"].append(time.time() - start_time)
        ########################################################################
        # Plan and sample action
        ########################################################################
        start_time = time.time()
        planner_inputs = [{} for e in range(self.cfg.NUM_ENVIRONMENTS)]
        pf_visualizations = None
        if self.cfg.GLOBAL_AGENT.visualize:
            pf_visualizations = self.g_policy.visualizations
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy() > 0
            p_input["pose_pred"] = planner_pose_inputs[e]
            p_input["goal"] = goal_maps[e] > 0  # global_goals[e]
            p_input["new_goal"] = l_step == gcfg.num_local_steps - 1
            p_input["found_goal"] = found_goal[e]
            if self.g_policy.has_action_output:
                p_input["wait"] = (not found_goal[e]) or wait_env[e] or finished[e]
            else:
                p_input["wait"] = wait_env[e] or finished[e]
            if self.cfg.GLOBAL_AGENT.visualize:
                p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy() > 0
                local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                p_input["pf_pred"] = pf_visualizations[e]

        actions, replan_flags = self.planners.plan_and_act(
            planner_inputs, l_masks.cpu().numpy()
        )  # (B, 1) ndarray
        actions = torch.from_numpy(actions)
        if self.g_policy.has_action_output:
            for e in range(self.cfg.NUM_ENVIRONMENTS):
                if not found_goal[e]:
                    actions[e] = g_action[e]
        for e, flag in enumerate(replan_flags):
            if flag and self.cfg.GLOBAL_AGENT.reset_map_upon_replan:
                # print(f'=====> Reseting map for process {e}')
                local_map[e].fill_(0)
            if flag and self.cfg.GLOBAL_AGENT.stop_upon_replan:
                self.replan_count += 1
                if self.replan_count >= self.cfg.GLOBAL_AGENT.stop_upon_replan_thresh:
                    print("========> Early stopping after 2 replans")
                    self.replan_count = 0
                    actions[e] = self.cfg.PLANNER.ACTION.stop
        # Cache for visualization purposes
        self._cached_planner_inputs = planner_inputs

        self.time_benchmarks["plan_and_act"].append(time.time() - start_time)
        if self.seg_interval > 0:
            if actions[0, 0].item() == self.cfg.PLANNER.ACTION.move_forward:
                self.num_conseq_fwds += 1
                self.num_conseq_fwds = self.num_conseq_fwds % self.seg_interval
            else:
                self.num_conseq_fwds = 0
        ########################################################################
        # Update agent states
        ########################################################################
        agent_states["full_map"] = full_map
        agent_states["full_pose"] = full_pose
        agent_states["lmb"] = lmb
        agent_states["local_map"] = local_map
        agent_states["local_pose"] = local_pose
        agent_states["planner_pose_inputs"] = planner_pose_inputs
        agent_states["origins"] = origins
        agent_states["wait_env"] = wait_env
        agent_states["finished"] = finished
        agent_states["global_orientation"] = global_orientation
        agent_states["global_input"] = global_input
        agent_states["global_goals"] = global_goals
        agent_states["extras"] = extras

        if steps % 50 == 0:
            logger.info("=====> Time benchmarks")
            for k, v in self.time_benchmarks.items():
                logger.info(f"{k:<20s} : {np.mean(v).item():6.4f} secs")

        return actions, agent_states

    def preprocess_obs(self, batched_obs):
        cfg = self.cfg.GLOBAL_AGENT
        rgb = batched_obs["rgb"]  # (B, H, W, 3) torch Tensor
        depth = batched_obs["depth"]  # (B, H, W, 1) torch Tensor
        # Process depth
        depth = self.preprocess_depth(depth)
        # Re-format observations
        rgb = rearrange(rgb, "b h w c -> b c h w").float()
        depth = rearrange(depth, "b h w c -> b c h w").float()
        # Get semantic segmentation prediction
        if "semantic_category" in batched_obs:  # (B, H, W)
            sem = batched_obs["semantic_category"]
            n_classes = self.cfg.IMAGE_SEGMENTATION.n_classes
            B, H, W = sem.shape
            sem_seg_pred = torch.zeros(B, n_classes + 1, H, W, device=sem.device)
            for i in range(n_classes):
                class_img = (sem == i).float().unsqueeze(1)  # (B, 1, H, W)
                kernel = torch.ones(10, 10, device=depth.device)
                class_img = self.kornia.morphology.erosion(class_img, kernel)
                sem_seg_pred[:, i] = class_img[:, 0]
        elif self.seg_interval > 0 and self.num_conseq_fwds != 0:
            sem_seg_pred = self.zero_sem_seg
        else:
            if self.cfg.IMAGE_SEGMENTATION.type == "mask_rcnn":
                sem_seg_pred = self.sem_seg_model.get_predictions(rgb)  # (B, N, H, W)
            else:
                # Convert depth to meters
                depth_m = depth / 100.0
                sem_seg_pred = self.sem_seg_model.get_predictions(
                    rgb, depth_m
                )  # (B, N, H, W)
        # Downscale observations
        ds = cfg.env_frame_width // cfg.frame_width
        if ds != 1:
            rgb = F.interpolate(
                rgb,
                (cfg.frame_height, cfg.frame_width),
                mode="nearest",
            )
            depth = depth[:, :, (ds // 2) :: ds, (ds // 2) :: ds]
            sem_seg_pred = sem_seg_pred[:, :, (ds // 2) :: ds, (ds // 2) :: ds]

        state = torch.cat([rgb, depth, sem_seg_pred], dim=1)

        return state

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) torch Tensor
        task_cfg = self.cfg.TASK_CONFIG
        min_depth = task_cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = task_cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

        return depth

    def close(self):
        self.planners.close()

    @property
    def full_map_size(self):
        return self._full_map_size

    @property
    def local_map_size(self):
        return self._local_map_size

    @property
    def cached_planner_inputs(self):
        return self._cached_planner_inputs

    def _get_poses_from_obs(self, batched_obs, agent_states, g_masks):
        curr_sim_location = torch.stack(
            [
                batched_obs["gps"][:, 0],  # -Z
                -batched_obs["gps"][:, 1],  # -X
                batched_obs["compass"][:, 0],  # Heading
            ],
            dim=1,
        )
        prev_sim_location = torch.from_numpy(agent_states["prev_sim_location"]).to(
            curr_sim_location.device
        )
        # Measure pose change
        pose = self._get_rel_pose_change(prev_sim_location, curr_sim_location)
        # If episode terminated in last step, set pose change to zero
        pose = pose * g_masks
        # Update prev_sim_location
        agent_states["prev_sim_location"] = curr_sim_location.cpu().numpy()
        return pose

    def _get_rel_pose_change(self, pos1, pos2):
        x1, y1, o1 = torch.unbind(pos1, dim=1)
        x2, y2, o2 = torch.unbind(pos2, dim=1)

        theta = torch.atan2(y2 - y1, x2 - x1) - o1
        dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        dx = dist * torch.cos(theta)
        dy = dist * torch.sin(theta)
        do = o2 - o1

        return torch.stack([dx, dy, do], dim=1)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.cfg.GLOBAL_AGENT.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self, agent_states):

        gcfg = self.cfg.GLOBAL_AGENT

        local_w, local_h = self.local_map_size
        full_w, full_h = self.full_map_size

        full_map = agent_states["full_map"]
        full_pose = agent_states["full_pose"]
        lmb = agent_states["lmb"]
        origins = agent_states["origins"]
        local_map = agent_states["local_map"]
        local_pose = agent_states["local_pose"]
        planner_pose_inputs = agent_states["planner_pose_inputs"]

        full_map.fill_(0.0)
        full_pose.fill_(0.0)
        full_pose[:, :2] = gcfg.map_size_cm / 100.0 / 2.0
        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs

        for e in range(self.cfg.NUM_ENVIRONMENTS):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / gcfg.map_resolution),
                int(c * 100.0 / gcfg.map_resolution),
            ]

            full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            lmb[e] = self.get_local_map_boundaries(
                (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
            )

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [
                lmb[e][2] * gcfg.map_resolution / 100.0,
                lmb[e][0] * gcfg.map_resolution / 100.0,
                0.0,
            ]

        for e in range(self.cfg.NUM_ENVIRONMENTS):
            local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
            local_pose[e] = (
                full_pose[e] - torch.from_numpy(origins[e]).to(self.device).float()
            )

        # Update states (probably unnecessary)
        agent_states["full_map"] = full_map
        agent_states["full_pose"] = full_pose
        agent_states["lmb"] = lmb
        agent_states["origins"] = origins
        agent_states["local_map"] = local_map
        agent_states["local_pose"] = local_pose
        agent_states["planner_pose_inputs"] = planner_pose_inputs

        return agent_states

    def init_map_and_pose_for_env(self, agent_states, e):

        gcfg = self.cfg.GLOBAL_AGENT

        local_w, local_h = self.local_map_size
        full_w, full_h = self.full_map_size

        full_map = agent_states["full_map"]
        full_pose = agent_states["full_pose"]
        lmb = agent_states["lmb"]
        origins = agent_states["origins"]
        local_map = agent_states["local_map"]
        local_pose = agent_states["local_pose"]
        planner_pose_inputs = agent_states["planner_pose_inputs"]

        full_map[e].fill_(0.0)
        full_pose[e].fill_(0.0)
        full_pose[e, :2] = gcfg.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / gcfg.map_resolution),
            int(c * 100.0 / gcfg.map_resolution),
        ]

        full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        lmb[e] = self.get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
        )

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [
            lmb[e][2] * gcfg.map_resolution / 100.0,
            lmb[e][0] * gcfg.map_resolution / 100.0,
            0.0,
        ]

        local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
        local_pose[e] = (
            full_pose[e] - torch.from_numpy(origins[e]).to(self.device).float()
        )

        # Update states (probably unnecessary)
        agent_states["full_map"] = full_map
        agent_states["full_pose"] = full_pose
        agent_states["lmb"] = lmb
        agent_states["origins"] = origins
        agent_states["local_map"] = local_map
        agent_states["local_pose"] = local_pose
        agent_states["planner_pose_inputs"] = planner_pose_inputs

        return agent_states

    def get_new_agent_states(self):
        ########################################################################
        # Create agent states:
        ########################################################################
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = self.cfg.GLOBAL_AGENT.num_sem_categories + 4  # num channels
        full_w, full_h = self.full_map_size
        local_w, local_h = self.local_map_size

        B = self.cfg.NUM_ENVIRONMENTS

        full_map = torch.zeros(B, nc, full_w, full_h).to(self.device)
        local_map = torch.zeros(B, nc, local_w, local_h).to(self.device)

        # Create pose estimates
        full_pose = torch.zeros(B, 3).to(self.device)
        local_pose = torch.zeros(B, 3).to(self.device)

        # Origins of local map
        origins = np.zeros((B, 3))

        # Local map boundaries
        lmb = np.zeros((B, 4), dtype=int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        planner_pose_inputs = np.zeros((B, 7))

        # Global policy states
        ngc = 8 + self.cfg.GLOBAL_AGENT.num_sem_categories
        es = 2
        global_input = torch.zeros(B, ngc, local_w, local_h).to(self.device)
        global_orientation = torch.zeros(B, 1).long().to(self.device)
        extras = torch.zeros(B, es).to(self.device)
        prev_sim_location = np.zeros((B, 3), dtype=np.float32)

        # Other states
        wait_env = np.zeros((B,))
        finished = np.zeros((B,))

        agent_states = {
            "full_map": full_map,
            "local_map": local_map,
            "full_pose": full_pose,
            "local_pose": local_pose,
            "origins": origins,
            "lmb": lmb,
            "planner_pose_inputs": planner_pose_inputs,
            "global_input": global_input,
            "global_orientation": global_orientation,
            "global_goals": None,
            "extras": extras,
            "prev_sim_location": prev_sim_location,
            "finished": finished,
            "wait_env": wait_env,
        }

        return agent_states

    def visualize_states(self, inputs, rgb_image, goal_name, height=None):

        gcfg = self.cfg.GLOBAL_AGENT

        has_pf_maps = hasattr(self.g_policy, "prev_maps")
        num_pf_maps = 0
        if has_pf_maps:
            num_pf_maps += 1
            if "area_pfs" in self.g_policy.prev_maps:
                num_pf_maps += 2

        vis_image = svu.init_vis_image(goal_name, self.legend, num_pf_maps)

        map_pred = inputs["map_pred"]
        exp_pred = inputs["exp_pred"]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs["pose_pred"]

        goal = inputs["goal"]
        sem_map = inputs["sem_map_pred"]

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == (self.cfg.IMAGE_SEGMENTATION.n_classes + 5)
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        # sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = cv2.dilate(goal.astype(np.float32), selem)
        # goal_mat = 1 - skimage.morphology.binary_dilation(
        #     goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        sem_map_vis.putpalette(self.color_palette)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(
            sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )

        pos = (
            (start_x * 100.0 / gcfg.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100.0 / gcfg.map_resolution + gx1)
            * 480
            / map_pred.shape[1],
            np.deg2rad(-start_o),
        )

        vis_image[50:530, 15:655] = rgb_image[..., ::-1]
        vis_image[50:530, 670:1150] = sem_map_vis

        agent_arrow = svu.get_contour_points(pos, origin=(670, 50))
        color = (
            int(gibson_palette[11] * 255),
            int(gibson_palette[10] * 255),
            int(gibson_palette[9] * 255),
        )
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        if "pf_pred" in inputs:
            # Rescale pf_pred to match the height of vis_image
            vis_maps = inputs["pf_pred"]
            vis_maps_list = [vis_maps["pfs"]]
            if "area_pfs" in vis_maps:
                vis_maps_list.append(vis_maps["raw_pfs"])
                vis_maps_list.append(vis_maps["area_pfs"])
            for i, vis_map in enumerate(vis_maps_list):
                start_x = 1150 + 15 * (i + 1) + 480 * i
                start_y = 50
                end_x = start_x + 480
                end_y = start_y + 480
                vis_map = cv2.resize(vis_map, (480, 480))
                # Apply up-down flipping similar to vis_image
                vis_map = np.flipud(vis_map)
                vis_image[start_y:end_y, start_x:end_x] = vis_map[..., ::-1]

        # BGR -> RGB
        vis_image = vis_image[..., ::-1]

        # Resize if needed
        if height is not None:
            oH, oW = vis_image.shape[:2]
            nH, nW = height, int(height / oH * oW)
            vis_image = cv2.resize(vis_image, (nW, nH))

        return vis_image
