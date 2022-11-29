import math
import os
import time

import cv2
import imageio
import numpy as np
import semexp.agents.utils.visualization as vu
import semexp.envs.utils.pose as pu
import skimage.morphology
import torch
from PIL import Image
from poni.geometry import crop_map, crop_map_with_pad, spatial_transform_map
from semexp.agents.utils.semantic_prediction import SemanticPredMaskRCNN
from semexp.constants import color_palette

from semexp.envs.utils.fmm_planner import FMMPlanner
from torchvision import transforms

from .objectgoal_env import MultiObjectGoal_Env


class Sem_Exp_Env_Agent(MultiObjectGoal_Env):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (args.frame_height, args.frame_width), interpolation=Image.NEAREST
                ),
            ]
        )

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        self.stg_selem = skimage.morphology.disk(10)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.prev_goal_ix = None
        self.start_map_loc_and_ort = None
        self.zero_sem_seg = None
        if args.seg_interval > 0:
            self.num_conseq_fwd = None

        if args.visualize or args.print_images:
            self.legend = cv2.imread("docs/legend_gibson.png")
            self.vis_image = None
            self.rgb_vis = None
            self.video_writer = None

    def reset(self):
        args = self.args

        obs, info = super().reset()
        obs = self._preprocess_obs(obs)
        self.starting_map_loc_and_ort = self.sim_continuous_to_sim_map(
            self.get_sim_location()
        )

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (
            args.map_size_cm // args.map_resolution,
            args.map_size_cm // args.map_resolution,
        )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [
            args.map_size_cm / 100.0 / 2.0,
            args.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.prev_goal_ix = self.active_goal_ix
        self.num_conseq_fwd = None
        if args.seg_interval > 0:
            self.num_conseq_fwd = 0

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(
                self.goal_names[0],
                self.legend,
                args.num_pf_maps,
                add_sem_seg=True,
            )
            # Delete existing video writer
            if self.video_writer is not None:
                self.video_writer.close()
            # Create video writer
            dump_dir = "{}/dump/{}/".format(self.args.dump_location, self.args.exp_name)
            thread_dir = "{}/episodes/thread_{}".format(dump_dir, self.rank)
            os.makedirs(thread_dir, exist_ok=True)
            save_path = "{}/eps_{:04d}.mp4".format(thread_dir, self.episode_no)
            self.video_writer = imageio.get_writer(
                save_path,
                codec="h264",
                fps=10,
                quality=None,
                pixelformat="yuv420p",
                bitrate=0,
                output_params=["-crf", "31"],
            )

        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0.0, 0.0, 0.0]
            self.info["gps"] = [0.0, 0.0]
            self.info["compass"] = [0.0]
            return np.zeros(self.obs.shape), 0.0, False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        start_time = time.time()
        if "atomic_action" in planner_inputs and not planner_inputs["found_goal"]:
            action = planner_inputs["atomic_action"]
        else:
            action = self._plan(planner_inputs)
        planning_time = time.time() - start_time

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        if action >= 0:

            start_time = time.time()
            # act
            action = {"action": action}
            obs, rew, done, info = super().step(action)

            env_time = time.time() - start_time

            start_time = time.time()
            # preprocess obs
            if self.args.seg_interval > 0:
                use_seg = True if self.num_conseq_fwd == 0 else False
                self.num_conseq_fwd = (self.num_conseq_fwd + 1) % self.args.seg_interval
                if action["action"] != 1:  # not forward
                    use_seg = True
                    self.num_conseq_fwd = 0
            else:
                use_seg = True

            obs = self._preprocess_obs(obs, use_seg=use_seg)
            self.last_action = action["action"]
            self.obs = obs
            self.info = info

            preprocess_time = time.time() - start_time

            info["g_reward"] += rew
            info["planning_time"] = planning_time
            info["env_time"] = env_time
            info["preprocess_time"] = preprocess_time

            return obs, rew, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0.0, 0.0, 0.0]
            self.info["gps"] = [0.0, 0.0]
            self.info["compass"] = [0.0]
            self.info["planning_time"] = planning_time
            self.info["env_time"] = 0.0
            self.info["preprocess_time"] = 0.0
            return np.zeros(self.obs_shape), 0.0, False, self.info

    def get_reachability_map(self, planner_inputs):
        """Function responsible for planning, and identifying reachable locations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)

        Returns:
            reachability_map (ndarray): (M, M) map of reachable locations
            fmm_dist (ndarray): (M, M) map of geodesic distance
        """
        args = self.args

        start_time = time.time()
        # Get agent position + local boundaries
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        # Get map
        map_pred = np.rint(planner_inputs["map_pred"])
        # Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)
        # Create a goal map (start is goal)
        goal_map = np.zeros(map_pred.shape)
        goal_map[start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1] = 1
        # Figure out reachable locations
        reachability, fmm_dist = self._get_reachability(
            map_pred, goal_map, planning_window
        )
        planning_time = time.time() - start_time

        return reachability, fmm_dist

    def get_frontier_map(self, planner_inputs):
        """Function responsible for computing frontiers in the input map

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obs_map' (ndarray): (M, M) map of obstacle locations
                    'exp_map' (ndarray): (M, M) map of explored locations

        Returns:
            frontier_map (ndarray): (M, M) binary map of frontier locations
        """
        args = self.args

        obs_map = np.rint(planner_inputs["obs_map"])
        exp_map = np.rint(planner_inputs["exp_map"])
        # compute free and unexplored maps
        free_map = (1 - obs_map) * exp_map
        unk_map = 1 - exp_map
        # Clean maps
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(free_map, cv2.MORPH_CLOSE, kernel)
        unk_map[free_map == 1] = 0
        # https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132
        unk_map_shiftup = np.pad(
            unk_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unk_map_shiftdown = np.pad(
            unk_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unk_map_shiftleft = np.pad(
            unk_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unk_map_shiftright = np.pad(
            unk_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (H, W)
        frontiers = frontiers.astype(np.uint8)
        # Select only large-enough frontiers
        contours, _ = cv2.findContours(
            frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) > 0:
            contours = [c[:, 0].tolist() for c in contours]  # Clean format
            new_frontiers = np.zeros_like(frontiers)
            # Only pick largest 5 frontiers
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            for contour in contours[:5]:
                contour = np.array(contour)
                # Select only the central point of the contour
                lc = len(contour)
                if lc > 0:
                    new_frontiers[contour[lc // 2, 1], contour[lc // 2, 0]] = 1
            frontiers = new_frontiers
        frontiers = frontiers > 0
        # Mask out frontiers very close to the agent
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        ## Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, frontiers.shape)
        ## Mask out a 100.0 x 100.0 cm region center on the agent
        ncells = int(100.0 / args.map_resolution)
        frontiers[
            (start[0] - ncells) : (start[0] + ncells + 1),
            (start[1] - ncells) : (start[1] + ncells + 1),
        ] = False
        # Handle edge case where frontier becomes zero
        if not np.any(frontiers):
            # Set a random location to True
            rand_y = np.random.randint(start[0] - ncells, start[0] + ncells + 1)
            rand_x = np.random.randint(start[1] - ncells, start[1] + ncells + 1)
            frontiers[rand_y, rand_x] = True

        return frontiers

    def convert_dist_to_pf(self, dist, pf_cfg):
        return np.clip((pf_cfg["dthresh"] - dist) / pf_cfg["dthresh"], 0.0, 1.0)

    def get_fmm_dists(self, planner_inputs):
        """Function responsible for planning, and identifying reachable locations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'pred_map'   (ndarray): (N, H, W) map with 0 as floor, 1 - N as categories
                    'map_resolution' (int): size of grid-cell in pred_map

        Returns:
            fmm_dists (ndarray): (N, H, W) map of FMM dists per category
        """
        pred_map = planner_inputs["pred_map"]
        fmm_dists = np.zeros_like(pred_map)
        fmm_dists.fill(np.inf)
        map_resolution = planner_inputs["map_resolution"]
        orig_map_resolution = self.args.map_resolution
        assert orig_map_resolution == map_resolution

        # Setup planner
        traversible = pred_map[0]
        planner = FMMPlanner(traversible)
        # Get FMM dists to each category
        selem = skimage.morphology.disk(
            int(self.object_boundary / 4.0 * 100.0 / self.args.map_resolution)
        )
        for i in range(1, fmm_dists.shape[0]):
            if np.count_nonzero(pred_map[i]) == 0:
                continue
            goal_map = cv2.dilate(pred_map[i], selem)
            planner.set_multi_goal(goal_map)
            fmm_dist = planner.fmm_dist * map_resolution / 100.0
            fmm_dists[i] = fmm_dist

        return fmm_dists

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs["map_pred"])
        goal = planner_inputs["goal"]

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [
                int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1),
            ]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(
                last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]
            )

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * (
                            (i + buf) * np.cos(np.deg2rad(t1))
                            + (j - width // 2) * np.sin(np.deg2rad(t1))
                        )
                        wy = y1 + 0.05 * (
                            (i + buf) * np.sin(np.deg2rad(t1))
                            - (j - width // 2) * np.cos(np.deg2rad(t1))
                        )
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), int(
                            c * 100 / args.map_resolution
                        )
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window)

        # Deterministic Local Policy
        if stop and planner_inputs["found_goal"] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.0:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.0:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)

        # goal = cv2.dilate(goal, self.stg_selem)

        # step_size = 5
        # stg_x, stg_y = None, None
        # obstacles = (1 - traversible).astype(np.float32)
        # astar_goal = goal.astype(np.float32)
        # astar_start = [int(start[1] - y1 + 1), int(start[0] - x1 + 1)]
        # path_y, path_x = pyastar.multi_goal_astar_planner(
        #     obstacles, astar_start, astar_goal, True
        # )
        # if path_x is not None:
        #     # The paths are in reversed order
        #     stg_x = path_x[-min(step_size, len(path_x))]
        #     stg_y = path_y[-min(step_size, len(path_y))]
        #     stop = False
        #     if len(path_x) < step_size:
        #         # Measure distance along the shortest path
        #         path_xy = np.stack([path_x, path_y], axis=1)
        #         d2g = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
        #         d2g = d2g.sum() * self.args.map_resolution / 100.0 # In meters
        #         if d2g <= 0.25:
        #             stop = True
        #             print(f'=======> Estimated DTS: {d2g:.2f}')

        # if stg_x is None:
        #     # Pick some arbitrary location as the short-term goal
        #     random_theta = np.random.uniform(-np.pi, np.pi, (1, ))[0].item()
        #     stg_x = int(step_size * np.cos(random_theta))
        #     stg_y = int(step_size * np.sin(random_theta))
        #     stop = False

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = cv2.dilate(goal, self.stg_selem)
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _get_reachability(self, grid, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(3)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution / 100.0

        reachability = fmm_dist < fmm_dist.max()

        return reachability.astype(np.float32), fmm_dist.astype(np.float32)

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        if args.use_gt_segmentation:
            semantic_category = obs[:, :, 4]
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            kernel = np.ones((10, 10))
            for i in range(0, sem_seg_pred.shape[2]):
                cat_img = (semantic_category == i).astype(np.float32)
                cat_img = cv2.erode(cat_img, kernel)
                # Fixes a bug in rendering where it semantics are vertically flipped
                sem_seg_pred[..., i] = cat_img
            self.rgb_vis = rgb[:, :, ::-1]
        else:
            sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), use_seg=use_seg)
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2 :: ds, ds // 2 :: ds]
            sem_seg_pred = sem_seg_pred[ds // 2 :: ds, ds // 2 :: ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            if self.zero_sem_seg is None:
                self.zero_sem_seg = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            semantic_pred = self.zero_sem_seg
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred

    def _visualize(self, inputs):
        args = self.args

        # Re-initialize visualization if goal changed
        if self.prev_goal_ix != self.active_goal_ix:
            self.prev_goal_ix = self.active_goal_ix
            self.vis_image = vu.init_vis_image(
                self.goal_names[self.active_goal_ix],
                self.legend,
                num_pf_maps=args.num_pf_maps,
                add_sem_seg=True,
            )

        map_pred = inputs["map_pred"]
        exp_pred = inputs["exp_pred"]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs["pose_pred"]

        goal = inputs["goal"]
        sem_map = inputs["sem_map_pred"]
        sem_seg = inputs["sem_seg"]

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5
        sem_seg += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = cv2.dilate(goal.astype(np.float32), selem)

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.0) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(
            sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )

        sem_seg_vis = Image.new("P", (sem_seg.shape[1], sem_seg.shape[0]))
        sem_seg_vis.putpalette(color_pal)
        sem_seg_vis.putdata(sem_seg.flatten().astype(np.uint8))
        sem_seg_vis = sem_seg_vis.convert("RGB")
        sem_seg_vis = np.array(sem_seg_vis)

        sem_seg_vis = sem_seg_vis[:, :, [2, 1, 0]]
        sem_seg_vis = cv2.resize(
            sem_seg_vis, (640, 480), interpolation=cv2.INTER_NEAREST
        )
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        self.vis_image[50:530, 1165:1805] = sem_seg_vis

        pos = (
            (start_x * 100.0 / args.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100.0 / args.map_resolution + gx1)
            * 480
            / map_pred.shape[1],
            np.deg2rad(-start_o),
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (
            int(color_palette[11] * 255),
            int(color_palette[10] * 255),
            int(color_palette[9] * 255),
        )
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if "pf_pred" in inputs:
            # Rescale pf_pred to match the height of vis_image
            vis_maps = inputs["pf_pred"]
            vis_maps_list = [vis_maps["pfs"]]
            if "area_pfs" in vis_maps:
                vis_maps_list.append(vis_maps["raw_pfs"])
                vis_maps_list.append(vis_maps["area_pfs"])
            for i, vis_map in enumerate(vis_maps_list):
                start_x = 1805 + 15 * (i + 1) + 480 * i
                start_y = 50
                end_x = start_x + 480
                end_y = start_y + 480
                vis_map = cv2.resize(vis_map, (480, 480))
                # Apply up-down flipping similar to vis_image
                vis_map = np.flipud(vis_map)
                self.vis_image[start_y:end_y, start_x:end_x] = vis_map[..., ::-1]

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            self.video_writer.append_data(self.vis_image[..., ::-1])
