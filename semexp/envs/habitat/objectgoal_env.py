import _pickle as cPickle
import bz2
import copy
import gzip
import json
import math
import random

import cv2
import gym
import habitat
import networkx as nx
import numpy as np
import quaternion
import semexp.envs.utils.pose as pu
import skimage.morphology

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from semexp.constants import coco_categories

from semexp.envs.utils.fmm_planner import FMMPlanner


inv_coco_categories = {v: k for k, v in coco_categories.items()}


def normalize_angle(angle):
    angle = angle % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


class MultiObjectGoal_Env(habitat.RLEnv):
    """The Multi Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank
        self.cat_offset = self.args.object_cat_offset
        if args.use_gt_segmentation:
            config_env.defrost()
            H = config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT
            W = config_env.SIMULATOR.DEPTH_SENSOR.WIDTH
            hfov = config_env.SIMULATOR.DEPTH_SENSOR.HFOV
            pos = config_env.SIMULATOR.DEPTH_SENSOR.POSITION
            ori = config_env.SIMULATOR.DEPTH_SENSOR.ORIENTATION
            config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = H
            config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = W
            config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = hfov
            config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = pos
            config_env.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = ori
            config_env.SIMULATOR.AGENT_0.SENSORS.append("SEMANTIC_SENSOR")
            config_env.TASK.SEMANTIC_CATEGORY_SENSOR.HEIGHT = H
            config_env.TASK.SEMANTIC_CATEGORY_SENSOR.WIDTH = W
            config_env.TASK.SENSORS.append("SEMANTIC_CATEGORY_SENSOR")
            config_env.freeze()

        super().__init__(config_env, dataset)

        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(split=self.split)

        dataset_info_file = self.episodes_dir + "{split}_info.pbz2".format(
            split=self.split
        )
        with bz2.BZ2File(dataset_info_file, "rb") as f:
            self.dataset_info = cPickle.load(f)

        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            0, 255, (3, args.frame_height, args.frame_width), dtype="uint8"
        )

        # Initializations
        self.episode_no = 0

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.current_eps = None
        self.gt_planners = None
        self.object_boundary = None
        self.goal_idxs = None
        self.active_goal_ix = None
        self.goal_names = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distances = None
        self.starting_greedy_distances = None
        self.optimal_goal_locs = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.called_reached = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.episode_progress = None
        self.episode_progress_dists = None
        self.info = {}
        self.info["ppl"] = None
        self.info["spl"] = None
        self.info["gspl"] = None
        self.info["gppl"] = None
        self.info["success"] = None
        self.info["progress"] = None
        self.info["task_status"] = None
        self.info["ginsts"] = None
        self.info["gcats"] = None
        self.info["oinsts"] = None
        self.info["ocats"] = None
        self.info["goal_distance"] = None
        # States for computing rewards
        self.reward_states = {}
        self.reward_states["object_categories_visited"] = None
        self.reward_states["object_instances_visited"] = None
        self.reward_states["goal_categories_visited"] = None
        self.reward_states["goal_instances_visited"] = None

    def load_new_episode(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """

        args = self.args
        self.scene_path = self.habitat_env.sim.habitat_config.SCENE
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = self.episodes_dir + "content/{}_episodes.json.gz".format(
                scene_name
            )

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, "r") as f:
                self.eps_data = json.loads(f.read().decode("utf-8"))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.current_eps = episode
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        rot = quaternion.from_float_array(episode["start_rotation"])

        goal_names = episode["object_categories"][: self.args.num_goals]
        goal_idxs = episode["object_ids"][: self.args.num_goals]
        floor_idx = episode["floor_id"]

        # Load scene info
        scene_info = self.dataset_info[scene_name]
        sem_map = scene_info[floor_idx]["sem_map"]
        map_obj_origin = scene_info[floor_idx]["origin"]

        # Get starting loc in GT map coordinates
        x, y, o = self.convert_3d_to_2d_pose(pos, rot)
        min_x, min_y = map_obj_origin / 100.0
        map_loc = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)

        # Setup ground truth planner
        object_boundary = args.success_dist
        map_resolution = args.map_resolution
        selem = skimage.morphology.disk(2)
        traversible = cv2.dilate(sem_map[0], selem)
        planners = []
        for i, goal_idx in enumerate(goal_idxs):
            planner = FMMPlanner(traversible)
            selem = skimage.morphology.disk(
                int(object_boundary * 100.0 / map_resolution)
            )
            goal_map = cv2.dilate(sem_map[goal_idx + self.cat_offset], selem)
            planner.set_multi_goal(goal_map, validate_goal=True)
            planners.append(planner)
            success_condition = (
                planner.fmm_dist[int(map_loc[0]), int(map_loc[1])]
                < planner.fmm_dist.max().item()
            )
            #################################### Debugging #########################################
            if not success_condition:
                trav_img = np.clip(traversible * 255, 0, 255).astype(np.uint8)
                trav_img = np.repeat(trav_img[..., np.newaxis], 3, axis=2)
                trav_img_cpy = np.copy(trav_img)
                goal_y, goal_x = np.where(goal_map > 0)
                for gx, gy in zip(goal_x, goal_y):
                    cv2.circle(trav_img_cpy, (int(gx), int(gy)), 4, (0, 255, 0), -1)
                cv2.circle(
                    trav_img_cpy, (int(map_loc[1]), int(map_loc[0])), 4, (255, 0, 0), -1
                )
                trav_img = np.concatenate([trav_img, trav_img_cpy], axis=1)
                trav_img = cv2.resize(trav_img, None, fx=2.0, fy=2.0)
                randint = np.random.randint(0, 50000)
                save_path = f"debug_goals_{randint:07d}.png"
                print(f"========> Writing image to {save_path}")
                cv2.imwrite(save_path, trav_img)
                print(self.current_eps)
                print(
                    f"load_new_episode(): Goal is unreachable from start!\n"
                    f"Distance to goal: {planner.fmm_dist[int(pos[0]), int(pos[1])]:.3f}\n"
                    f"Max distance on map: {planner.fmm_dist.max().item():.3f}\n"
                    f"Goal #{i}\n"
                )
            ########################################################################################

        self.gt_planners = planners
        self.traversible = traversible
        self.common_planner = FMMPlanner(traversible)  # Used for various purposes
        self.starting_loc = map_loc
        self.object_boundary = object_boundary
        self.goal_idxs = goal_idxs
        self.goal_names = goal_names
        self.active_goal_ix = 0
        self.map_obj_origin = map_obj_origin
        self.sem_map = sem_map

        sdists, glocs = self.get_multi_goal_shortest_path_length(traversible)
        gsdists, gglocs = self.get_multi_goal_greedy_path_length(traversible)
        self.starting_distances = sdists
        self.starting_greedy_distances = gsdists
        self.optimal_goal_locs = glocs
        self.optimal_greedy_goal_locs = gglocs
        self.optimal_goal_fmm_dists = []
        for gloc in glocs:
            selem = skimage.morphology.disk(int(0.25 * 100.0 / map_resolution))
            goal_map = np.zeros_like(sem_map[0])
            goal_map[int(gloc[0]), int(gloc[1])] = 1
            goal_map = cv2.dilate(goal_map, selem)
            self.common_planner.set_multi_goal(goal_map, validate_goal=True)
            self.optimal_goal_fmm_dists.append(np.copy(self.common_planner.fmm_dist))
        # Is greedy path also the optimal path?
        self.info["goal_distance"] = self.starting_distances[-1]

        self.prev_distance = (
            self.gt_planners[self.active_goal_ix].fmm_dist[self.starting_loc] / 20.0
            + self.object_boundary
        )
        self._env.sim.set_agent_state(pos, rot)

        # The following two should match approximately
        # print(starting_loc)
        # print(self.sim_continuous_to_sim_map(self.get_sim_location()))

        obs = self._env.sim.get_observations_at(pos, rot)

        return obs

    def generate_new_episode(self):
        """The function generates a random valid episode. This function is used
        for training a model on the train split.
        """

        args = self.args

        self.scene_path = self.habitat_env.sim.habitat_config.SCENE
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        scene_info = self.dataset_info[scene_name]
        map_resolution = args.map_resolution

        floor_idx = random.choice(list(scene_info.keys()))
        floor_height = scene_info[floor_idx]["floor_height"]
        sem_map = scene_info[floor_idx]["sem_map"]
        map_obj_origin = scene_info[floor_idx]["origin"]

        cat_counts = sem_map.sum(2).sum(1)
        possible_cats = list(np.arange(6))

        for i in range(6):
            if cat_counts[i + self.cat_offset] == 0:
                possible_cats.remove(i)

        object_boundary = args.success_dist

        selem = skimage.morphology.disk(2)
        traversible = cv2.dilate(sem_map[0], selem)

        loc_found = False
        while not loc_found:
            if len(possible_cats) < self.args.num_goals:
                print(
                    "Insufficient valid objects for {} in scene {}".format(
                        floor_height, scene_name
                    )
                )
                eps = eps - 1
                continue

            goal_idxs = np.random.permutation(possible_cats).tolist()
            goal_idxs = goal_idxs[: self.args.num_goals]
            goal_names = [inv_coco_categories[goal_idx] for goal_idx in goal_idxs]

            selem = skimage.morphology.disk(
                int(object_boundary * 100.0 / map_resolution)
            )
            planners = []
            # If all goal locations are not traversible, then ignore category
            cats_to_ignore = []
            for goal_idx in goal_idxs:
                goal_map = cv2.dilate(sem_map[goal_idx + self.cat_offset], selem)
                if (goal_map * traversible).sum() == 0:
                    cats_to_ignore.append(goal_idx)
                    break
                planner = FMMPlanner(traversible)
                planner.set_multi_goal(goal_map, validate_goal=True)
                planners.append(planner)
            if len(cats_to_ignore) > 0:
                for cat in cats_to_ignore:
                    print(f"========> Removing unreachable category: {cat}")
                    possible_cats.remove(cat)
                continue

            m1 = sem_map[0] > 0
            m2 = planners[0].fmm_dist > (args.min_d - object_boundary) * 20.0
            m3 = planners[0].fmm_dist < (args.max_d - object_boundary) * 20.0
            # reachability constraints
            m4 = np.ones_like(m3)
            for planner in planners:
                mrch = planner.fmm_dist < planner.fmm_dist.max().item()
                m4 = np.logical_and(m4, mrch)

            possible_starting_locs = np.logical_and(m1, m2)
            possible_starting_locs = np.logical_and(possible_starting_locs, m4)
            possible_starting_locs = np.logical_and(possible_starting_locs, m3) * 1.0
            if possible_starting_locs.sum() != 0:
                loc_found = True
            else:
                print(
                    "Invalid object: {} / {} / {}".format(
                        scene_name, floor_height, goal_names[0]
                    )
                )
                possible_cats.remove(goal_idxs[0])
                scene_info[floor_idx]["sem_map"][
                    goal_idxs[0] + self.cat_offset, :, :
                ] = 0.0
                self.dataset_info[scene_name][floor_idx]["sem_map"][
                    goal_idxs[0] + self.cat_offset, :, :
                ] = 0.0

        loc_found = False
        loop_count = 0
        while not loc_found:
            pos = self._env.sim.sample_navigable_point()
            x = -pos[2]
            y = -pos[0]
            min_x, min_y = map_obj_origin / 100.0
            map_loc = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)
            is_same_floor = abs(pos[1] - floor_height) < args.floor_thr / 100.0
            if is_same_floor and possible_starting_locs[map_loc[0], map_loc[1]] == 1:
                loc_found = True
            loop_count += 1
            if loop_count > 10000 and is_same_floor:
                print("========> Exceeded loop 2 count, selecting random starting loc")
                loc_found = True
                break

        agent_state = self._env.sim.get_agent_state(0)
        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)
        rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        self.gt_planners = planners
        self.traversible = traversible
        self.common_planner = FMMPlanner(traversible)  # Used for various purposes
        self.starting_loc = map_loc
        self.object_boundary = object_boundary
        self.active_goal_ix = 0
        self.goal_idxs = goal_idxs
        self.goal_names = goal_names
        self.map_obj_origin = map_obj_origin
        self.sem_map = sem_map

        sdists, glocs = self.get_multi_goal_shortest_path_length(traversible)
        gsdists, gglocs = self.get_multi_goal_greedy_path_length(traversible)
        self.starting_distances = sdists
        self.starting_greedy_distances = gsdists
        self.optimal_goal_locs = glocs
        self.optimal_greedy_goal_locs = gglocs
        self.optimal_goal_fmm_dists = []
        for gloc in glocs:
            selem = skimage.morphology.disk(int(0.25 * 100.0 / map_resolution))
            goal_map = np.zeros_like(sem_map[0])
            goal_map[int(gloc[0]), int(gloc[1])] = 1
            goal_map = cv2.dilate(goal_map, selem)
            self.common_planner.set_multi_goal(goal_map, validate_goal=True)
            self.optimal_goal_fmm_dists.append(np.copy(self.common_planner.fmm_dist))
        self.info["goal_distance"] = self.starting_distances[-1]

        self.starting_distances = sdists
        self.optimal_goal_locs = glocs
        self.prev_distance = (
            self.gt_planners[self.active_goal_ix].fmm_dist[self.starting_loc] / 20.0
            + self.object_boundary
        )

        self._env.sim.set_agent_state(pos, rot)

        # The following two should match approximately
        # print(starting_loc)
        # print(self.sim_continuous_to_sim_map(self.get_sim_location()))

        obs = self._env.sim.get_observations_at(pos, rot)

        return obs

    def sim_map_to_sim_continuous(self, coords):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position and rotation.
        """
        agent_state = self._env.sim.get_agent_state(0)
        y, x = coords
        min_x, min_y = self.map_obj_origin / 100.0

        cont_x = x / 20.0 + min_x
        cont_y = y / 20.0 + min_y
        agent_state.position[0] = cont_y
        agent_state.position[2] = cont_x

        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)

        if self.args.train_single_eps:
            rvec[1] = 0.0
        else:
            rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        return agent_state.position, rot

    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y, o = sim_loc
        min_x, min_y = self.map_obj_origin / 100.0
        x, y = int((-x - min_x) * 20.0), int((-y - min_y) * 20.0)

        o = np.rad2deg(o) + 180.0
        return y, x, o

    def convert_quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        new_scene = self.episode_no % args.num_train_episodes == 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.called_reached = False
        self.trajectory_states = []
        self.episode_progress = np.zeros((self.args.num_goals,))
        self.episode_progress_dists = np.zeros((self.args.num_goals,))

        if new_scene:
            obs = super().reset()
            next_scene = self.habitat_env.sim.habitat_config.SCENE
            if next_scene != self.scene_name:
                self.scene_name = next_scene
                print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.habitat_config.SCENE

        if self.split == "val" and self.args.eval == 1:
            obs = self.load_new_episode()
        else:
            obs = self.generate_new_episode()

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        state = [rgb, depth]
        if args.use_gt_segmentation:
            state.append(obs["semantic_category"][..., np.newaxis])
        state = np.concatenate(state, axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()
        agent_state = super().habitat_env.sim.get_agent_state(0)
        self.episode_start_position = np.copy(agent_state.position)
        self.episode_start_rotation = copy.deepcopy(agent_state.rotation)

        # Set info
        self.info["time"] = self.timestep
        self.info["sensor_pose"] = [0.0, 0.0, 0.0]
        self.info["gps"] = [0.0, 0.0]
        self.info["compass"] = [0.0]
        self.info["goal_cat_id"] = self.goal_idxs[self.active_goal_ix]
        self.info["goal_name"] = self.goal_names[self.active_goal_ix]

        # Update reward states
        for k in self.reward_states.keys():
            self.reward_states[k] = 0.0

        return state, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            # This is required for reward computation
            self.called_reached = True
            if self.active_goal_ix == self.args.num_goals - 1:
                self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, info = super().step(action)

        # Update goal status
        if self.called_reached:
            curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
            self.called_reached = False
            self.active_goal_ix += 1
            if self.active_goal_ix < self.args.num_goals:
                self.info["goal_cat_id"] = self.goal_idxs[self.active_goal_ix]
                self.info["goal_name"] = self.goal_names[self.active_goal_ix]
                self.prev_distance = (
                    self.gt_planners[self.active_goal_ix].fmm_dist[
                        curr_loc[0], curr_loc[1]
                    ]
                    / 20.0
                )

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info["sensor_pose"] = [dx, dy, do]
        self.info["gps"] = self.get_gps_reading()
        self.info["compass"] = self.get_compass_reading()
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        spl, success, dts, ppl, progress, gspl, gppl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if done:
            spl, success, dts, ppl, progress, gspl, gppl = self.get_metrics()
            self.info["ppl"] = ppl
            self.info["progress"] = progress
            self.info["spl"] = spl
            self.info["success"] = success
            self.info["dts"] = dts
            self.info["gspl"] = gspl
            self.info["gppl"] = gppl

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        state = [rgb, depth]
        if self.args.use_gt_segmentation:
            state.append(obs["semantic_category"][..., np.newaxis])
        state = np.concatenate(state, axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info["time"] = self.timestep

        return state, rew, done, self.info

    def get_gps_reading(self):
        agent_state = super().habitat_env.sim.get_agent_state()

        origin = self.episode_start_position
        rotation_world_start = self.episode_start_rotation

        agent_position = agent_state.position
        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        return [-agent_position[2], agent_position[0]]

    def get_compass_reading(self):
        agent_state = super().habitat_env.sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = self.episode_start_rotation
        compass = self.convert_quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        ).item()
        return [compass]

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0.0, 1.0)

    def get_reward(self, observations):
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        self.curr_distance = (
            self.gt_planners[self.active_goal_ix].fmm_dist[curr_loc[0], curr_loc[1]]
            / 20.0
        )

        reward = (self.prev_distance - self.curr_distance) * self.args.reward_coeff

        if self.called_reached:
            if self.curr_distance <= self.args.success_distance:
                self.episode_progress[self.active_goal_ix] = 1.0
                self.episode_progress_dists[self.active_goal_ix] = self.path_length

        self.prev_distance = self.curr_distance
        return reward

    def get_multi_goal_greedy_path_length(self, traversible):
        return self.get_multi_goal_shortest_path_length(traversible)

    def get_multi_goal_shortest_path_length(self, traversible):
        """This function computes the shortest path length from starting
        position through all the goals in sequence. Since there can be multiple
        success locations for each goal, a graph-search is used to find the
        optimal locations to visit for each goal.
        """
        # get set of valid locations for each goal
        all_goal_locs = [[self.starting_loc]]
        for planner in self.gt_planners:
            # identify set of valid goal locations
            goal_map = (planner.fmm_dist == 0) & (traversible > 0)
            goal_map = goal_map.astype(np.uint8) * 255
            outputs = cv2.connectedComponentsWithStats(goal_map, connectivity=8)
            num_labels, goal_labels = outputs[:2]
            goal_locs = []
            # label 0 is background
            for i in range(1, num_labels):
                contour = cv2.findContours(
                    (goal_labels == i).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_TC89_KCOS,
                )
                contour = contour[0][0][:, 0]
                goal_y = contour[:, 1]
                goal_x = contour[:, 0]
                for gy, gx in zip(goal_y, goal_x):
                    goal_locs.append((gy.item(), gx.item()))
            all_goal_locs.append(goal_locs)
        # measure pairwise distances between goals
        pairwise_dists = []
        planner = FMMPlanner(traversible)
        for i in range(len(all_goal_locs) - 1):
            clocs, nlocs = all_goal_locs[i], np.array(all_goal_locs[i + 1])
            dists = []
            for cloc in clocs:
                planner.set_goal(cloc)
                dists.append(
                    planner.fmm_dist[nlocs[:, 0], nlocs[:, 1]] / 20.0
                    + self.object_boundary
                )
            pairwise_dists.append(np.array(dists))
        ############### measure multi-goal shortest path length ################
        # find the shortest path length connecting the starting_loc to
        # one instance of each goal in order
        graph = nx.DiGraph()
        # add nodes
        node_counts = sum([len(goal_locs) for goal_locs in all_goal_locs])
        graph.add_nodes_from(range(node_counts))
        # add edges
        cntr = 0
        edges = []
        for pdists in pairwise_dists:
            n_curr, n_next = pdists.shape
            edges += [
                (cntr + j, cntr + n_curr + k, pdists[j, k])
                for j in range(n_curr)
                for k in range(n_next)
            ]
            cntr += n_curr
        graph.add_weighted_edges_from(edges)
        # add representative nodes that connect all goal instances to a single node
        # this is useful for computing the shortest path to each goal in the sequence
        cntr = 0
        goal_nodes = []
        for pdists in pairwise_dists:
            n_curr, n_next = pdists.shape
            cntr += n_curr
            pnode = node_counts
            pedges = [(cntr + j, pnode, 0.0) for j in range(n_next)]
            goal_nodes.append(pnode)
            graph.add_node(pnode)
            graph.add_weighted_edges_from(pedges)
            node_counts += 1
        # maintain a mapping from goal node to goal location
        goal_node_to_loc = {}
        cntr = 0
        for goal_locs in all_goal_locs:
            for goal_loc in goal_locs:
                goal_node_to_loc[cntr] = goal_loc
                cntr += 1
        # find shortest path distance to each goal
        dists = []
        final_goal_nodes = []
        for node in goal_nodes:
            spath = nx.shortest_path(graph, source=0, target=node, weight="weight")
            dist = nx.shortest_path_length(
                graph, source=0, target=node, weight="weight"
            )
            dists.append(dist)
            final_goal_nodes.append(spath[-2])
        assert len(spath) == self.args.num_goals + 2, print(
            f"\nFailed! Number of goals in episode: {self.args.num_goals}\n"
            f"Shortest path: {spath}\n"
            f"Shortest path dists: {dists}\n"
            f"Node count: {node_counts}\n"
        )
        ################# get locations of intermediate goals ##################
        final_goal_locs = [goal_node_to_loc[node] for node in final_goal_nodes]

        return dists, final_goal_locs

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat

    def get_shortest_path(self, traversible, start, goal):
        traversible = np.copy(traversible)
        traversible[
            int(start[0]) - 1 : int(start[0]) + 2, int(start[1]) - 1 : int(start[1]) + 2
        ] = 1
        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(4)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal, validate_goal=True)
        # Goal must be reachable from start
        assert (
            planner.fmm_dist[int(start[0]), int(start[1])]
            < planner.fmm_dist.max().item()
        ), (
            "====> MultiObjectGoal_Env: get_shortest_path() failed"
            " since goal was unreachable!"
        )

        curr_loc = start
        spath = [curr_loc]
        ctr = 0
        while True:
            ctr += 1
            if ctr > 100:
                print("get_shortest_path() --- Run into infinite loop!")
            next_y, next_x, _, stop = planner.get_short_term_goal(curr_loc)
            if stop:
                break
            curr_loc = (next_y, next_x)
            spath.append(curr_loc)
        return spath

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        if np.all(self.episode_progress > 0):
            success = 1
        else:
            success = 0
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        dist = self.gt_planners[0].fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
        S = self.starting_distances[-1]
        gS = self.starting_greedy_distances[-1]
        P = self.path_length
        spl = success * S / max(P, S)
        gspl = success * gS / max(P, gS)
        # progress metrics
        num_goals_reached, pS, pGS, pP = 0, 0.0, 0.0, 1e-5
        for prog in self.episode_progress.tolist():
            if prog == 1:
                num_goals_reached += 1
            else:
                break
        if num_goals_reached > 0:
            pS = self.starting_distances[num_goals_reached - 1]
            pGS = self.starting_greedy_distances[num_goals_reached - 1]
            pP = self.episode_progress_dists[num_goals_reached - 1]
        progress = 1.0 * num_goals_reached / self.args.num_goals
        ppl = progress * pS / max(pP, pS)
        gppl = progress * pGS / max(pP, pGS)
        return spl, success, dist, ppl, progress, gspl, gppl

    def get_done(self, observations):
        if self.info["time"] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = self.habitat_env.get_metrics()
        return info

    def get_spaces(self):
        """Returns observation and action spaces for the ObjectGoal task."""
        return self.observation_space, self.action_space

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def convert_3d_to_2d_pose(self, position, rotation):
        x = -position[2]
        y = -position[0]
        axis = quaternion.as_euler_angles(rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
