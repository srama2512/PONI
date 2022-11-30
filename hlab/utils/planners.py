import math
import time
from multiprocessing import Pipe, Process

import cv2
import numpy as np
import pyastar
import skimage

# Relies on SemExp codebase
import utils.pose as pu
from poni.fmm_planner import FMMPlanner


# Define commands
PLAN_AND_ACT_COMMAND = "plan_and_act"
REACHABILITY_COMMAND = "get_reachability_map"
FRONTIER_COMMAND = "get_frontier_map"
CLOSE_COMMAND = "close"


class PlannerActor:
    def __init__(self, cfg):
        self.cfg = cfg

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        self.stg_selem = skimage.morphology.disk(self.cfg.stg_disk_size)
        self.stg_selem_close = skimage.morphology.disk(self.cfg.stg_disk_size // 2)

        self.collision_map = None
        self.visited = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.conseq_replans = None
        self.is_close_to_goal = False
        self.is_close_time = None

    def reset(self):

        cfg = self.cfg
        # Initialize maps
        map_shape = (
            cfg.map_size_cm // cfg.map_resolution,
            cfg.map_size_cm // cfg.map_resolution,
        )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.col_width = 1
        self.curr_loc = [
            cfg.map_size_cm / 100.0 / 2.0,
            cfg.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.last_action_collision = False
        self.is_close_to_goal = False
        self.is_close_time = 0
        self.conseq_replans = 0

    def plan_and_act(self, planner_inputs):
        """Function responsible for planning, and sampling the action

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                    and planning window (gx1, gx2, gy1, gy2)
                    'found_goal'   (bool): whether the goal object is found

        Returns:
            action (ndarray): (1, ) action sampled by planner
        """
        if planner_inputs["wait"]:
            self.last_action = None
            return np.array([-1]), False

        cfg = self.cfg

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs["map_pred"].astype(np.float32))
        goal = np.rint(planner_inputs["goal"].astype(np.float32))

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / cfg.map_resolution - gx1),
            int(c * 100.0 / cfg.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        # Collision check
        self.last_action_collision = False
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
            if dist < cfg.collision_threshold:  # Collision
                self.last_action_collision = True
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
                        r, c = int(r * 100 / cfg.map_resolution), int(
                            c * 100 / cfg.map_resolution
                        )
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, replan, stop = self._get_stg(
            map_pred,
            start,
            np.copy(goal),
            planning_window,
        )

        # Deterministic Local Policy
        if (
            self.cfg.move_as_close_as_possible
            and self.is_close_to_goal
            and (
                stop
                or self.last_action_collision
                or replan
                or self.is_close_time >= self.cfg.move_close_limit
            )
        ):
            if self.is_close_time >= self.cfg.move_close_limit:
                print("=====> Unable to reach closer. Reached time limit.")
            else:
                print("======> Reached as close as possible")
            action = cfg.ACTION.stop
        elif (
            (not self.cfg.move_as_close_as_possible)
            and stop
            and planner_inputs["found_goal"] == 1
        ):
            action = cfg.ACTION.stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.cfg.turn_angle / 2.0:
                action = cfg.ACTION.turn_right
            elif relative_angle < -self.cfg.turn_angle / 2.0:
                action = cfg.ACTION.turn_left
            else:
                action = cfg.ACTION.move_forward
            if self.is_close_to_goal:
                print(f"======> Reached close to object, but still acting: {action}")

        if stop and planner_inputs["found_goal"]:
            self.is_close_to_goal = True
        if self.is_close_to_goal:
            self.is_close_time += 1

        action = np.array([action])
        # TODO(SR) - does this have any edge cases?
        self.last_action = action
        if replan:
            self.conseq_replans += 1
        else:
            self.conseq_replans = 0
        replan = True if self.conseq_replans >= self.cfg.conseq_replan_thresh else False

        return (action, replan)

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        # traversible = skimage.morphology.binary_dilation(
        #     grid[x1:x2, y1:y2],
        #     self.selem) != True
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        s = self.cfg.stg_downsampling
        traversible[
            int(start[0] - x1) - s : int(start[0] - x1) + s + 1,
            int(start[1] - y1) - s : int(start[1] - y1) + s + 1,
        ] = 1

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)

        scale = self.cfg.stg_downsampling
        step_size = 5 if scale == 1 else 3 * scale
        selem = self.stg_selem
        if self.cfg.move_as_close_as_possible and self.is_close_to_goal:
            selem = self.stg_selem_close
        goal = cv2.dilate(goal, selem)
        # if True:
        #     planner = FMMPlanner(traversible, scale=scale, step_size=step_size)
        #     planner.set_multi_goal(goal)

        #     state = [start[0] - x1 + 1, start[1] - y1 + 1]
        #     stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        if True:
            stg_x, stg_y = None, None
            obstacles = (1 - traversible).astype(np.float32)
            astar_goal = goal.astype(np.float32)
            astar_start = [int(start[1] - y1 + 1), int(start[0] - x1 + 1)]
            if scale != 1:
                obstacles = obstacles[::scale]
                astar_goal = astar_goal[::scale]
                astar_start = [astar_start[0] // scale, astar_start[1] // scale]
                step_size = step_size // scale
            path_y, path_x = pyastar.multi_goal_weighted_astar_planner(
                obstacles,
                astar_start,
                astar_goal,
                True,
                wscale=self.cfg.weighted_scale,
                niters=self.cfg.weighted_niters,
            )
            if scale != 1:
                path_y = [y * scale for y in path_y]
                path_x = [x * scale for x in path_x]
            if path_x is not None:
                # The paths are in reversed order
                stg_x = path_x[-min(step_size, len(path_x))]
                stg_y = path_y[-min(step_size, len(path_y))]
                replan = False
                stop = False
                if len(path_x) < step_size:
                    # Measure distance along the shortest path
                    path_xy = np.stack([path_x, path_y], axis=1)
                    d2g = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
                    d2g = d2g.sum() * self.cfg.map_resolution / 100.0  # In meters
                    if d2g <= 0.25:
                        stop = True

            if stg_x is None:
                # Pick some arbitrary location as the short-term goal
                random_theta = np.random.uniform(-np.pi, np.pi, (1,))[0].item()
                stg_x = int(step_size * np.cos(random_theta))
                stg_y = int(step_size * np.sin(random_theta))
                replan = True
                stop = False

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat

    def get_reachability_map(self, planner_inputs):
        """Function responsible for planning, and identifying reachable locations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)

        Returns:
            reachability_map (ndarray): (M, M) boolean map of reachable locations
            fmm_dist (ndarray): (M, M) integer map of geodesic distance (cm)
        """
        # Get agent position + local boundaries
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        # Get map
        map_pred = np.rint(planner_inputs["map_pred"]).astype(np.float32)
        # Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / self.cfg.map_resolution - gx1),
            int(c * 100.0 / self.cfg.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)
        # Create a goal map (start is goal)
        goal_map = np.zeros(map_pred.shape)
        goal_map[start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1] = 1
        # Figure out reachable locations
        reachability, fmm_dist = self._get_reachability(
            map_pred, goal_map, planning_window
        )

        return reachability, fmm_dist

    def _get_reachability(self, grid, goal, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1 - cv2.dilate(grid, self.selem)
        if self.collision_map is not None:
            traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(3)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = (planner.fmm_dist * self.cfg.map_resolution).astype(np.int32)

        reachability = fmm_dist < fmm_dist.max()

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
        cfg = self.cfg

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
            int(r * 100.0 / cfg.map_resolution - gx1),
            int(c * 100.0 / cfg.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, frontiers.shape)
        ## Mask out a 100.0 x 100.0 cm region center on the agent
        ncells = int(100.0 / cfg.map_resolution)
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


def worker(remote, parent_remote, planner_cfg, worker_id):
    parent_remote.close()
    planner = PlannerActor(planner_cfg)
    while True:
        cmd, data = remote.recv()
        if cmd == PLAN_AND_ACT_COMMAND:
            planner_inputs, mask = data
            if mask == 0:
                planner.reset()
            action, replan = planner.plan_and_act(planner_inputs)
            remote.send((action, replan))
        elif cmd == REACHABILITY_COMMAND:
            planner_inputs, mask = data
            if mask == 0:
                planner.reset()
            rmap, fmm_dist = planner.get_reachability_map(planner_inputs)
            remote.send((rmap, fmm_dist))
        elif cmd == FRONTIER_COMMAND:
            planner_inputs, mask = data
            if mask == 0:
                planner.reset()
            frontiers = planner.get_frontier_map(planner_inputs)
            remote.send(frontiers)
        elif cmd == CLOSE_COMMAND:
            remote.close()
            break
        else:
            raise ValueError("Unknown command")


class PlannerActorVector(object):
    def __init__(self, cfg):
        self.cfg = cfg
        n_planners = cfg.n_planners
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_planners)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, cfg, worker_id))
            for (work_remote, remote, worker_id) in zip(
                self.work_remotes, self.remotes, range(n_planners)
            )
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def call_async(self, command, inputs, masks):
        self._assert_not_closed()
        for remote, input, mask in zip(self.remotes, inputs, masks):
            remote.send((command, (input, mask)))
        self.waiting = True

    def call_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def plan_and_act(self, planner_inputs_all, masks):
        self.call_async(PLAN_AND_ACT_COMMAND, planner_inputs_all, masks)
        outputs = self.call_wait()
        actions, replan = zip(*outputs)
        actions = np.stack(actions, axis=0)  # (N, 1)
        replan = np.array(replan)  # (N, ) boolen array
        return actions, replan

    def get_reachability_maps(self, inputs, masks):
        self.call_async(REACHABILITY_COMMAND, inputs, masks)
        outputs = self.call_wait()
        reachability_maps, fmm_dists = list(zip(*outputs))
        reachability_maps = np.stack(reachability_maps, axis=0)
        fmm_dists = np.stack(fmm_dists, axis=0)
        return reachability_maps, fmm_dists

    def get_frontier_maps(self, inputs, masks):
        self.call_async(FRONTIER_COMMAND, inputs, masks)
        outputs = self.call_wait()
        frontier_maps = np.stack(outputs, axis=0)
        return frontier_maps

    def close(self):
        for remote in self.remotes:
            remote.send((CLOSE_COMMAND, None))

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on an PlannerActorVector after calling close()"


def worker_fn(planner, cmd, data):
    if cmd == PLAN_AND_ACT_COMMAND:
        planner_inputs, mask = data
        if mask == 0:
            planner.reset()
        action, replan = planner.plan_and_act(planner_inputs)
        return action, replan
    elif cmd == REACHABILITY_COMMAND:
        planner_inputs, mask = data
        if mask == 0:
            planner.reset()
        rmap, fmm_dist = planner.get_reachability_map(planner_inputs)
        return (rmap, fmm_dist)
    elif cmd == FRONTIER_COMMAND:
        planner_inputs, mask = data
        if mask == 0:
            planner.reset()
        frontiers = planner.get_frontier_map(planner_inputs)
        return frontiers
    elif cmd == CLOSE_COMMAND:
        pass
    else:
        raise ValueError("Unknown command")


class PlannerActorSequential(object):
    def __init__(self, cfg):
        self.cfg = cfg
        n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActor(cfg) for _ in range(n_planners)]

    def call(self, command, inputs, masks):
        self._assert_not_closed()
        results = [
            worker_fn(planner, command, (inp, mask))
            for planner, inp, mask in zip(self.planners, inputs, masks)
        ]
        return results

    def plan_and_act(self, planner_inputs_all, masks):
        outputs = self.call(PLAN_AND_ACT_COMMAND, planner_inputs_all, masks)
        actions, replan = zip(*outputs)
        actions = np.stack(actions, axis=0)  # (N, 1)
        replan = np.array(replan)  # (N, )
        return actions, replan

    def get_reachability_maps(self, inputs, masks):
        outputs = self.call(REACHABILITY_COMMAND, inputs, masks)
        reachability_maps, fmm_dists = list(zip(*outputs))
        reachability_maps = np.stack(reachability_maps, axis=0)
        fmm_dists = np.stack(fmm_dists, axis=0)
        return reachability_maps, fmm_dists

    def get_frontier_maps(self, inputs, masks):
        outputs = self.call(FRONTIER_COMMAND, inputs, masks)
        frontier_maps = np.stack(outputs, axis=0)
        return frontier_maps

    def close(self):
        self.closed = True

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on an PlannerActorVector after calling close()"
