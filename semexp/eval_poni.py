import json
import logging
import math
import os
import re
import time
from collections import deque

import gym
import numpy as np

import semexp.envs.utils.pose as pu
import torch
import torch.nn as nn
from semexp.arguments import get_args
from semexp.envs import make_vec_envs

from semexp.model import Semantic_Mapping
from semexp.model_pf import RL_Policy
from semexp.utils.storage import GlobalRolloutStorage
from torch.utils.tensorboard import SummaryWriter


os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def main(args=None):
    if args is None:
        args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    tb_dir = "{}/tb/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    if (not os.path.exists(tb_dir)) and (not args.eval):
        os.makedirs(tb_dir)

    logging.basicConfig(
        filename=log_dir + "train.log", level=logging.INFO, filemode="a"
    )
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)
    if not args.eval:
        writer = SummaryWriter(log_dir=tb_dir)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    METRICS = [
        "success",
        "dts",
        "gspl",
        "spl",
        "progress",
        "gppl",
        "ppl",
        "goal_distance",
    ]
    if args.eval:
        episode_metrics = {
            m: [deque(maxlen=num_episodes) for _ in range(args.num_processes)]
            for m in METRICS
        }
    else:
        episode_metrics = {m: deque(maxlen=1000) for m in METRICS}

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args, workers_ignore_signals=not args.eval)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Explored Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Sanity check
    ## This is critical since we use the local_map for PF prediction, and
    ## use args.map_resolution as the resolution for GT PF baseline
    assert args.global_downscaling == 1
    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    assert args.global_downscaling == 1

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
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

    def init_map_and_pose():
        full_map.fill_(0.0)
        full_pose.fill_(0.0)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution),
            ]

            full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries(
                (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
            )

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [
                lmb[e][2] * args.map_resolution / 100.0,
                lmb[e][0] * args.map_resolution / 100.0,
                0.0,
            ]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
            local_pose[e] = (
                full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
            )

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.0)
        full_pose[e].fill_(0.0)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
        )

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [
            lmb[e][2] * args.map_resolution / 100.0,
            lmb[e][0] * args.map_resolution / 100.0,
            0.0,
        ]

        local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
        local_pose[e] = full_pose[e] - torch.from_numpy(origins[e]).to(device).float()

    def update_intrinsic_rew(e):
        prev_explored_area = full_map[e, 1].sum(1).sum(0)
        full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = local_map[e]
        curr_explored_area = full_map[e, 1].sum(1).sum(0)
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.0) ** 2  # to m^2

    init_map_and_pose()

    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    es = 2
    g_observation_space = gym.spaces.Box(0, 1, (ngc, local_w, local_h), dtype="uint8")

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy
    g_policy = RL_Policy(args, args.pf_model_path).to(device)
    needs_egocentric_transform = g_policy.needs_egocentric_transform
    if needs_egocentric_transform:
        print("\n\n=======> Needs egocentric transformation!")
    needs_dist_maps = args.add_agent2loc_distance or args.add_agent2loc_distance_v2

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, es)

    g_rollouts = GlobalRolloutStorage(
        args.num_global_steps,
        num_scenes,
        g_observation_space.shape,
        g_action_space,
        g_policy.rec_state_size,
        es,
    ).to(device)

    assert args.eval, "Only evaluation enabled for PF model"
    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = (
        torch.from_numpy(
            np.asarray([infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)])
        )
        .float()
        .to(device)
    )

    _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        local_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :]
    )
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
    goal_cat_id = torch.from_numpy(
        np.asarray([infos[env_idx]["goal_cat_id"] for env_idx in range(num_scenes)])
    )

    extras = torch.zeros(num_scenes, es)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    # Get fmm distance from agent in predicted map
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        obs_map = local_map[e, 0, :, :].cpu().numpy()
        exp_map = local_map[e, 1, :, :].cpu().numpy()
        # set unexplored to navigable by default
        p_input["map_pred"] = obs_map * np.rint(exp_map)
        p_input["pose_pred"] = planner_pose_inputs[e]
    _, fmm_dists = envs.get_reachability_map(planner_inputs)

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    agent_locations = []
    for e in range(num_scenes):
        pose_pred = planner_pose_inputs[e]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        map_r, map_c = start_y, start_x
        map_loc = [
            int(map_r * 100.0 / args.map_resolution - gx1),
            int(map_c * 100.0 / args.map_resolution - gy1),
        ]
        map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
        agent_locations.append(map_loc)

    ################################################################################################
    # Transform to egocentric coordinates if needed
    # Note: The agent needs to be at the center of the map facing rightward
    # Conventions: start_x, start_y, start_o are as follows.
    # X -> downward, Y -> rightward, origin (top-left corner of map)
    # O -> measured from Y to X clockwise.
    ################################################################################################
    # Perform transformations if needed
    g_obs = g_rollouts.obs[0]
    g_obs_old = g_obs
    unk_map = 1.0 - local_map[:, 1, :, :]
    ego_agent_poses = None
    if needs_egocentric_transform:
        ego_agent_poses = []
        for e in range(num_scenes):
            map_loc = agent_locations[e]
            # Crop map about a center
            # Note conventions shift for crop fn: X is right and Y is down.
            ego_agent_poses.append([map_loc[0], map_loc[1], math.radians(start_o)])
        ego_agent_poses = torch.Tensor(ego_agent_poses).to(g_obs.device)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states, prev_pfs = g_policy.act(
        g_obs,
        g_rollouts.rec_states[0],
        g_rollouts.masks[0],
        extras=g_rollouts.extras[0],
        extra_maps={
            "dmap": fmm_dists,
            "umap": unk_map,
            "agent_locations": agent_locations,
            "ego_agent_poses": ego_agent_poses,
        },
        deterministic=False,
    )

    if not g_policy.has_action_output:
        cpu_actions = g_action.cpu().numpy()
        if len(cpu_actions.shape) == 2:  # (B, 2) XY locations
            global_goals = [
                [int(action[0] * local_w), int(action[1] * local_h)]
                for action in cpu_actions
            ]
            global_goals = [
                [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                for x, y in global_goals
            ]
        else:
            assert len(cpu_actions.shape) == 3  # (B, H, W) action map
            global_goals = None

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    if not g_policy.has_action_output:
        # Ignore goal and use nearest frontier baseline if requested
        if not args.use_nearest_frontier:
            for e in range(num_scenes):
                if global_goals is not None:
                    goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                else:
                    goal_maps[e][:, :] = cpu_actions[e]
        else:
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                obs_map = local_map[e, 0, :, :].cpu().numpy()
                exp_map = local_map[e, 1, :, :].cpu().numpy()
                p_input["obs_map"] = obs_map
                p_input["exp_map"] = exp_map
                p_input["pose_pred"] = planner_pose_inputs[e]
            frontier_maps = envs.get_frontier_map(planner_inputs)
            for e in range(num_scenes):
                fmap = frontier_maps[e].cpu().numpy()
                goal_maps[e][fmap] = 1

    # if False:
    #     ############################################################################################
    #     # Visualize for debugging
    #     ############################################################################################
    #     # Process maps
    #     g_obs_old = g_policy.do_proc(g_obs_old) # (B, N, H, W)
    #     g_obs_new = g_policy.do_proc(g_obs) # (B, N, H, W)
    #     # Convert to RGB
    #     g_obs_old = g_obs_old.cpu().numpy()
    #     g_obs_new = g_obs_new.cpu().numpy()
    #     for e in range(g_obs_old.shape[0]):
    #         if e > 1:
    #             break
    #         g_obs_old_rgb = PFDataset.visualize_map(g_obs_old[e])
    #         g_obs_new_rgb = PFDataset.visualize_map(g_obs_new[e])
    #         # Mark agent positions on the map
    #         cx, cy = center_locs[e]
    #         cx, cy = int(cx), int(cy)
    #         g_obs_old_rgb[cx - 5 : cx + 6, cy - 5 : cy + 6, :] = np.array([255, 0, 0])
    #         cx, cy = g_obs_new_rgb.shape[0] // 2, g_obs_new_rgb.shape[1] // 2
    #         cx, cy = int(cx), int(cy)
    #         g_obs_new_rgb[cx - 5 : cx + 6, cy - 5 : cy + 6, :] = np.array([255, 0, 0])
    #         # Mark goal positions on the map
    #         cx, cy = global_goals[e]
    #         g_obs_old_rgb[cx - 5 : cx + 6, cy - 5 : cy + 6, :] = np.array([0, 0, 255])
    #         cx, cy = global_goals_new[e]
    #         g_obs_new_rgb[cx - 5 : cx + 6, cy - 5 : cy + 6, :] = np.array([0, 0, 255])
    #         imageio.imwrite(
    #             os.path.join(
    #                 dump_dir, 'proc_{}_old_semmap_{:05d}.png'.format(e, 0)
    #             ),
    #             g_obs_old_rgb
    #         )
    #         imageio.imwrite(
    #             os.path.join(
    #                 dump_dir, 'proc_{}_new_semmap_{:05d}.png'.format(e, 0)
    #             ),
    #             g_obs_new_rgb
    #         )

    planner_inputs = [{} for e in range(num_scenes)]

    pf_visualizations = None
    if args.visualize or args.print_images:
        pf_visualizations = g_policy.visualizations
    for e, p_input in enumerate(planner_inputs):
        p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
        p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
        p_input["pose_pred"] = planner_pose_inputs[e]
        p_input["goal"] = goal_maps[e]  # global_goals[e]
        p_input["new_goal"] = 1
        p_input["found_goal"] = 0
        p_input["wait"] = wait_env[e] or finished[e]
        if g_policy.has_action_output:
            p_input["atomic_action"] = g_action[e]
            pass

        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
            p_input["pf_pred"] = pf_visualizations[e]
            obs[e, -1, :, :] = 1e-5
            p_input["sem_seg"] = obs[e, 4:].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)

    steps_max = args.num_training_frames // args.num_processes + 1
    for step in range(0, steps_max):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                # Update metrics
                for m in METRICS:
                    v = infos[e][m]
                    if args.eval:
                        episode_metrics[m][e].append(v)
                    else:
                        episode_metrics[m].append(v)
                if args.eval:
                    if len(episode_metrics["success"][e]) == num_episodes:
                        finished[e] = 1

                wait_env[e] = 1.0
                update_intrinsic_rew(e)
                init_map_and_pose_for_env(e)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = (
            torch.from_numpy(
                np.asarray(
                    [infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)]
                )
            )
            .float()
            .to(device)
        )

        _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution),
            ]
            local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.0
                else:
                    update_intrinsic_rew(e)

                full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ] = local_map[e]
                full_pose[e] = (
                    local_pose[e] + torch.from_numpy(origins[e]).to(device).float()
                )

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / args.map_resolution),
                    int(c * 100.0 / args.map_resolution),
                ]

                lmb[e] = get_local_map_boundaries(
                    (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
                )

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [
                    lmb[e][2] * args.map_resolution / 100.0,
                    lmb[e][0] * args.map_resolution / 100.0,
                    0.0,
                ]

                local_map[e] = full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]
                local_pose[e] = (
                    full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
                )

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
                full_map[:, 0:4, :, :]
            )
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(
                np.asarray(
                    [infos[env_idx]["goal_cat_id"] for env_idx in range(num_scenes)]
                )
            )
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            # Get exploration reward and metrics
            g_reward = (
                torch.from_numpy(
                    np.asarray(
                        [infos[env_idx]["g_reward"] for env_idx in range(num_scenes)]
                    )
                )
                .float()
                .to(device)
            )
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # # Add samples to global policy storage
            # if step == 0:
            #     g_rollouts.obs[0].copy_(global_input)
            #     g_rollouts.extras[0].copy_(extras)
            # else:
            #     g_rollouts.insert(
            #         global_input, g_rec_states,
            #         g_action, g_action_log_prob, g_value,
            #         g_reward, g_masks, extras
            #     )

            # Get fmm_dists from agent in predicted map
            fmm_dists = None
            if needs_dist_maps:
                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    obs_map = local_map[e, 0, :, :].cpu().numpy()
                    exp_map = local_map[e, 1, :, :].cpu().numpy()
                    # set unexplored to navigable by default
                    p_input["map_pred"] = obs_map * np.rint(exp_map)
                    p_input["pose_pred"] = planner_pose_inputs[e]
                _, fmm_dists = envs.get_reachability_map(planner_inputs)

            agent_locations = []
            for e in range(num_scenes):
                pose_pred = planner_pose_inputs[e]
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                map_r, map_c = start_y, start_x
                map_loc = [
                    int(map_r * 100.0 / args.map_resolution - gx1),
                    int(map_c * 100.0 / args.map_resolution - gy1),
                ]
                map_loc = pu.threshold_poses(map_loc, global_input[e].shape[1:])
                agent_locations.append(map_loc)

            ########################################################################################
            # Transform to egocentric coordinates if needed
            # Note: The agent needs to be at the center of the map facing right.
            # Conventions: start_x, start_y, start_o are as follows.
            # X -> downward, Y -> rightward, origin (top-left corner of map)
            # O -> measured from Y to X clockwise.
            ########################################################################################
            g_obs = global_input.to(local_map.device)  # g_rollouts.obs[g_step]
            unk_map = 1.0 - local_map[:, 1, :, :]
            ego_agent_poses = None
            if needs_egocentric_transform:
                ego_agent_poses = []
                for e in range(num_scenes):
                    map_loc = agent_locations[e]
                    # Crop map about a center
                    ego_agent_poses.append(
                        [map_loc[0], map_loc[1], math.radians(start_o)]
                    )
                ego_agent_poses = torch.Tensor(ego_agent_poses).to(g_obs.device)

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states, prev_pfs = g_policy.act(
                g_obs,
                None,  # g_rollouts.rec_states[g_step],
                g_masks.to(g_obs.device),  # g_rollouts.masks[g_step],
                extras=extras.to(g_obs.device),  # g_rollouts.extras[g_step],
                extra_maps={
                    "dmap": fmm_dists,
                    "umap": unk_map,
                    "pfs": prev_pfs,
                    "agent_locations": agent_locations,
                    "ego_agent_poses": ego_agent_poses,
                },
                deterministic=False,
            )

            if not g_policy.has_action_output:
                cpu_actions = g_action.cpu().numpy()
                if len(cpu_actions.shape) == 2:  # (B, 2) XY locations
                    global_goals = [
                        [int(action[0] * local_w), int(action[1] * local_h)]
                        for action in cpu_actions
                    ]
                    global_goals = [
                        [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                        for x, y in global_goals
                    ]
                else:
                    assert len(cpu_actions.shape) == 3  # (B, H, W) action maps
                    global_goals = None

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

            # Compute frontiers if needed
            if args.use_nearest_frontier:
                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    obs_map = local_map[e, 0, :, :].cpu().numpy()
                    exp_map = local_map[e, 1, :, :].cpu().numpy()
                    p_input["obs_map"] = obs_map
                    p_input["exp_map"] = exp_map
                    p_input["pose_pred"] = planner_pose_inputs[e]
                frontier_maps = envs.get_frontier_map(planner_inputs)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        if not g_policy.has_action_output:
            # Ignore goal and use nearest frontier baseline if requested
            if not args.use_nearest_frontier:
                for e in range(num_scenes):
                    if global_goals is not None:
                        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                    else:
                        goal_maps[e][:, :] = cpu_actions[e]
            else:
                for e in range(num_scenes):
                    fmap = frontier_maps[e].cpu().numpy()
                    goal_maps[e][fmap] = 1
                    # #================ Visualize for debugging ======================
                    # kernel = np.ones((5, 5), dtype=np.uint8)
                    # fmap = cv2.morphologyEx((fmap * 255.0).astype(np.uint8), cv2.MORPH_DILATE, kernel)
                    # obs_map = np.rint(planner_inputs[e]['obs_map'])
                    # exp_map = np.rint(planner_inputs[e]['exp_map'])
                    # vis_map = np.zeros((*obs_map.shape, 3), dtype=np.uint8)
                    # # Green is free
                    # vis_map[:, :, 1] = (((1 - obs_map) * exp_map) * 255.0).astype(np.uint8)
                    # # Blue is obstacles
                    # vis_map[:, :, 0] = (obs_map * exp_map * 255.0).astype(np.uint8)
                    # # Red is frontier
                    # vis_map[:, :, 2] = fmap
                    # vis_map[fmap > 0, 1]  = 0
                    # cv2.imshow("Frontier map", vis_map)
                    # cv2.waitKey(0)

        for e in range(num_scenes):
            cn = infos[e]["goal_cat_id"] + 4
            cat_semantic_map = local_map[e, cn, :, :]

            if cat_semantic_map.sum() != 0.0:
                cat_semantic_map = cat_semantic_map.cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        pf_visualizations = None
        if args.visualize or args.print_images:
            pf_visualizations = g_policy.visualizations
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = planner_pose_inputs[e]
            p_input["goal"] = goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = l_step == args.num_local_steps - 1
            p_input["found_goal"] = found_goal[e]
            p_input["wait"] = wait_env[e] or finished[e]
            if g_policy.has_action_output:
                p_input["atomic_action"] = g_action[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                p_input["pf_pred"] = pf_visualizations[e]
                obs[e, -1, :, :] = 1e-5
                p_input["sem_seg"] = obs[e, 4:].argmax(0).cpu().numpy()

        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            fps = int((step) * num_scenes / (end - start))
            log = " ".join(
                [
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(step * num_scenes),
                    "FPS {},".format(fps),
                ]
            )
            if not args.eval:
                tbitr = step * num_scenes
                writer.add_scalar("FPS", fps, tbitr)

            log += "\n\tRewards:"

            if len(g_episode_rewards) > 0:
                log += " ".join(
                    [
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards), np.median(per_step_g_rewards)
                        ),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards),
                        ),
                    ]
                )
                if not args.eval:
                    tbitr = step * num_scenes
                    writer.add_scalar(
                        "StepRewards/mean", np.mean(per_step_g_rewards), tbitr
                    )
                    writer.add_scalar(
                        "StepRewards/median", np.median(per_step_g_rewards), tbitr
                    )
                    writer.add_scalar(
                        "EpisodeRewards/mean", np.mean(g_episode_rewards), tbitr
                    )
                    writer.add_scalar(
                        "EpisodeRewards/median", np.median(g_episode_rewards), tbitr
                    )
                    writer.add_scalar(
                        "EpisodeRewards/min", np.min(g_episode_rewards), tbitr
                    )
                    writer.add_scalar(
                        "EpisodeRewards/max", np.max(g_episode_rewards), tbitr
                    )

            if args.eval:
                total_metrics = {m: [] for m in METRICS}
                for m in METRICS:
                    for e in range(args.num_processes):
                        total_metrics[m] += episode_metrics[m][e]
                # Log full objectnav metrics
                if len(total_metrics["success"]) > 0:
                    metrics_str = "/".join([m[:4] for m in METRICS])
                    values_float = [np.mean(total_metrics[m]) for m in METRICS]
                    values_str = "/".join([f"{v:.3f}" for v in values_float])
                    count_str = "{:.0f}".format(len(total_metrics["spl"]))
                    log += f"\n===> ObjectNav (full) {metrics_str}: {values_str}({count_str})"
            else:
                # Log full objectnav metrics
                if len(episode_metrics["success"]) > 100:
                    metrics_str = "/".join([m[:4] for m in METRICS])
                    values_float = [np.mean(episode_metrics[m]) for m in METRICS]
                    values_str = "/".join([f"{v:.3f}" for v in values_float])
                    count_str = "{:.0f}".format(len(episode_metrics["spl"]))
                    log += f"\n===> ObjectNav (full) {metrics_str}: {values_str}({count_str})"
                    tbitr = step * num_scenes
                    for m in METRICS:
                        writer.add_scalar(
                            f"Metric/{m}", np.mean(episode_metrics[m]), tbitr
                        )

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join(
                    [
                        " Policy Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies),
                        ),
                    ]
                )
                tbitr = step * num_scenes
                writer.add_scalar("Losses/value", np.mean(g_value_losses), tbitr)
                writer.add_scalar("Losses/action", np.mean(g_action_losses), tbitr)
                writer.add_scalar(
                    "Losses/dist_entropy", np.mean(g_dist_entropies), tbitr
                )

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")

        total_metrics = {m: [] for m in METRICS}
        for m in METRICS:
            for e in range(args.num_processes):
                total_metrics[m] += episode_metrics[m][e]

        # Log full objectnav metrics
        if len(total_metrics["success"]) > 0:
            metrics_str = "/".join([m[:4] for m in METRICS])
            values_float = [np.mean(total_metrics[m]) for m in METRICS]
            values_str = "/".join([f"{v:.3f}" for v in values_float])
            count_str = "{:.0f}".format(len(total_metrics["spl"]))
            log += f"\nFinal ObjectNav (full) {metrics_str}: {values_str}({count_str})"

        # Dump metrics if evaluating periodically
        save_data = None
        if os.path.isfile(args.load) and "periodic" in os.path.basename(args.load):
            match = re.search("periodic_(.*).pth", args.load)
            assert match is not None
            ckpt_steps = int(match.group(1))
            save_path = os.path.join(dump_dir, f"eval_periodic_{ckpt_steps:08d}.json")
        else:
            save_path = os.path.join(dump_dir, f"final_eval_stats.json")
            ckpt_steps = None
        save_data = {
            "total_metrics": {k: np.mean(v).item() for k, v in total_metrics.items()},
            "total_steps": ckpt_steps,
            "total_raw_metrics": {k: v for k, v in total_metrics.items()},
        }
        json.dump(save_data, open(save_path, "w"))

        print(log)
        logging.info(log)

        print(log)
        logging.info(log)

        return save_data


if __name__ == "__main__":
    main()
