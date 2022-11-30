import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from global_agent import GlobalAgent

from habitat import Config, logger, VectorEnv
from habitat.utils.visualizations import maps
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
    ObservationBatchingCache,
)
from utils.env_utils import construct_envs
from utils.visualization import generate_video


@baseline_registry.register_trainer(name="transfer_evaluator")
class TransferEvaluator(BaseRLTrainer):
    r"""Evaluator class for hierarchical models
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv

    def __init__(self, config=None):
        super().__init__(config)
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._obs_space = None

        self._obs_batching_cache = ObservationBatchingCache()

        self.using_velocity_ctrl = (self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS) == [
            "VELOCITY_CONTROL"
        ]

        self._synchronize_config()

    def _synchronize_config(self):
        """
        Synchronize the model config with the remaining configs.
        """
        self.config.defrost()
        task_cfg = self.config.TASK_CONFIG
        gcfg = self.config.GLOBAL_AGENT
        ########################################################################
        # Update the planner config
        self.config.PLANNER.n_planners = self.config.NUM_ENVIRONMENTS
        self.config.PLANNER.map_size_cm = gcfg.map_size_cm
        self.config.PLANNER.map_resolution = gcfg.map_resolution
        self.config.PLANNER.turn_angle = task_cfg.SIMULATOR.TURN_ANGLE
        actions = task_cfg.TASK.POSSIBLE_ACTIONS
        self.config.PLANNER.ACTION.stop = actions.index("STOP")
        self.config.PLANNER.ACTION.move_forward = actions.index("MOVE_FORWARD")
        self.config.PLANNER.ACTION.turn_left = actions.index("TURN_LEFT")
        self.config.PLANNER.ACTION.turn_right = actions.index("TURN_RIGHT")
        # Update the image segmentation config
        self.config.IMAGE_SEGMENTATION.sem_gpu_id = self.config.TORCH_GPU_ID
        self.config.IMAGE_SEGMENTATION.min_depth = (
            task_cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        )
        self.config.IMAGE_SEGMENTATION.max_depth = (
            task_cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        )
        # Update the global agent config
        self.config.GLOBAL_AGENT.env_frame_width = task_cfg.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.config.GLOBAL_AGENT.env_frame_height = (
            task_cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT
        )
        self.config.GLOBAL_AGENT.camera_height = (
            task_cfg.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        )
        # GT segmentation
        env_W = self.config.GLOBAL_AGENT.env_frame_width
        env_H = self.config.GLOBAL_AGENT.env_frame_height
        if self.config.SEMANTIC_MAPPING.use_gt_segmentation:
            self.config.TASK_CONFIG.TASK.SENSORS.append("SEMANTIC_CATEGORY_SENSOR")
            self.config.TASK_CONFIG.TASK.SEMANTIC_CATEGORY_SENSOR.HEIGHT = env_H
            self.config.TASK_CONFIG.TASK.SEMANTIC_CATEGORY_SENSOR.WIDTH = env_W
            self.config.SENSORS.append("SEMANTIC_SENSOR")
            self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = env_H
            self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = env_W
            hfov = task_cfg.SIMULATOR.RGB_SENSOR.HFOV
            position = task_cfg.SIMULATOR.RGB_SENSOR.POSITION
            orientation = task_cfg.SIMULATOR.RGB_SENSOR.ORIENTATION
            self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HFOV = hfov
            self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.POSITION = position
            self.config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = orientation

        if len(self.config.VIDEO_OPTION) > 0:
            gcfg.visualize = True
        ## Sanity check
        fr_W = self.config.GLOBAL_AGENT.frame_width
        fr_H = self.config.GLOBAL_AGENT.frame_height
        assert env_W % fr_W == 0
        assert env_H % fr_H == 0
        assert (env_H // fr_H) == (env_W // fr_W)
        ########################################################################
        self.config.freeze()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _setup_actor_critic_agent(self, cfg: Config) -> None:
        r"""Sets up global agent for transfer evaluation.

        Args:
            cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.global_agent = GlobalAgent(cfg, self.device)
        self.obs_space = observation_space

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=False,
        )

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def eval(self) -> None:
        r"""Main evaluation method"""
        ########################################################################
        # Initial setup for evaluation
        ########################################################################
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        writer = TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        )

        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        ########################################################################
        # Perform evaluation
        ########################################################################
        self._init_envs(config)

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0]["VELOCITY_CONTROL"]
            action_shape = (2,)
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long

        self._setup_actor_critic_agent(config)

        observations = self.envs.reset()
        for observation in observations:
            if "semantic" in observation:
                observation["semantic"] = observation["semantic"].astype(np.int32)
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        ########################################################################
        # Initialize agent states. It consists of
        # * full_map
        # * local_map
        # * full_pose
        # * local_pose
        # * origins
        # * lmb
        # * planner_pose_inputs
        # * global_policy_inputs (for caching)
        # * global_orientation (for caching)
        # * global_goals
        # * prev_sim_location
        # * extras (for caching)
        # * wait_env
        # * finished
        ########################################################################
        agent_states = self.global_agent.get_new_agent_states()
        agent_states = self.global_agent.init_map_and_pose(agent_states)
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        global_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        ########################################################################
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        steps = 0
        start_time = time.time()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and agent_states["finished"].sum() < self.envs.num_envs
            # and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (actions, agent_states,) = self.global_agent.act(
                    batch,
                    agent_states,
                    steps,
                    global_masks,
                    not_done_masks,
                )
            steps += 1

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if self.using_velocity_ctrl:
                step_data = [
                    action_to_velocity_control(a) for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            wait_mask = (
                (agent_states["wait_env"] == 1) | (agent_states["finished"] == 1)
            ).tolist()
            outputs = self.envs.step(step_data, wait_mask=wait_mask)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            for observation in observations:
                if "semantic" in observation:
                    observation["semantic"] = observation["semantic"].astype(np.int32)
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            ####################################################################
            # Handle episode termination
            ####################################################################
            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for e, done in enumerate(dones):
                # episode ended
                if done:
                    agent_states["wait_env"][e] = 1.0
                    if (
                        next_episodes[e].scene_id,
                        next_episodes[e].episode_id,
                    ) in stats_episodes:
                        agent_states["finished"][e] = 1.0
                    agent_states = self.global_agent.init_map_and_pose_for_env(
                        agent_states, e
                    )

                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[e].item()
                    episode_stats["goal_distance"] = current_episodes[e].info[
                        "geodesic_distance"
                    ]
                    episode_stats.update(self._extract_scalars_from_info(infos[e]))
                    current_episode_reward[e] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[e].scene_id,
                            current_episodes[e].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[e],
                            scene_id=current_episodes[e].scene_id,
                            episode_id=current_episodes[e].episode_id,
                            checkpoint_idx=0,
                            metrics=self._extract_scalars_from_info(infos[e]),
                            tb_writer=writer,
                        )

                        rgb_frames[e] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0 and infos[e] != {}:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    # frame = observations_to_image(
                    #     {k: v[e] for k, v in batch.items()}, infos[e]
                    # )
                    assert "rgb" in batch.keys()
                    rgb_image = (
                        batch["rgb"][e].cpu().numpy().astype(np.uint8)
                    )  # (H, W, C)
                    frame = self.global_agent.visualize_states(
                        self.global_agent.cached_planner_inputs[e],
                        rgb_image,
                        current_episodes[e].object_category,
                    )
                    # Add GT top-down map
                    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                        infos[e]["top_down_map"], frame.shape[0]
                    )
                    frame = np.concatenate([frame, top_down_map], axis=1)
                    rgb_frames[e].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            global_masks *= not_done_masks

            ####################################################################
            # Logging intermediate statistics
            ####################################################################
            num_episodes = len(stats_episodes)
            if steps % 50 == 0 and num_episodes > 0:
                fps = (steps * self.envs.num_envs) / (time.time() - start_time + 1e-10)
                logger.info(f"========> # steps: {steps}, fps: {fps:6.3f}")
                aggregated_stats = {}
                for stat_key in next(iter(stats_episodes.values())).keys():
                    aggregated_stats[stat_key] = (
                        sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
                    )
                stat_strs = ""
                stat_keys = sorted(list(aggregated_stats.keys()))
                stat_names = []
                for stat_key in stat_keys:
                    stat_names.append(stat_key)
                stat_strs += "/".join(stat_names)
                stat_strs += " : "
                stat_vals = []
                for stat_key in stat_keys:
                    stat_vals.append(f"{aggregated_stats[stat_key]:6.4f}")
                stat_strs += "/".join(stat_vals)
                stat_strs += f" ({num_episodes})"
                logger.info(stat_strs)
            # (
            #     self.envs,
            #     test_recurrent_hidden_states,
            #     not_done_masks,
            #     current_episode_reward,
            #     prev_actions,
            #     batch,
            #     rgb_frames,
            # ) = self._pause_envs(
            #     envs_to_pause,
            #     self.envs,
            #     test_recurrent_hidden_states,
            #     not_done_masks,
            #     current_episode_reward,
            #     prev_actions,
            #     batch,
            #     rgb_frames,
            # )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = 0

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        stat_save_path = f"{self.config.TENSORBOARD_DIR}/stats.json"
        stats_json = {}
        for k, v in stats_episodes.items():
            k_new = f"{k[0].split('/')[-1].split('.')[0]}_{k[1]}"
            stats_json[k_new] = v
        with open(stat_save_path, "w") as fp:
            json.dump(stats_json, fp)

        self.global_agent.close()
        self.envs.close()
