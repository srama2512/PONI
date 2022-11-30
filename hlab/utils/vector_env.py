import signal
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gym

import habitat
import numpy as np
from habitat.core.logging import logger
from habitat.core.vector_env import VectorEnv as HabitatVectorEnv
from habitat.utils import profiling_wrapper


STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
CALL_COMMAND = "call"
COUNT_EPISODES_COMMAND = "count_episodes"

EPISODE_OVER_NAME = "episode_over"
GET_METRICS_NAME = "get_metrics"
CURRENT_EPISODE_NAME = "current_episode"
NUMBER_OF_EPISODE_NAME = "number_of_episodes"
ACTION_SPACE_NAME = "action_space"
OBSERVATION_SPACE_NAME = "observation_space"


class VectorEnv(HabitatVectorEnv):
    r"""Habitat VectorEnv adapted to include a wait_mask for each environment."""

    @staticmethod
    @profiling_wrapper.RangeContext("_worker_env")
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(env, (habitat.RLEnv, gym.Env)):
                        # habitat.RLEnv
                        action, wait_mask = data
                        if wait_mask:
                            # Create dummy observations
                            obs_space = env.observation_space
                            observations = {}
                            for sensor in obs_space.spaces:
                                observations[sensor] = np.zeros(
                                    obs_space.spaces[sensor].shape,
                                    dtype=obs_space.spaces[sensor].dtype,
                                )
                            reward, done, info = 0, False, {}
                        else:
                            observations, reward, done, info = env.step(**action)
                            if auto_reset_done and done:
                                observations = env.reset()
                        with profiling_wrapper.RangeContext("worker write after step"):
                            connection_write_fn((observations, reward, done, info))
                    elif isinstance(env, habitat.Env):  # type: ignore
                        # habitat.Env
                        action, wait_mask = data
                        if wait_mask:
                            # Create dummy observations
                            obs_space = env.observation_space
                            observations = {}
                            for sensor in obs_space.spaces:
                                observations[sensor] = np.zeros(
                                    obs_space.spaces[sensor].shape,
                                    dtype=obs_space.spaces[sensor].dtype,
                                )
                        else:
                            observations = env.step(**action)
                            if auto_reset_done and env.episode_over:
                                observations = env.reset()
                        connection_write_fn(observations)
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None:
                        function_args = {}

                    result_or_fn = getattr(env, function_name)

                    if len(function_args) > 0 or callable(result_or_fn):
                        result = result_or_fn(**function_args)
                    else:
                        result = result_or_fn

                    connection_write_fn(result)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                with profiling_wrapper.RangeContext("worker wait for command"):
                    command, data = connection_read_fn()

        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            if child_pipe is not None:
                child_pipe.close()
            env.close()

    def async_step_at(
        self,
        index_env: int,
        action: Union[int, str, Dict[str, Any]],
        wait_mask: Optional[bool] = None,
    ) -> None:
        # Backward compatibility
        if isinstance(action, (int, np.integer, str)):
            action = {"action": {"action": action}}

        self._warn_cuda_tensors(action)
        self._connection_write_fns[index_env]((STEP_COMMAND, (action, wait_mask)))

    def async_step(
        self,
        data: Sequence[Union[int, str, Dict[str, Any]]],
        wait_mask: Optional[Sequence[bool]] = None,
    ) -> None:
        r"""Asynchronously step in the environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """

        if wait_mask is None:
            wait_mask = [False for _ in data]
        for index_env, act in enumerate(data):
            self.async_step_at(index_env, act, wait_mask=wait_mask[index_env])

    def step(
        self,
        data: Sequence[Union[int, str, Dict[str, Any]]],
        wait_mask: Optional[Sequence[bool]] = None,
    ) -> List[Any]:
        r"""Perform actions in the vectorized environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(data, wait_mask=wait_mask)
        return self.wait_step()
