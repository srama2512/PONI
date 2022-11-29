import torch

from .habitat import construct_envs


def make_vec_envs(args, workers_ignore_signals: bool = False, **kwargs):
    envs = construct_envs(args, workers_ignore_signals=workers_ignore_signals, **kwargs)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch:
    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def plan_act_and_preprocess(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_reachability_map(self, inputs):
        reachability_maps, fmm_dists = self.venv.get_reachability_map(inputs)
        reachability_maps = torch.from_numpy(reachability_maps).float().to(self.device)
        fmm_dists = torch.from_numpy(fmm_dists).float().to(self.device)
        return reachability_maps, fmm_dists

    def get_frontier_map(self, inputs):
        frontier_maps = self.venv.get_frontier_map(inputs)
        frontier_maps = torch.from_numpy(frontier_maps).to(self.device)
        return frontier_maps

    def get_fmm_dists(self, inputs):
        fmm_dists = self.venv.get_fmm_dists(inputs)
        fmm_dists = torch.from_numpy(fmm_dists).to(self.device)
        return fmm_dists

    def current_episodes(self):
        curr_eps = self.venv.current_episodes()
        return curr_eps

    def get_current_episodes(self):
        curr_eps = self.venv.get_current_episodes()
        return curr_eps

    def close(self):
        return self.venv.close()
