import numpy as np
import gymnasium as gym
from envs.wrappers.timeout import Timeout
from panda_obstacle_wall import PandaObstacleEnv  # 假设你的类名为 PandaEnv

class PandaTDMPC2Wrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        # 确保动作空间在 [-1, 1] 之间，这是 TD-MPC2 的默认假设
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=env.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32)

    def step(self, action):
        # 记得将 [-1, 1] 的动作映射回你环境真实的范围（如果不同的话）
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # TD-MPC2 需要 info 中包含 success（可选）
        info['success'] = info.get('success', False) 
        return obs.astype(np.float32), reward, done, info

def make_env(cfg):
    if cfg.task != 'panda-wall':
        raise ValueError('Unknown task:', cfg.task)
    env = PandaObstacleEnv(
        visualize = cfg.get('visualize', False),
        obstacle_type = cfg.get('obstacle_type', 'box'),
        obstacle_randomize_pos = cfg.get('obstacle_randomize_pos', True),
        randomize_init_qpos = cfg.get('randomize_init_qpos', True),
        randomize_goal_pos = cfg.get('randomize_goal_pos', True),
        max_episode_time = cfg.get('max_episode_time', 20.0),
        pause_on_collision = cfg.get('pause_on_collision', False),
        initial_pause_s = cfg.get('initial_pause_s', 0),
    ) # 实例化你自己的环境
    env = PandaTDMPC2Wrapper(env, cfg)
    # 必须包装 Timeout，TD-MPC2 依赖此包装器来处理最大步数
    env = Timeout(env, max_episode_steps=10000)
    
    # 从创建好的环境中获取观测和动作空间信息，并更新cfg
    cfg.obs_shape = env.observation_space.shape
    cfg.action_dim = env.action_space.shape[0]

    return env