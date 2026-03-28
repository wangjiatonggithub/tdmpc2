import numpy as np
import mujoco
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:
    raise ImportError(
        "stable-baselines3>=2.x requires gymnasium. Please install gymnasium in the current env."
    ) from exc
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
# import setproctitle
from typing import Optional
from scipy.spatial.transform import Rotation as R

# setproctitle.setproctitle("python train_myself.py") # 服务器伪装
# 忽略stable_baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"): # 创建标志文件，指示当前环境是否可视化
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"): # 检查标志文件是否存在，如果存在则说明已有环境在可视化，当前环境应禁用可视化
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"): # 删除标志文件，允许其他环境可视化
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class PandaObstacleEnv(gym.Env):
    def __init__(
        self,
        visualize: bool = False,
        obstacle_type: str = "sphere",
        obstacle_randomize_pos: bool = True,
        randomize_init_qpos: bool = False,
        enforce_collision_free_init: bool = True, # 机械臂初始位姿碰撞检测
        init_qpos_max_attempts: int = 50, # 机械臂位姿初始化尝试次数上限
        init_min_distance: float = 0.08, # 机械臂初始位姿与障碍物的最小距离要求
        goal_min_distance: float = 0.08, # 目标位置与障碍物的最小距离要求
        randomize_goal_pos: bool = True,
        # use_sim_time_for_timeout: bool = False,
        max_episode_time: float = 20, # 20.0
        pause_on_collision: bool = False, # 测试时碰到障碍物就停止，以便清楚展示碰撞情况
        initial_pause_s: float = 0, # 每次reset后初始位姿停留时长（秒）
        # collision_pause_time: float = 0.8,
        # collision_pause_mode: str = "sleep",
    ):
        super(PandaObstacleEnv, self).__init__()
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None # 初始可视化窗口句柄

        self.model = mujoco.MjModel.from_xml_path('/home/wangjiatong/tdmpc2/tdmpc2/envs/tasks/scene_pos_with_obstacles.xml') # 加载环境模型
        self.data = mujoco.MjData(self.model) # 仿真数据
        # for i in range(self.model.ngeom):
        #     if self.model.geom_group[i] == 3:
        #         self.model.geom_conaffinity[i] = 0
        
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0 # 相机距离
            self.handle.cam.azimuth = 0.0 # 相机方位角
            self.handle.cam.elevation = -30.0 # 相机仰角
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4]) # 相机注视点
        
        self.np_random = np.random.default_rng(None)
        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body') # 获取末端执行器的body_id
        self.home_joint_pos = np.array(self.model.key_qpos[0][:7], dtype=np.float32) # 获取初始关节位姿
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # 归一化动作空间
        
        # self.goal_position_base = np.array([0.4, -0.3, 0.4], dtype=np.float32)
        self.goal_position_base = np.array([0.0, 0.6, 0.5], dtype=np.float32) # 目标位置基准，y轴会随机变化
        self.goal_position = self.goal_position_base.copy()
        self.goal_arrival_threshold = 0.005 # 目标到达阈值，单位为米
        self.goal_visu_size = 0.02 # 目标可视化球尺寸
        self.goal_visu_rgba = [0.1, 0.3, 0.3, 0.8] # 目标可视化球颜色和透明度
        self.contact_visu_size = 0.012 # 碰撞点可视化球尺寸
        self.contact_visu_rgba = [1.0, 0.2, 0.2, 0.9] # 碰撞点颜色
        self.last_contact_pos = None # 记录上一次碰撞位置，用于可视化显示碰撞点
        self.last_contact_info = None # 记录碰撞信息

        self.proximity_threshold = 0.05 # 近距离惩罚阈值（单位：米）
        self.proximity_penalty_scale = 20.0 # 近距离惩罚强度

        self.randomize_init_qpos = randomize_init_qpos
        self.enforce_collision_free_init = enforce_collision_free_init
        self.init_qpos_max_attempts = init_qpos_max_attempts
        self.init_min_distance = init_min_distance
        self.goal_min_distance = goal_min_distance
        self.randomize_goal_pos = randomize_goal_pos
        # self.use_sim_time_for_timeout = use_sim_time_for_timeout
        self.max_episode_time = max_episode_time
        self.pause_on_collision = pause_on_collision
        self.initial_pause_s = max(0.0, float(initial_pause_s))
        # self.collision_pause_time = collision_pause_time
        # self.collision_pause_mode = collision_pause_mode

        # 可选障碍物类型: sphere | box | cylinder
        self.obstacle_type = obstacle_type
        self.obstacle_randomize_pos = obstacle_randomize_pos
        self.obstacle_geom_names = {
            "sphere": ["obstacle_sphere"],
            "box": ["obstacle_u_left", "obstacle_u_right", "obstacle_u_base"],
            "cylinder": ["obstacle_cylinder"],
        }
        if self.obstacle_type not in self.obstacle_geom_names:
            raise ValueError(f"Unsupported obstacle_type: {self.obstacle_type}")

        self.obstacle_geom_ids = {} # 记录障碍物名字与对应的索引，为字典变量
        self.obstacle_geom_rgba = {} # 记录障碍物名字与对应的颜色
        all_obstacle_names = set(sum(self.obstacle_geom_names.values(), []))
        for i in range(self.model.ngeom): # model.ngeom返回模型中几何体的数量
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) # 提取出模型中全部障碍物名字，如obstacle_sphere，obstacle_u_left等
            if name in all_obstacle_names:
                self.obstacle_geom_ids[name] = i
                self.obstacle_geom_rgba[name] = self.model.geom_rgba[i].copy()

        active_names = self.obstacle_geom_names[self.obstacle_type]
        missing = [n for n in active_names if n not in self.obstacle_geom_ids]
        if missing:
            raise ValueError(f"Obstacle geom(s) not found in model: {missing}")
        self.obstacle_ids = [self.obstacle_geom_ids[n] for n in active_names] # 提取本项目中真正考虑的障碍物id
        self.obstacle_id_1 = self.obstacle_ids[0]
        self._set_active_obstacle(active_names)

        self.obstacle_group_center = self._get_obstacle_center() # 障碍物整体中心位置（用于随机移动）
        self.obstacle_positions = self._get_obstacle_centers() # 每个障碍物几何体中心位置
        self.obstacle_sizes = self._get_obstacle_sizes_obs() # 每个障碍物几何体尺寸
        # print("self.obstacle_group_center ", self.obstacle_group_center )
        # print("self.obstacle_positions ", self.obstacle_positions )
        # print("self.obstacle_sizes ", self.obstacle_sizes )

        
        num_obstacles = self.obstacle_positions.shape[0]
        size_dim = self.obstacle_sizes.shape[1]
        # 需要改，输入观测改成相对值
        self.obs_size = 7 + 3 + (3 * num_obstacles) + (size_dim * num_obstacles) # 7轴关节角度+目标位置+每个障碍物位置+尺寸
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

        self.last_action = self.home_joint_pos
        self.start_sim_time = 0.0
        self.handle = None # 初始可视化窗口句柄
        self.renderer = None # 用于录制视频的渲染器

    def _render_scene(self) -> None: # 在可视化窗口中渲染目标位置和碰撞发生位置
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0 # 把场景中的几何体数量清零
        total_geoms = 1 + (1 if self.last_contact_pos is not None else 0) # 目标+碰撞点数量
        self.handle.user_scn.ngeom = total_geoms

        mujoco.mjv_initGeom( # 绘制目标点，形状为球形
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_visu_size, 0.0, 0.0],
            pos=self.goal_position,
            mat=np.eye(3).flatten(), # 目标点的旋转姿态
            rgba=np.array(self.goal_visu_rgba, dtype=np.float32)
        )

        if self.last_contact_pos is not None: # 绘制碰撞位置，形状为球形
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[1],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.contact_visu_size, 0.0, 0.0],
                pos=self.last_contact_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array(self.contact_visu_rgba, dtype=np.float32)
            )

    def render(self, mode='rgb_array', width=384, height=384, camera_id=None, camera_name=None):
        """
        实现渲染功能，返回一个 RGB 图像数组。
        """
        if mode == 'rgb_array':
            if self.renderer is None:
                # 只有在第一次调用 render 时才初始化渲染器，节省资源
                self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            
            # 更新渲染器中的物理状态
            camera = camera_name if camera_name is not None else camera_id
            if camera is None:
                camera = -1
            self.renderer.update_scene(self.data, camera=camera)
            # 返回 RGB 数组
            return self.renderer.render()

        elif mode == 'human':
            # 如果已经有 passive viewer (self.handle)，则不需要额外操作
            if self.handle is not None:
                self.handle.sync()
            return None

    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]: # 重置环境状态，包括机械臂初始位姿、目标位置和障碍物位置，并返回初始观察
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed) # 初始化随机数生成器
        
        
        mujoco.mj_resetData(self.model, self.data) # 将初始位姿重置为默认值
        if self.obstacle_randomize_pos: # 障碍物位置随机初始化
            group_center = self._get_obstacle_center()
            new_center = np.array([
                self.np_random.uniform(-0.3, 0.3),
                # self.np_random.uniform(-0.2, 0.2),
                # self.np_random.uniform(0.4, 0.55)
                # group_center[0],
                group_center[1],
                group_center[2]
            ], dtype=np.float32)
            delta = new_center - group_center
            for geom_id in self.obstacle_ids:
                self.model.geom_pos[geom_id] = self.model.geom_pos[geom_id] + delta
            self.obstacle_group_center = new_center
        else:
            self.obstacle_group_center = self._get_obstacle_center()

        # 根据障碍物位置选择初始关节角，确保不与障碍物碰撞
        joint_ranges = self.model.jnt_range[:7] # 获取关节角度范围
        def _set_qpos_and_forward(qpos: np.ndarray) -> None:
            self.data.qpos[:7] = qpos.astype(np.float32)
            self.data.qpos[7:] = [0.04, 0.04] # 设置手指关节固定
            mujoco.mj_forward(self.model, self.data) # 执行一步更新但不推进仿真时间，不做碰撞检测和动力学积分，用于在初始化关节位姿时调用

        if self.enforce_collision_free_init:
            max_attempts = max(1, int(self.init_qpos_max_attempts))
            found = False
            for attempt in range(max_attempts):
                if self.randomize_init_qpos or attempt > 0:
                    candidate = self.np_random.uniform(joint_ranges[:, 0], joint_ranges[:, 1]).astype(np.float32)
                else:
                    candidate = self.home_joint_pos
                _set_qpos_and_forward(candidate)
                now_ee_pos = self.data.body(self.end_effector_id).xpos.copy() # 获取末端执行器的位置
                if self.data.ncon == 0 and self._min_distance_to_obstacles(now_ee_pos) >= self.init_min_distance:
                    found = True
                    # print ("yes")
                    break
            if not found:
                warnings.warn("No collision-free init pose found; using home pose.")
                _set_qpos_and_forward(self.home_joint_pos)
        else:
            if self.randomize_init_qpos:
                candidate = self.np_random.uniform(joint_ranges[:, 0], joint_ranges[:, 1]).astype(np.float32)
            else:
                candidate = self.home_joint_pos
            _set_qpos_and_forward(candidate)

        mujoco.mj_step(self.model, self.data) # 执行一步更新并推进仿真时间

        self.obstacle_positions = self._get_obstacle_centers()
        self.obstacle_sizes = self._get_obstacle_sizes_obs()

        self.goal_position = self._sample_goal_position(randomize=self.randomize_goal_pos)
        
        if self.visualize:
            self._render_scene()
            if self.initial_pause_s > 0:
                time.sleep(self.initial_pause_s)
        
        self.last_action = self.data.qpos[:7].copy()
        obs = self._get_observation()
        self.start_t = time.time()
        self.start_sim_time = float(self.data.time)
        self.last_contact_pos = None
        self.last_contact_info = None
        return obs, {}

    def _sample_goal_position(self, randomize: bool = True) -> np.ndarray:
        """Sample a goal that is outside obstacles and at least goal_min_distance away.
        If `randomize` is False, prefer the base goal but fallback to random valid goal when needed.
        """
        max_attempts = 200

        def _is_valid(pt: np.ndarray) -> bool:
            return (not self._is_point_in_any_obstacle(pt)) and (
                self._min_distance_to_obstacles(pt) >= self.goal_min_distance
            )

        if not randomize:
            base = self.goal_position_base.copy()
            if _is_valid(base):
                return base

        best = None
        best_dist = -np.inf
        for _ in range(max_attempts):
            # 原始三维随机采样（保留注释，便于回退或对比）
            candidate = np.array([
                self.np_random.uniform(-0.5, 0.5),
                self.np_random.uniform(-0.7, 0.7),
                self.np_random.uniform(0.2, 0.8),
            ], dtype=np.float32)
            # candidate = self.goal_position_base.copy()
            # candidate[1] = self.np_random.uniform(-0.3, 0.3)
            dist = self._min_distance_to_obstacles(candidate)
            if dist > best_dist:
                best_dist = dist
                best = candidate
            if _is_valid(candidate):
                return candidate

        warnings.warn("Goal position unsafe after retries; using farthest candidate.")
        return best if best is not None else self.goal_position_base.copy()

    def _get_observation(self) -> np.ndarray: 
        # 需要改，变成相对值
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        self.obstacle_positions = self._get_obstacle_centers()
        self.obstacle_sizes = self._get_obstacle_sizes_obs()
        obstacle_pos_noise = np.random.normal(0, 0.001, size=self.obstacle_positions.shape) # 障碍物位置再加一个随机扰动
        return np.concatenate([
            joint_pos,
            self.goal_position,
            (self.obstacle_positions + obstacle_pos_noise).flatten(),
            self.obstacle_sizes.flatten()
        ])

    def _get_obstacle_centers(self) -> np.ndarray: # 从配置文件中获取障碍物中心
        centers = []
        for geom_id in self.obstacle_ids:
            centers.append(self.model.geom_pos[geom_id].astype(np.float32))
        return np.stack(centers, axis=0)

    def _get_obstacle_sizes_obs(self) -> np.ndarray: # 从配置文件中获取障碍物尺寸
        sizes = []
        if self.obstacle_type == "sphere":
            for geom_id in self.obstacle_ids:
                sizes.append([self.model.geom_size[geom_id][0]])
        elif self.obstacle_type == "box":
            for geom_id in self.obstacle_ids:
                sizes.append(self.model.geom_size[geom_id].astype(np.float32).tolist())
        elif self.obstacle_type == "cylinder":
            for geom_id in self.obstacle_ids:
                sizes.append(self.model.geom_size[geom_id][:2].astype(np.float32).tolist())
        else:
            for geom_id in self.obstacle_ids:
                sizes.append([self.model.geom_size[geom_id][0]])
        return np.array(sizes, dtype=np.float32)

    def _is_point_in_any_obstacle(self, point: np.ndarray) -> bool: # 判断随机初始化的目标点是否在障碍物内部
        centers = self._get_obstacle_centers()
        sizes = self._get_obstacle_sizes_obs()
        for idx in range(centers.shape[0]):
            center = centers[idx]
            if self.obstacle_type == "sphere":
                radius = sizes[idx][0]
                if np.linalg.norm(point - center) <= radius:
                    return True
            elif self.obstacle_type == "box":
                half_extents = sizes[idx]
                if np.all(np.abs(point - center) <= half_extents):
                    return True
            elif self.obstacle_type == "cylinder":
                radius = sizes[idx][0]
                half_height = sizes[idx][1]
                delta = point - center
                if (delta[0] ** 2 + delta[1] ** 2) <= radius ** 2 and np.abs(delta[2]) <= half_height:
                    return True
            else:
                radius = sizes[idx][0]
                if np.linalg.norm(point - center) <= radius:
                    return True
        return False

    def _set_active_obstacle(self, active_names: list[str]) -> None: # 只保留指定的环境障碍物进行可视化和考虑碰撞
        active_set = set(active_names) # 把列表转换为集合
        for name, geom_id in self.obstacle_geom_ids.items():
            if name in active_set:
                self.model.geom_conaffinity[geom_id] = 1 # 允许接触检测
                self.model.geom_contype[geom_id] = 1 # 允许碰撞类型匹配
                self.model.geom_rgba[geom_id] = self.obstacle_geom_rgba[name] # 颜色正常显示
            else:
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_contype[geom_id] = 0
                rgba = self.model.geom_rgba[geom_id].copy()
                rgba[3] = 0.0 # 透明度变成0
                self.model.geom_rgba[geom_id] = rgba

    def _get_obstacle_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        min_xyz = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        for geom_id in self.obstacle_ids:
            pos = self.model.geom_pos[geom_id].astype(np.float32)
            size = self.model.geom_size[geom_id].astype(np.float32)
            min_xyz = np.minimum(min_xyz, pos - size)
            max_xyz = np.maximum(max_xyz, pos + size)
        return min_xyz, max_xyz

    def _get_obstacle_center(self) -> np.ndarray:
        if self.obstacle_type == "box":
            min_xyz, max_xyz = self._get_obstacle_aabb()
            return ((min_xyz + max_xyz) / 2.0).astype(np.float32)
        return np.array(self.model.geom_pos[self.obstacle_id_1], dtype=np.float32)

    def _min_distance_to_obstacles(self, point: np.ndarray) -> float: # 检测机械臂距离障碍物的最小距离，用于近距离惩罚
        centers = self._get_obstacle_centers()
        sizes = self._get_obstacle_sizes_obs()
        min_dist = np.inf
        for idx in range(centers.shape[0]):
            center = centers[idx]
            if self.obstacle_type == "sphere":
                radius = sizes[idx][0]
                dist = max(0.0, np.linalg.norm(point - center) - radius)
            elif self.obstacle_type == "box":
                half_extents = sizes[idx]
                delta = np.maximum(np.abs(point - center) - half_extents, 0.0)
                dist = np.linalg.norm(delta)
            elif self.obstacle_type == "cylinder":
                radius = sizes[idx][0]
                half_height = sizes[idx][1]
                delta = point - center
                radial = max(0.0, np.sqrt(delta[0] ** 2 + delta[1] ** 2) - radius)
                vertical = max(0.0, np.abs(delta[2]) - half_height)
                dist = np.sqrt(radial ** 2 + vertical ** 2)
            else:
                radius = sizes[idx][0]
                dist = max(0.0, np.linalg.norm(point - center) - radius)
            min_dist = min(min_dist, dist)
        return float(min_dist)

    def _calc_reward(self, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        # 奖励函数改
        now_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        dist_to_goal = np.linalg.norm(now_ee_pos - self.goal_position)

        # # 非线性距离奖励
        # if dist_to_goal < self.goal_arrival_threshold:
        #     distance_reward = 100.0*(1.0+(1.0-(dist_to_goal / self.goal_arrival_threshold)))
        # elif dist_to_goal < 2*self.goal_arrival_threshold:
        #     distance_reward = 50.0*(1.0+(1.0-(dist_to_goal / (2*self.goal_arrival_threshold))))
        # elif dist_to_goal < 3*self.goal_arrival_threshold:
        #     distance_reward = 10.0*(1.0+(1.0-(dist_to_goal / (3*self.goal_arrival_threshold))))
        # else:
        #     distance_reward = 1.0 / (1.0 + dist_to_goal)
        
        # 改成没达到目标之前，每走一步都是惩罚，可以防止机械臂为了获得高奖励在运动过程中刻意拖拉不早点结束
        # 修改奖励函数，使得离目标点越近惩罚梯度越大，有利于提升精度
        T = self.goal_arrival_threshold
        try:
            if dist_to_goal < T:
                # distance_penalty = 10.0 * (dist_to_goal / T)
                distance_penalty = 50.0 * (dist_to_goal / T)
            elif dist_to_goal < 2 * T:
                # distance_penalty = 10.0 + 20.0 * ((dist_to_goal - T) / T)
                distance_penalty = 50.0 + 10.0 * ((dist_to_goal - T) / T)
            elif dist_to_goal < 3 * T:
                # distance_penalty = 30.0 + 30.0 * ((dist_to_goal - 2 * T) / T)
                distance_penalty = 60.0 + 5.0 * ((dist_to_goal - 2 * T) / T)
            else:
                # 增加 epsilon 防止对数运算异常
                # distance_penalty = 60.0 + 10.0 * np.log(1.0 + max(0, dist_to_goal - 3 * T))
                distance_penalty = 65.0 + 10.0 * np.log(1.0 + (dist_to_goal - 3 * T))
        except Exception as e:
            # 如果出错，给一个默认的大惩罚，而不是让整个程序崩溃
            print(f"Reward calculation error: {e}")
            distance_penalty = 100.0
        distance_penalty = 0.01*distance_penalty

        # # 连续平滑的距离奖励
        # # 无论多远都有梯度！距离为 0 时满分，距离越大平滑下降趋近于 0
        # distance_reward = 5.0 * np.exp(-5.0 * dist_to_goal)
        
        # 到达目标的终极重奖 (Sparse Bonus)
        is_success = dist_to_goal < self.goal_arrival_threshold
        success_bonus = 100.0 if is_success else 0.0

        # 平滑惩罚(当接近目标点时去掉平滑惩罚来提升精度)
        if dist_to_goal < 3 * self.goal_arrival_threshold:
            smooth_penalty = 0.0
        else:
            smooth_penalty = 0.001 * np.linalg.norm(action - self.last_action)
        # smooth_penalty = 0.001 * np.linalg.norm(action - self.last_action)

        # 碰撞惩罚
        contact_reward = 1000.0 * self.data.ncon # 利用接触对数量检测碰撞 *100

        # 近距离惩罚（未接触但过近）
        min_dist = self._min_distance_to_obstacles(now_ee_pos)
        proximity_penalty = 0.0
        if min_dist < self.proximity_threshold:
            # proximity_penalty = self.proximity_penalty_scale * (self.proximity_threshold - min_dist)
            proximity_penalty = self.proximity_penalty_scale * np.exp(-10.0 * min_dist) # 改成指数形式
        
        # 关节角度限制惩罚
        joint_penalty = 0.0
        for i in range(7):
            min_angle, max_angle = self.model.jnt_range[:7][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        # time_penalty = 0.001 * (float(self.data.time) - self.start_sim_time)
        time_penalty = 0.05

        total_reward = (- distance_penalty
                    + success_bonus
                    - contact_reward 
                    - proximity_penalty
                    - smooth_penalty 
                    - joint_penalty
                    - time_penalty)
        
        self.last_action = action.copy()
        
        return total_reward, dist_to_goal, self.data.ncon > 0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7): # 把动作从[-1,1]线性映射到关节真正的物理范围
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        self.data.ctrl[:7] = scaled_action
        self.data.qpos[7:] = [0.04,0.04]
        mujoco.mj_step(self.model, self.data)
        
        reward, dist_to_goal, collision = self._calc_reward(self.data.qpos[:7], action)
        terminated = False

        if collision:
            contact = self.data.contact[0] # 获取第一个接触信息
            body1_id = self.model.geom_bodyid[contact.geom1] # 记录发生碰撞的geom，body名；geom指附着在某个body上的几何体，body指机械臂关节，连杆或障碍物本身
            body2_id = self.model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            self.last_contact_pos = contact.pos.copy() # 记录碰撞位置
            self.last_contact_info = {
                "geom1": geom1_name,
                "geom2": geom2_name,
                "body1": body1_name,
                "body2": body2_name,
            }
            # print(
            #     f"[碰撞] geom: {geom1_name} vs {geom2_name} | "
            #     f"body: {body1_name} vs {body2_name} | pos: {self.last_contact_pos}"
            # )
            if self.visualize and self.pause_on_collision:
                self._render_scene() 
                self.handle.sync() # 将数据物理层面上的更新反映到可视化窗口中
                # input("[碰撞] 按回车继续...")


                # if self.collision_pause_mode == "manual":
                #     input("[碰撞] 按回车继续...")
                # elif self.collision_pause_mode == "sleep":
                #     if self.collision_pause_time > 0:
                #         time.sleep(self.collision_pause_time)
            reward -= 10.0
            terminated = True

        if dist_to_goal < self.goal_arrival_threshold:
            terminated = True
            print(f"[成功] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")
        # else:
        #     print(f"[失败] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        if not terminated:
            elapsed = float(self.data.time) - self.start_sim_time
            if elapsed > self.max_episode_time:
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                print(f"[失败] 距离目标: {dist_to_goal:.3f}")
                terminated = True

        if self.visualize and self.handle is not None:
            self._render_scene()
            self.handle.sync()
            
        
        obs = self._get_observation()
        info = {
            'is_success': not collision and terminated and (dist_to_goal < self.goal_arrival_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision,
            'collision_pos': self.last_contact_pos,
            'collision_info': self.last_contact_info,
        }
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None

def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 80_000_000,
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None,
    obstacle_type: str = "sphere",
    obstacle_randomize_pos: bool = True,
    randomize_init_qpos: bool = False,
    randomize_goal_pos: bool = True
) -> None:

    ENV_KWARGS = {
        'visualize': visualize,
        'obstacle_type': obstacle_type,
        'obstacle_randomize_pos': obstacle_randomize_pos,
        'randomize_init_qpos': randomize_init_qpos,
        'randomize_goal_pos': randomize_goal_pos
    }
    
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(** ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"} # 把字典作为参数传递给SubprocVecEnv构造函数，指定多进程启动方式为"fork"，在windows系统中必须用"spawn"
    )
    
    if resume_from is not None:
        model = PPO.load(resume_from, env=env)
    else:
        # POLICY_KWARGS = dict(
        #     activation_fn=nn.ReLU,
        #     net_arch=[dict(pi=[256, 128], vf=[256, 128])]
        # )
        
        
        POLICY_KWARGS = dict( # 神经网络结构
            activation_fn=nn.LeakyReLU,
            net_arch=[
                dict(
                    pi=[512, 256, 128],
                    vf=[512, 256, 128]
                )
            ]
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1, # 打印训练日志
            n_steps=2048, # 每个环境采样2048步再更新一次策略      
            batch_size=2048, # 每次采样的数据根据batch_size划分为若干minibatch，每次梯度下降利用batch_size个数据   
            n_epochs=10, # 利用一次采集到的n_steps个数据进行n_epochs次梯度更新，每个数据利用n_epochs次。n_epochs越大数据利用率越高但会过拟合使训练不稳定         
            gamma=0.99,
            # ent_coef=0.02,  # 增加熵系数，保留后期探索以提升泛化性
            ent_coef = 0.001, 
            clip_range=0.15,  # 限制策略更新幅度
            max_grad_norm=0.5,  # 梯度裁剪防止爆炸
            learning_rate=lambda f: 1e-4 * (1 - f),  # 学习率线性衰减（初始1e-4，后期逐步降低）
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./logs/tensorboard/panda_obstacle_avoidance/"
        )
    
    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True # 显示进度条
    )
    
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_obstacle_avoidance",
    total_episodes: int = 5,
    obstacle_type: str = "sphere",
    obstacle_randomize_pos: bool = True,
    randomize_init_qpos: bool = False,
    randomize_goal_pos: bool = True,
    # use_sim_time_for_timeout: bool = True,
    max_episode_time: float = 20.0,
    pause_on_collision: bool = True,
    # collision_pause_time: float = 0.8,
    # collision_pause_mode: str = "manual",
) -> None:
    env = PandaObstacleEnv(
        visualize=True,
        obstacle_type=obstacle_type,
        obstacle_randomize_pos=obstacle_randomize_pos,
        randomize_init_qpos=randomize_init_qpos,
        randomize_goal_pos=randomize_goal_pos,
        # use_sim_time_for_timeout=use_sim_time_for_timeout,
        max_episode_time=max_episode_time,
        pause_on_collision=pause_on_collision,
        # collision_pause_time=collision_pause_time,
        # collision_pause_mode=collision_pause_mode,
    )
    model = PPO.load(model_path, env=env)

    
    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            obs = env._get_observation()
            # print(f"观察: {obs}")
            action, _states = model.predict(obs, deterministic=False)
            # action += np.random.normal(0, 0.002, size=7)  # 加入噪声
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"动作: {action}, 奖励: {reward}, 终止: {terminated}, 截断: {truncated}, 信息: {info}")
            episode_reward += reward
            done = terminated or truncated
        
        if info['is_success']:
            success_count += 1
        print(
            f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'} "
            # f"| 碰撞: {info['collision']} | 距离: {info['distance_to_goal']:.4f}"
        )
    
    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    TRAIN_MODE = True  # 设为True开启训练模式
    OBSTACLE_TYPE = "box"  # 可选: sphere | box | cylinder
    OBSTACLE_RANDOMIZE_POS = True  # 是否随机变化障碍物位置
    RANDOMIZE_INIT_QPOS = True  # 是否随机机械臂初始位姿
    RANDOMIZE_GOAL_POS = True  # 是否随机目标位姿
    if TRAIN_MODE:
        import os 
        os.system("rm -rf /home/dar/mujoco-bin/mujoco-learning/tensorboard*")
    delete_flag_file()
    MODEL_PATH = "assets/model/rl_obstacle_avoidance_checkpoint/panda_obstacle_avoidance_v6_local"
    RESUME_MODEL_PATH = "assets/model/rl_obstacle_avoidance_checkpoint/panda_obstacle_avoidance_v2"
    if TRAIN_MODE:
        train_ppo(
            n_envs=64,                
            total_timesteps=60_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,
            # resume_from=RESUME_MODEL_PATH
            resume_from=None,
            obstacle_type=OBSTACLE_TYPE,
            obstacle_randomize_pos=OBSTACLE_RANDOMIZE_POS,
            randomize_init_qpos=RANDOMIZE_INIT_QPOS,
            randomize_goal_pos=RANDOMIZE_GOAL_POS
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=10, # 100
            obstacle_type=OBSTACLE_TYPE,
            obstacle_randomize_pos=OBSTACLE_RANDOMIZE_POS,
            randomize_init_qpos=RANDOMIZE_INIT_QPOS,
            randomize_goal_pos=RANDOMIZE_GOAL_POS,
        )
    os.system("date")