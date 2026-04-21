from copy import deepcopy # 深度拷贝，主要用于复制Q网络以创建目标网络

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1) # 任务嵌入，最大范数为1
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim)) # 动作掩码，register_buffer是nn.Module中的一个方法
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		if cfg.obs != 'state':
			raise NotImplementedError("Encoder removed: only state observations are supported.")
		self.state_dim = cfg.obs_shape[cfg.obs][0]
		self._dynamics = layers.mlp(self.state_dim - 3 + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], self.state_dim - 3, act=layers.SimNorm(cfg)) # act参数传入激活函数，2*[cfg.mlp_dim]代表两层每层维度为cfg.mlp_dim的隐藏层
		# self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		reward_dim = 1 if cfg.get('continuous_reward', False) else max(cfg.num_bins, 1)
		self._reward = layers.mlp(self.state_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], reward_dim)
		self._pi = layers.mlp(self.state_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		# self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self._Qs = layers.Ensemble([layers.mlp(self.state_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], reward_dim, dropout=cfg.dropout) for _ in range(cfg.num_q)]) # 使用集成方法同时创建num_q个Q网络
		self.apply(init.weight_init)
		if not cfg.get('continuous_reward', False):
			init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]]) # 将奖励函数和Q函数的最后一层权重初始化为0，以稳定训练初期的训练

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self): # 创建目标Q网络
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True) # 分离Q网络，用于更新策略时是用，提供Q值但不参与网络更新
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True) # 目标Q网络，用于软更新

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Dynamics', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._dynamics, self._reward, self._pi, self._Qs]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self, step=None):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		if step is None:
			tau = self.cfg.tau
		else:
			start_step = 100_000
			end_step = 200_000
			min_tau = 0.0005
			if step <= start_step:
				tau = self.cfg.tau
			elif step >= end_step:
				tau = min_tau
			else:
				frac = (step - start_step) / (end_step - start_step)
				tau = self.cfg.tau + (min_tau - self.cfg.tau) * frac
		self._target_Qs_params.lerp_(self._detach_Qs_params, tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1) # 将任务嵌入变量拼接到状态后面

	def encode(self, obs, task):
		"""
		Identity mapping for state observations (encoder removed).
		"""
		return obs

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z_in = self.task_emb(z, task)
		else:
			z_in = z
		
		# Exclude goal from dynamics model
		z_in_no_goal = torch.cat([z_in[..., :7], z_in[..., 10:]], dim=-1)
		z_in_no_goal = torch.cat([z_in_no_goal, a], dim=-1)
		next_z_no_goal = self._dynamics(z_in_no_goal)

		# The goal is static
		goal_pos = z[..., 7:10]
		next_z = torch.cat([next_z_no_goal[..., :7], goal_pos, next_z_no_goal[..., 7:]], dim=-1)
		return next_z

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task):
		"""
		Deterministic termination based on distance to goal (0 or 1).
		Assumes observation layout: [joint_pos(7), goal_pos(3), ee_pos(3), ...].
		"""
		assert task is None
		ee_pos = z[..., 10:13]
		goal_pos = z[..., 7:10]
		dist = torch.linalg.norm(ee_pos - goal_pos, dim=-1, keepdim=True)
		return (dist < self.cfg.goal_arrival_threshold).float()
		

	def pi(self, z, task): # 策略网络也要限幅，范围[-1，1]
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1) # 将输出平均切分成两部分
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif) # 限制标准差范围
		eps = torch.randn_like(mean) # 从标准正态分布中随机采样噪声

		if self.cfg.multitask: # Mask out unused action dimensions 添加动作掩码
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std) # 计算重参数化采样后，tanh变换前的高斯分布动作的对数概率

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size # 熵在原来基础上又乘了一遍动作维数，使维数越大越鼓励探索

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob) # 动作进行tanh变换，并修正概率分布

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob, # 熵的蒙特卡洛采样值
			"scaled_entropy": -log_prob * entropy_scale, # 进行概率缩放前的动作熵在原来的基础上又乘了一个动作维数
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = out[qidx]
		if not self.cfg.get('continuous_reward', False):
			Q = math.two_hot_inv(Q, self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
