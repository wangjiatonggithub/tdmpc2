from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		video_eval_freq = int(self.cfg.get('video_eval_freq', 0))
		video_enabled = self.cfg.save_video and (video_eval_freq <= 0 or self._step % video_eval_freq == 0)
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0 and video_enabled))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video and video_enabled:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None): # 将当前时刻数据存储到轨迹中
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self): # 收集数据，模型评估，网络训练
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically 每隔一定步数进行一次模型评估
			# print(self._step)
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
			if self._step==0:
				eval_next = False
			# Reset environment
			if done:
				# print("进入done")
				if eval_next:
					# print("进入eval")
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0: # 记录上一条轨迹信息
					# print("记录上一条轨迹信息")
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update( # train_metrics为字典，将新数据添加到字典中
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics()) 
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds)) # 存储整条轨迹的各种信息，包括观测、动作、奖励等
					# print("创建buffer")

				# _t0 = time()
				# print("进入环境重置")
				obs = self.env.reset()
				# print(f"[debug] env.reset took {time() - _t0:.3f}s")
				self._tds = [self.to_td(obs)]

			# Collect experience 采集一定步数数据后开始使用策略网络生成动作而不是随机生成
			# print(self.cfg.seed_steps)
			if self._step > self.cfg.seed_steps:
				_t0 = time()
				action = self.agent.act(obs, t0=len(self._tds)==1)
				if self._step < 3:
					print(f"[debug] agent.act took {time() - _t0:.3f}s")
			else:
				action = self.env.rand_act()
			# _t0 = time()
			obs, reward, done, info = self.env.step(action)
			# if self._step < 3:
			# 	print(f"[debug] env.step took {time() - _t0:.3f}s")
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent 更新网络
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
