# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA
from tqdm import tqdm

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-6)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        # # 默认情况下，主网络启用梯度计算
        # for param in self.critic.parameters():
        #     print(param.requires_grad)  # 输出: True
        self.critic_target = copy.deepcopy(self.critic)
        # # 目标网络通常禁用梯度计算
        # for param in self.critic_target.parameters():
        #     print(param.requires_grad)  # 输出: False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-5)

        if lr_decay:
            # self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            # self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

            # self.actor_lr_scheduler = CosineAnnealingLR(
            #     self.actor_optimizer, 
            #     T_max=500,    # 小周期测试
            #     eta_min=5e-7
            # )
            # self.critic_lr_scheduler = CosineAnnealingLR(
            #     self.critic_optimizer, 
            #     T_max=500,    # 小周期测试  
            #     eta_min=5e-6
            # )

            self.actor_lr_scheduler = CosineAnnealingWarmRestarts(
                self.actor_optimizer, T_0=500, T_mult=2, eta_min=1e-7
            )
            self.critic_lr_scheduler = CosineAnnealingWarmRestarts(
                self.critic_optimizer, T_0=500, T_mult=2, eta_min=1e-6
            )

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

        self.val_losses = []

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def validate(self, val_loader):
        """验证当前模型性能（简化版，主要关注BC损失）"""
        self.actor.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # 1. 用tqdm包裹val_loader，创建验证进度条
        val_progress = tqdm(val_loader, desc="Validating (BC Loss)", leave=False, ncols=100)
        
        with torch.no_grad():  # 禁用梯度计算，节省内存并提速
            for batch in val_progress:
                # 安全解包批次数据，避免格式异常
                try:
                    state, action, reward, next_state, not_done = batch
                except ValueError as e:
                    print(f"警告：批次数据格式错误 - {str(e)}，跳过该批次")
                    continue
                
                try:
                    state = state.to(self.device, non_blocking=True).unsqueeze(0)
                    action = action.to(self.device, non_blocking=True).unsqueeze(0)
                    reward = reward.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
                    next_state = next_state.to(self.device, non_blocking=True).unsqueeze(0)
                    not_done = not_done.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
                except Exception as e:
                    print(f"警告：批次数据预处理失败 - {str(e)}，跳过该批次")
                    continue
                
                # 计算行为克隆损失，校验维度匹配
                try:
                    pred_actions = self.actor(state)
                    # 确保预测动作与真实动作维度一致
                    if pred_actions.shape != action.shape:
                        raise ValueError(f"动作维度不匹配：预测={pred_actions.shape}，真实={action.shape}")
                    loss = nn.MSELoss()(pred_actions, action)
                except Exception as e:
                    print(f"警告：损失计算失败 - {str(e)}，跳过该批次")
                    continue
                
                # 累加损失和批次计数
                total_loss += loss.item()
                num_batches += 1
                
                # 实时更新进度条信息
                current_avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                val_progress.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{current_avg_loss:.4f}"
                })
        
        val_progress.close()  # 显式关闭进度条
        self.actor.train()    # 切回训练模式
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"\nValidation Finished | Avg BC Loss: {avg_loss:.4f} | Total Batches: {num_batches}")
        return avg_loss
    
    def evaluate(self, test_loader):
        """测试模型性能，计算多种损失指标"""
        self.actor.eval()
        
        all_obs = []
        all_actions = []
        all_pred_actions = []
        
        # 添加进度条和错误处理，类似于validate方法
        test_progress = tqdm(test_loader, desc="Evaluating", leave=False, ncols=100)
        
        with torch.no_grad():
            for batch in test_progress:
                # 安全解包批次数据，避免格式异常
                try:
                    # 从批次中提取观测和动作（适配DataLoader的数据结构）
                    observations, actions, _, _, _ = batch
                except ValueError as e:
                    print(f"警告：批次数据格式错误 - {str(e)}，跳过该批次")
                    continue
                
                try:
                    observations = observations.to(self.device, non_blocking=True).unsqueeze(0)
                    actions = actions.to(self.device, non_blocking=True).unsqueeze(0)
                except Exception as e:
                    print(f"警告：批次数据预处理失败 - {str(e)}，跳过该批次")
                    continue
                
                # 根据模型类型预测动作
                try:
                    pred_actions = self.actor(observations)
                except Exception as e:
                    print(f"警告：动作预测失败 - {str(e)}，跳过该批次")
                    continue
                
                # 保存结果用于后续计算
                all_obs.append(observations.cpu())
                all_actions.append(actions.cpu())
                all_pred_actions.append(pred_actions.cpu())
        
        test_progress.close()  # 显式关闭进度条
        
        # 合并所有批次结果
        all_obs = torch.cat(all_obs, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_pred_actions = torch.cat(all_pred_actions, dim=0)
        
        # 计算多种损失指标（与训练时的损失相对应）
        mse_loss = nn.MSELoss()(all_pred_actions, all_actions).item()
        mae_loss = nn.L1Loss()(all_pred_actions, all_actions).item()
        
        print(f"\n=== Test Performance ===")
        print(f"MSE Loss: {mse_loss:.6f}")
        print(f"MAE Loss: {mae_loss:.6f}")
        
        # 动作分量误差分析
        action_errors = (all_pred_actions - all_actions).abs().mean(dim=0).numpy()
        action_names = ['Throttle', 'Steer', 'Brake']
        for i, name in enumerate(action_names[:len(action_errors)]):
            print(f"{name} Error: {action_errors[i]:.4f}")
        
        self.actor.train()  # 切回训练模式
        
        # 返回计算的损失值，可用于后续绘图或记录
        return {
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'action_errors': action_errors
        }

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for i in tqdm(range(iterations), desc="训练进度"):
            # Sample replay buffer / batch
            state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

            # print(f"采样数据维度:")
            # print(f"  state: {state.shape} (批次大小: {state.shape[0]}, 状态特征数: {state.shape[1] if len(state.shape)>=2 else 1})")
            # print(f"  action: {action.shape} (批次大小: {action.shape[0]}, 动作维度: {action.shape[1] if len(action.shape)>=2 else 1})")
            # print(f"  next_state: {next_state.shape} (批次大小: {next_state.shape[0]}, 状态特征数: {next_state.shape[1] if len(next_state.shape)>=2 else 1})")
            # print(f"  reward: {reward.shape} (批次大小: {reward.shape[0]}, 维度: {len(reward.shape)})")
            # print(f"  not_done: {not_done.shape} (批次大小: {not_done.shape[0]}, 维度: {len(not_done.shape)})")

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            '这段代码实现了梯度裁剪,用于防止训练过程中的梯度爆炸问题'
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """            
            # 在训练循环中定期打印学习率
            if self.step % 5000 == 0:
                print("Actor optimizer param groups:", len(self.actor_optimizer.param_groups))
                print("Critic optimizer param groups:", len(self.critic_optimizer.param_groups))
                current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
                current_critic_lr = self.critic_optimizer.param_groups[0]['lr']
                print(f"Step {self.step}: Actor LR = {current_actor_lr:.2e}, Critic LR = {current_critic_lr:.2e}")

            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value, dim=0), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


