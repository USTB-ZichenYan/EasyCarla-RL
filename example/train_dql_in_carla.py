import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import csv
import gym
import easycarla
from agents.ql_diffusion import Diffusion_QL
from datetime import datetime
import collections
import random
import warnings


# ===================== Helper Functions =====================
def convert_obs_dict_to_vector(obs_dict):
    """Convert observation dictionary to a flattened state vector."""
    # 补充维度注释，便于后续维护
    return np.concatenate([
        obs_dict['ego_state'],        # 9 dimensions (自车位置/速度/航向)
        obs_dict['lane_info'],        # 2 dimensions (车道偏移/航向偏差)
        obs_dict['lidar'],            # 240 dimensions (激光雷达特征)
        obs_dict['nearby_vehicles'],  # 20 dimensions (周围车辆状态)
        obs_dict['waypoints']         # 36 dimensions (导航点信息)
    ]).astype(np.float32)


# ===================== Offline Dataset Class =====================
class OfflineDataset:
    """离线数据集加载器"""
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        if not file_path.endswith(('.hdf5', '.h5')):
            raise ValueError(f"Unsupported file format: {file_path}, only .hdf5/.h5 allowed")
        
        self.file_path = file_path
        self._load_data()
    
    def _load_data(self):
        """加载HDF5数据集（优化内存占用与维度校验）"""
        print(f"Loading dataset from {self.file_path}...")
        try:
            with h5py.File(self.file_path, 'r') as f:
                # 加载核心数据（统一转为float32减少内存消耗）
                self.observations = torch.tensor(f['observations'][:], dtype=torch.float32)
                self.actions = torch.tensor(f['actions'][:], dtype=torch.float32)
                self.rewards = torch.tensor(f['rewards'][:], dtype=torch.float32)
                self.next_observations = torch.tensor(f['next_observations'][:], dtype=torch.float32)
                self.dones = torch.tensor(f['done'][:], dtype=torch.float32)
                
                # 维度一致性校验，避免训练中维度不匹配
                data_len = len(self.observations)
                assert len(self.actions) == data_len, f"Actions length({len(self.actions)}) mismatch with observations({data_len})"
                assert len(self.rewards) == data_len, f"Rewards length({len(self.rewards)}) mismatch with observations({data_len})"
                
                self.info = {}
                if 'info' in f and isinstance(f['info'], h5py.Group):
                    for key in f['info'].keys():
                        self.info[key] = torch.tensor(f['info'][key][:], dtype=torch.float32)
                        assert len(self.info[key]) == data_len, f"Info[{key}] length mismatch with observations"
                else:
                    print("Warning: 'info' group not found or invalid in HDF5 file")
        
        except Exception as e:
            raise RuntimeError(f"Dataset loading failed: {str(e)}") from e
        
        print(f"Dataset loaded successfully!")
        print(f"Observations shape: {self.observations.shape}")
        print(f"Actions shape: {self.actions.shape}")
        print(f"Rewards shape: {self.rewards.shape}")
        print(f"Next observations shape: {self.next_observations.shape}")
        print(f"Dones shape: {self.dones.shape}")
        
        self._print_statistics()
    
    def _print_statistics(self):
        """打印数据集统计信息（补充奖励范围与动作分量标注）"""
        print("\n=== Dataset Statistics ===")
        print(f"Total timesteps: {len(self.observations):,}")
        
        # 奖励统计（均值、范围、标准差）
        reward_mean = self.rewards.mean().item()
        reward_min = self.rewards.min().item()
        reward_max = self.rewards.max().item()
        reward_std = self.rewards.std().item()
        print(f"Average reward: {reward_mean:.3f} ± {reward_std:.3f} (Range: {reward_min:.3f} ~ {reward_max:.3f})")
        print(f"Done rate: {self.dones.mean().item():.2%}")
        
        # 状态变量统计（均值、范围）
        if hasattr(self, 'observations') and self.observations is not None:
            print("\n--- State Variables Statistics ---")
            obs_mean = self.observations.mean(dim=0).numpy()
            obs_std = self.observations.std(dim=0).numpy()
            obs_min = self.observations.min(dim=0).values.numpy()
            obs_max = self.observations.max(dim=0).values.numpy()
            
            # 假设状态变量有多个维度，按维度显示统计信息
            for i in range(min(9, len(obs_mean))):  # 限制显示前5个维度避免输出过长
                print(f"State dim {i}: Mean={obs_mean[i]:.3f}, Std={obs_std[i]:.3f}, Range=[{obs_min[i]:.3f}, {obs_max[i]:.3f}]")
            if len(obs_mean) > 9:
                print(f"... and {len(obs_mean) - 9} more dimensions")
        
        # 事件统计
        if 'is_collision' in self.info:
            collision_rate = self.info['is_collision'].mean().item()
            print(f"Collision rate: {collision_rate:.2%} (Total: {int(self.info['is_collision'].sum().item())})")
        
        if 'is_off_road' in self.info:
            off_road_rate = self.info['is_off_road'].mean().item()
            print(f"Off-road rate: {off_road_rate:.2%} (Total: {int(self.info['is_off_road'].sum().item())})")
        
        # 动作分量统计（均值、范围、标准差）
        print("\n--- Action Components Statistics ---")
        action_names = ['Throttle (油门)', 'Steer (转向)', 'Brake (刹车)']
        action_mean = self.actions.mean(dim=0).numpy()
        action_std = self.actions.std(dim=0).numpy()
        action_min = self.actions.min(dim=0).values.numpy()
        action_max = self.actions.max(dim=0).values.numpy()
        
        for i, (name, mean, std, min_val, max_val) in enumerate(zip(action_names, action_mean, action_std, action_min, action_max)):
            print(f"{name} - Mean: {mean:.3f}, Std: {std:.3f}, Range: [{min_val:.3f}, {max_val:.3f}]")
        
        # 额外添加动作值的分布概况
        action_range = action_max - action_min
        print(f"Action value ranges: {[f'{r:.3f}' for r in action_range]}")
    
    def get_datasets(self, val_ratio=0.1, test_ratio=0.1):
        """获取训练集、验证集和测试集（固定随机种子确保可复现）"""
        dataset_size = len(self.observations)
        
        # 计算各数据集大小
        val_size = int(dataset_size * val_ratio)
        test_size = int(dataset_size * test_ratio)
        train_size = dataset_size - val_size - test_size
        
        # 确保数据集大小之和等于原始大小（处理四舍五入误差）
        if train_size + val_size + test_size != dataset_size:
            diff = dataset_size - (train_size + val_size + test_size)
            train_size += diff
        
        # 创建完整数据集
        full_dataset = TensorDataset(
            self.observations, 
            self.actions, 
            self.rewards, 
            self.next_observations, 
            self.dones
        )
        
        # 先分割出训练集和临时集（验证+测试）
        train_dataset, temp_dataset = random_split(
            full_dataset,
            [train_size, val_size + test_size],
            generator=torch.Generator().manual_seed(42)  # 固定种子确保可复现
        )
        
        # 再从临时集中分割出验证集和测试集
        val_dataset, test_dataset = random_split(
            temp_dataset,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(43)  # 使用不同种子但保持固定
        )
        
        return train_dataset, val_dataset, test_dataset
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小({len(self.buffer)})小于批次大小({batch_size})")
            
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).view(-1, 1)
        
        return states, actions, rewards, next_states, dones

    def size(self): 
        return len(self.buffer)
    

# ===================== Trainer Class =====================
class OfflineTrainer:
    """离线训练器，支持Diffusion_QL"""
    def __init__(self, model, train_loader, val_loader, device='cuda', save_path='./offline_models'):
        # 模型与设备配置（补充设备校验）
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # 训练历史记录（初始化空列表避免索引错误）
        self.train_metrics = {'bc_loss': [], 
                              'ql_loss': [], 
                              'actor_loss': [], 
                              'critic_loss': [], 
                              'val_loss': []}

        self.log_dir = f'./training_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.log_dir, exist_ok=True)

    def train_off_policy_agent(self, replay_buffer, iterations, batch_size):
        # 训练模型
        metrics = self.model.train(
            replay_buffer, 
            iterations, 
            batch_size
        )
        return metrics

    def save_model(self, identifier):
        """保存模型权重"""
        if isinstance(self.model, Diffusion_QL):
            # Diffusion_QL模型保存（使用其内置方法）
            self.model.save_model(self.save_path, id=identifier)
        else:
            # 普通模型保存
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'best_val_loss': self.best_val_loss
            }, f"{self.save_path}/model_{identifier}.pth")
        
        print(f"模型已保存至 {self.save_path}/model_{identifier}.pth")

    def plot_training_history(self, save_fig=True):
        """绘制训练历史曲线，val_loss与bc_loss同图，且val_loss单独成图，其他损失不包含val_loss"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        
        plt.style.use('seaborn-v0_8')
        
        # 获取所有损失类型并调整顺序，确保bc_loss和val_loss在前
        loss_types = list(self.train_metrics.keys())
        # 确保bc_loss在前面
        if 'bc_loss' in loss_types:
            loss_types.insert(0, loss_types.pop(loss_types.index('bc_loss')))
        # 确保val_loss紧随bc_loss之后
        if 'val_loss' in loss_types:
            loss_types.insert(1, loss_types.pop(loss_types.index('val_loss')))
        
        num_losses = len(loss_types)
        
        # 创建适当大小的画布和子图
        fig, axes = plt.subplots(num_losses, 1, figsize=(12, 4 * num_losses), sharex=True)
        
        # 确保axes是数组形式，即使只有一个子图
        if num_losses == 1:
            axes = [axes]
        
        # 为每种损失绘制曲线
        for i, loss_name in enumerate(loss_types):
            ax = axes[i]
            loss_values = self.train_metrics[loss_name]
            
            # 绘制当前损失曲线
            ax.plot(loss_values, label=f'{loss_name.replace("_", " ").title()}', 
                    linewidth=2, color='blue' if loss_name == 'bc_loss' else 'green')
            
            # 仅在bc_loss图中添加val_loss曲线作为对比
            if loss_name == 'bc_loss' and 'val_loss' in self.train_metrics:
                ax.plot(self.train_metrics['val_loss'], label='Validation Loss', 
                        linewidth=2, linestyle='--', color='red')
            
            ax.set_title(f'{loss_name.replace("_", " ").title()} History', fontsize=14)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 设置x轴标签（只在最后一个子图设置）
        axes[-1].set_xlabel('Epoch', fontsize=12)
        
        plt.tight_layout()
        
        if save_fig:
            # 确保日志目录存在
            os.makedirs(self.log_dir, exist_ok=True)
            fig.savefig(f"{self.log_dir}/training_history.png")
            print(f"训练历史图表已保存至 {self.log_dir}/training_history.png")
        else:
            plt.show()
        
        plt.close()
    

# ===================== Main Function =====================
def main():
    parser = argparse.ArgumentParser(description='Train on EasyCarla Offline Dataset')
    parser.add_argument('--data_path', type=str, default='easycarla_offline_dataset.hdf5',
                       help='Path to the HDF5 dataset file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--buffer_capacity', type=int, default=30000, help='Replay buffer capacity')
    parser.add_argument('--replay_buffer_sample_ratio', type=float, default=0.01, help='Replay buffer sample ratio')
    
    parser.add_argument('--iterations', type=int, default=1000, help='Iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--val_ratio', type=float, default=0.005, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.005, help='Test ratio')
    args = parser.parse_args()
    
    # 加载数据集
    dataset = OfflineDataset(args.data_path)
    train_loader, val_loader, test_loader = dataset.get_datasets(val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    # ===================== Initialize Model =====================
    state_dim = 307
    action_dim = 3
    max_action = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Diffusion_QL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,
        tau=0.005,
        eta=0.01,
        beta_schedule='vp',
        n_timesteps=5
    )
    
    # 创建训练器
    trainer = OfflineTrainer(model, train_loader, val_loader, device)

    # 创建保存目录（确保存在）
    checkpoint_dir = './params_dql'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 首先将所有训练数据添加到回放缓冲区
    replay_buffer = ReplayBuffer(args.buffer_capacity)

    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 1. 配置采样参数（按需调整，二选一即可）
        sample_ratio = args.replay_buffer_sample_ratio  # 按训练集比例采样
        max_samples = int(len(train_loader.dataset) * sample_ratio)  # 按比例计算目标采样数
        print(f"训练集总样本数: {len(train_loader.dataset)}")
        print(f"目标采样数: {max_samples}")

        # 2. 随机选择目标索引
        target_indices = np.random.choice(len(train_loader.dataset), max_samples, replace=False)

        # 3. 直接按索引从数据集中抽取样本并添加到buffer
        count = 0
        for idx in target_indices:
            # 直接从数据集中获取单个样本
            sample = train_loader.dataset[idx]
            
            # 假设数据集返回的是 (state, action, reward, next_state, done) 元组
            state, action, reward, next_state, done = sample

            # 转换为numpy并添加到回放缓冲区
            replay_buffer.add(
                state=state.numpy() if hasattr(state, 'numpy') else state,
                action=action.numpy() if hasattr(action, 'numpy') else action,
                reward=reward.item() if hasattr(reward, 'item') else reward,
                next_state=next_state.numpy() if hasattr(next_state, 'numpy') else next_state,
                done=done.item() if hasattr(done, 'item') else done
            )
            
            count += 1
            
            if count % 10000 == 0:  # 每100条打印一次
                print(f"已添加 {count}/{max_samples} 样本到回放缓冲区")

        print(f"回放缓冲区最终大小: {replay_buffer.size()}")
        
        # 训练
        iterations = args.iterations
        batch_size = args.batch_size
        metrics = trainer.train_off_policy_agent(
            replay_buffer,
            iterations,
            batch_size
        )
        
        if epoch % 10 == 0 or epoch == args.epochs:
            avg_losses = {}
            for key in metrics:
                try:
                    valid_losses = [loss for loss in metrics[key] if isinstance(loss, (int, float)) and np.isfinite(loss)]
                    avg_losses[key] = np.mean(valid_losses) if valid_losses else 0.0
                except (TypeError, ValueError):
                    avg_losses[key] = 0.0
            
            val_loss = model.validate(val_loader)
            
            print(f"\nEpoch {epoch}:")
            for key, loss in avg_losses.items():
                count = len([loss_val for loss_val in metrics[key] if isinstance(loss_val, (int, float)) and np.isfinite(loss_val)])
                print(f"  {key}: {loss:.4f} (n={count})")
                trainer.train_metrics[key].append(loss)           
            trainer.train_metrics['val_loss'].append(val_loss)
        else:
            avg_losses = {}
            for key in metrics:
                try:
                    valid_losses = [loss for loss in metrics[key] if isinstance(loss, (int, float)) and np.isfinite(loss)]
                    avg_losses[key] = np.mean(valid_losses) if valid_losses else 0.0
                except (TypeError, ValueError):
                    avg_losses[key] = 0.0
                
            print(f"\nEpoch {epoch}:")
            for key, loss in avg_losses.items():
                count = len([loss_val for loss_val in metrics[key] if isinstance(loss_val, (int, float)) and np.isfinite(loss_val)])
                print(f"  {key}: {loss:.4f} (n={count})")
                trainer.train_metrics[key].append(loss)
            # Handle case where val_loss list is empty
            if trainer.train_metrics['val_loss']:
                trainer.train_metrics['val_loss'].append(trainer.train_metrics['val_loss'][-1])
            else:
                # 当val_loss历史为空时，使用一个默认值（例如0.0）而不是尝试访问空列表
                trainer.train_metrics['val_loss'].append(0.0)
            
        if epoch % 20 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_model(checkpoint_dir, id=f"epoch_{epoch}")            
            print(f"Checkpoint saved: model={f'{checkpoint_dir}'}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_model(checkpoint_dir, id=f"epoch_{epoch}")            
    print(f"Training completed! Final model saved as '{f'{checkpoint_dir}'}'")
    
    # 绘制训练历史
    trainer.plot_training_history()
    print("Training history plot saved as 'training_history.png'")
    
    # 测试模型性能
    test_loss = model.evaluate(test_loader)
   

if __name__ == "__main__":
    main()