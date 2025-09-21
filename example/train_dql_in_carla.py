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

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ===================== Helper Functions =====================
def convert_obs_dict_to_vector(obs_dict):
    """Convert observation dictionary to a flattened state vector."""
    return np.concatenate([
        obs_dict['ego_state'],        # 9 dimensions
        obs_dict['lane_info'],        # 2 dimensions
        obs_dict['lidar'],            # 240 dimensions
        obs_dict['nearby_vehicles'],  # 20 dimensions
        obs_dict['waypoints']         # 36 dimensions
    ]).astype(np.float32)


# ===================== Offline Dataset Class =====================
class OfflineDataset:
    """离线数据集加载器"""
    def __init__(self, file_path):
        self.file_path = file_path
        self._load_data()
    
    def _load_data(self):
        """加载HDF5数据集"""
        print(f"Loading dataset from {self.file_path}...")
        with h5py.File(self.file_path, 'r') as f:
            # 加载主要数据
            self.observations = torch.tensor(f['observations'][:], dtype=torch.float32)
            self.actions = torch.tensor(f['actions'][:], dtype=torch.float32)
            self.rewards = torch.tensor(f['rewards'][:], dtype=torch.float32)
            self.next_observations = torch.tensor(f['next_observations'][:], dtype=torch.float32)
            self.dones = torch.tensor(f['done'][:], dtype=torch.float32)
            
            # 加载信息字典
            self.info = {}
            info_group = f['info']
            for key in info_group.keys():
                self.info[key] = torch.tensor(info_group[key][:], dtype=torch.float32)
        
        print(f"Dataset loaded successfully!")
        print(f"Observations shape: {self.observations.shape}")
        print(f"Actions shape: {self.actions.shape}")
        print(f"Rewards shape: {self.rewards.shape}")
        print(f"Next observations shape: {self.next_observations.shape}")
        print(f"Dones shape: {self.dones.shape}")
        
        # 打印数据集统计信息
        self._print_statistics()
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        print("\n=== Dataset Statistics ===")
        print(f"Total timesteps: {len(self.observations):,}")
        print(f"Average reward: {self.rewards.mean().item():.3f}")
        print(f"Done rate: {self.dones.mean().item():.2%}")
        
        if 'is_collision' in self.info:
            collision_rate = self.info['is_collision'].mean().item()
            print(f"Collision rate: {collision_rate:.2%}")
        
        if 'is_off_road' in self.info:
            off_road_rate = self.info['is_off_road'].mean().item()
            print(f"Off-road rate: {off_road_rate:.2%}")
        
        # 动作统计
        print(f"Action mean: {self.actions.mean(dim=0).numpy()}")
        print(f"Action std: {self.actions.std(dim=0).numpy()}")
    
    def get_dataloader(self, batch_size=256, shuffle=True, val_ratio=0.1):
        """获取数据加载器"""
        # 划分训练集和验证集
        dataset_size = len(self.observations)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(
            TensorDataset(
                self.observations, 
                self.actions, 
                self.rewards, 
                self.next_observations, 
                self.dones
            ),
            [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    
# ===================== Trainer Class =====================
class OfflineTrainer:
    """离线训练器，支持Diffusion_QL"""
    def __init__(self, model, train_loader, val_loader, device='cuda', save_path='./offline_models'):
        # 模型与设备配置
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # 训练历史记录
        self.train_metrics = {
            'total_loss': [],
            'actor_loss': [],    # 仅用于Diffusion模型
            'critic_loss': [],   # 仅用于Diffusion模型
            'bc_loss': [],       # 仅用于Diffusion模型
            'ql_loss': []        # 仅用于Diffusion模型
        }

        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # 日志与保存配置
        self.log_dir = './training_logs'
        os.makedirs(self.log_dir, exist_ok=True)
    
    def train_off_policy_agent(self, num_episodes, replay_buffer, minimal_size, batch_size, train_loader):
        return_list = []
        # 初始化指标统计字典
        metrics_history = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'episode_returns': []
        }
        
        progress_bar = tqdm(total=num_episodes, desc="Training", leave=False)
        
        # 创建数据加载器的迭代器
        data_iter = iter(train_loader)
        
        for episode in range(num_episodes):
            try:
                # 从train_loader中获取一批数据
                batch = next(data_iter)
            except StopIteration:
                # 如果数据遍历完了，重新创建迭代器
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            episode_return = 0
            states, actions, rewards, next_states, dones = batch
            
            # 创建mini_buffer用于当前批次的训练
            mini_buffer = ReplayBuffer()
            
            # 将数据加入buffer
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                replay_buffer.add(state, action, reward, next_state, done)
                mini_buffer.add(state, action, reward, next_state, done)
                episode_return += reward
            
            # 训练条件判断
            if replay_buffer.size() > minimal_size:
                # 调用 Diffusion_QL.train() 方法
                metrics = self.model.train(
                        replay_buffer=mini_buffer,
                        iterations=10,
                        batch_size=obs.shape[0]
                        )
                
                # 记录指标
                for key in ['bc_loss', 'ql_loss', 'actor_loss', 'critic_loss']:
                    if key in metrics:
                        metrics_history[key].extend(metrics[key])
            
            return_list.append(episode_return)
            metrics_history['episode_returns'].append(episode_return)
            
            # 更新进度条
            if (episode + 1) % 10 == 0:
                progress_bar.set_postfix({
                    'episode': episode + 1,
                    'return': '%.3f' % np.mean(return_list[-10:]),
                    'bc_loss': '%.4f' % (np.mean(metrics_history['bc_loss'][-10:]) if metrics_history['bc_loss'] else 0),
                    'ql_loss': '%.4f' % (np.mean(metrics_history['ql_loss'][-10:]) if metrics_history['ql_loss'] else 0)
                })
            progress_bar.update(1)
        
        progress_bar.close()
        return return_list, metrics_history

    def validate(self):
        """验证当前模型性能"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                obs, actions, rewards, next_obs, dones = batch
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                # 计算验证损失
                if isinstance(self.model, Diffusion_QL):
                    # 对于Diffusion模型，计算行为克隆损失作为验证指标
                    pred_actions = self.model.actor(obs)
                    loss = nn.MSELoss()(pred_actions, actions)
                else:
                    # 对于普通模型，直接计算预测损失
                    pred_actions = self.model(obs)
                    loss = nn.MSELoss()(pred_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

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
        """绘制训练历史曲线"""
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        
        # 创建画布
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 绘制总损失
        ax1.plot(self.train_metrics['total_loss'], label='Train Total Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Validation Loss', linewidth=2)
        ax1.set_title('Total Loss History', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 绘制Diffusion特有损失（如果有）
        if len(self.train_metrics['actor_loss']) > 0:
            ax2.plot(self.train_metrics['actor_loss'], label='Actor Loss', linewidth=1.5)
            ax2.plot(self.train_metrics['critic_loss'], label='Critic Loss', linewidth=1.5)
            ax2.set_title('Actor & Critic Loss History', fontsize=14)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend()
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Diffusion Metrics Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_fig:
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
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    args = parser.parse_args()
    
    # 加载数据集
    dataset = OfflineDataset(args.data_path)
    train_loader, val_loader = dataset.get_dataloader(
        batch_size=args.batch_size, 
        val_ratio=args.val_ratio
    )
    
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
    
    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss = trainer.train_off_policy_agent()
        
        # 验证
        if epoch % 5 == 0 or epoch == args.epochs:
            val_loss = trainer.validate()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        
        # 定期保存检查点
        if epoch % 20 == 0:
            model.save_model(checkpoint_dir, id=f"epoch_{epoch}")            
            # 保存训练状态（使用 trainer.train_metrics 而非 trainer.train_losses）
            torch.save({
                'epoch': epoch,
                'train_losses': trainer.train_metrics['total_loss'],  # 修改这里
                'val_losses': trainer.val_losses,
                'actor_optimizer': model.actor_optimizer.state_dict(),
                'critic_optimizer': model.critic_optimizer.state_dict()
            }, f'{checkpoint_dir}/training_state_epoch_{epoch}.pth')
            print(f"Checkpoint saved as '{checkpoint_dir}/training_state_epoch_{epoch}.pth'")

    # 保存最终模型时同样修改
    torch.save({
        'epoch': epoch,
        'train_losses': trainer.train_metrics['total_loss'],  # 修改这里
        'val_losses': trainer.val_losses,
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict()
    }, f'{checkpoint_dir}/training_state_final.pth')   
    print("Training completed! Model saved as 'final_model.pth'")
    
    # 绘制训练历史
    trainer.plot_training_history()
    print("Training history plot saved as 'training_history.png'")
    
    # 测试模型性能
    test_model_performance(model, dataset, device)

def test_model_performance(model, dataset, device):
    """测试模型性能"""
    model.eval()
    
    # 随机选择一些测试样本
    test_indices = np.random.choice(len(dataset.observations), min(1000, len(dataset.observations)), replace=False)
    test_obs = dataset.observations[test_indices].to(device)
    test_actions = dataset.actions[test_indices].to(device)
    
    with torch.no_grad():
        if hasattr(model, 'sample_action'):  # Diffusion模型
            pred_actions = []
            for i in tqdm(range(len(test_obs)), desc="Testing"):
                obs = test_obs[i].unsqueeze(0)
                pred_action = model.sample_action(obs.cpu().numpy())
                pred_actions.append(pred_action)
            pred_actions = torch.tensor(np.array(pred_actions), device=device, dtype=torch.float32)
        else:  # BC模型
            pred_actions = model(test_obs)
        
        mse_loss = nn.MSELoss()(pred_actions, test_actions).item()
        mae_loss = nn.L1Loss()(pred_actions, test_actions).item()
    
    print(f"\n=== Test Performance ===")
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"MAE Loss: {mae_loss:.6f}")
    
    # 动作分量误差
    action_errors = (pred_actions - test_actions).abs().mean(dim=0).cpu().numpy()
    action_names = ['Throttle', 'Steer', 'Brake']
    for i, name in enumerate(action_names):
        print(f"{name} Error: {action_errors[i]:.4f}")

if __name__ == "__main__":
    main()
