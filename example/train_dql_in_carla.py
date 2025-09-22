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
        # 新增文件合法性校验，提前规避加载错误
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
                
                # 加载信息字典（补充键值校验）
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
        # 补充奖励范围统计，便于分析数据质量
        print(f"Average reward: {self.rewards.mean().item():.3f} (Range: {self.rewards.min().item():.3f} ~ {self.rewards.max().item():.3f})")
        print(f"Done rate: {self.dones.mean().item():.2%}")
        
        if 'is_collision' in self.info:
            collision_rate = self.info['is_collision'].mean().item()
            print(f"Collision rate: {collision_rate:.2%} (Total: {int(self.info['is_collision'].sum().item())})")
        
        if 'is_off_road' in self.info:
            off_road_rate = self.info['is_off_road'].mean().item()
            print(f"Off-road rate: {off_road_rate:.2%} (Total: {int(self.info['is_off_road'].sum().item())})")
        
        # 动作分量标注中文名称，提升可读性
        action_names = ['Throttle (油门)', 'Steer (转向)', 'Brake (刹车)']
        action_mean = self.actions.mean(dim=0).numpy()
        action_std = self.actions.std(dim=0).numpy()
        for i, (name, mean, std) in enumerate(zip(action_names, action_mean, action_std)):
            print(f"{name} - Mean: {mean:.3f}, Std: {std:.3f}")
    
    def get_datasets(self, val_ratio=0.1):
        """获取训练集和验证集（固定随机种子确保可复现）"""
        # 划分训练集和验证集（设置seed确保每次划分一致）
        dataset_size = len(self.observations)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size
        
        # 固定split种子，解决每次运行数据划分不一致问题
        train_dataset, val_dataset = random_split(
            TensorDataset(
                self.observations, 
                self.actions, 
                self.rewards, 
                self.next_observations, 
                self.dones
            ),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定种子
        )
        
        return train_dataset, val_dataset
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, reward, next_state, done): 
        # 直接存储numpy数组，避免在添加时转换为tensor
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区大小({len(self.buffer)})小于批次大小({batch_size})")
            
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # 转换为tensor并移动到设备，确保正确的维度
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
        self.train_metrics = {
            'total_loss': [] 
        }


        
        # 日志配置（补充时间戳避免覆盖）
        self.log_dir = f'./training_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.log_dir, exist_ok=True)
        # 初始化CSV日志，便于后续分析
        self.csv_log_path = os.path.join(self.log_dir, 'train_log.csv')
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_bc_loss', 'train_ql_loss', 'val_loss'])
            writer.writeheader()

    
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
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.001, help='Validation ratio')
    parser.add_argument('--minimal_size', type=int, default=1000, help='Minimal replay buffer size for training')
    args = parser.parse_args()
    
    # 加载数据集
    dataset = OfflineDataset(args.data_path)
    train_loader, val_loader = dataset.get_datasets(val_ratio=args.val_ratio)
    
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
    replay_buffer = ReplayBuffer(capacity=1000000)

    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 1. 配置采样参数（按需调整，二选一即可）
        sample_ratio = 0.001  # 按训练集比例采样（例如取0.1%）
        # max_samples = 50000  # 按固定数量采样（例如最多5万条）
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
            
            # 进度提示
            if count % 1000 == 0:  # 每100条打印一次
                print(f"已添加 {count}/{max_samples} 样本到回放缓冲区")

        print(f"回放缓冲区最终大小: {replay_buffer.size()}")
        
        # 训练
        iterations = 1000
        metrics = trainer.train_off_policy_agent(
            replay_buffer,
            iterations,
            batch_size=args.batch_size
        )
        
        # 验证和打印训练信息
        if epoch % 2 == 0 or epoch == args.epochs:
            # 计算各项平均损失
            avg_losses = {}
            for key in metrics:
                try:
                    valid_losses = [loss for loss in metrics[key] if isinstance(loss, (int, float)) and np.isfinite(loss)]
                    avg_losses[key] = np.mean(valid_losses) if valid_losses else 0.0
                except (TypeError, ValueError):
                    avg_losses[key] = 0.0
            
            # 执行验证
            val_loss = model.validate(val_loader)
            total_train_loss = np.mean(list(avg_losses.values())) if avg_losses else 0.0
            
            # 打印结果
            print(f"\nEpoch {epoch}:")
            for key, loss in avg_losses.items():
                count = len([loss_val for loss_val in metrics[key] if isinstance(loss_val, (int, float)) and np.isfinite(loss_val)])
                print(f"  {key}: {loss:.4f} (n={count})")
            print(f"  总训练损失: {total_train_loss:.4f}, 验证损失: {val_loss:.4f}\n")
            
            trainer.train_metrics['total_loss'].append(total_train_loss)
        else:
            try:
                total_train_loss = np.mean([
                    np.mean([loss for loss in metrics[key] if isinstance(loss, (int, float)) and np.isfinite(loss)] or [0])
                    for key in metrics
                ]) if metrics else 0.0
            except Exception as e:
                print(f"计算训练损失时出错: {e}")
                total_train_loss = 0.0
                
            print(f"Epoch {epoch}: 训练损失 = {total_train_loss:.4f}")
            trainer.train_metrics['total_loss'].append(total_train_loss)

            
        # 定期保存检查点
        if epoch % 2 == 0:
            # 确保检查点目录存在
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存模型权重
            model_save_path = model.save_model(checkpoint_dir, id=f"epoch_{epoch}")
            
            # 保存训练状态（包含更完整的信息）
            checkpoint = {
                'epoch': epoch,
                'train_metrics': trainer.train_metrics,
                'actor_optimizer': model.actor_optimizer.state_dict(),
                'critic_optimizer': model.critic_optimizer.state_dict(),
                'loss': avg_losses  # 增加当前损失，便于后续分析
            }
            
            state_save_path = f'{checkpoint_dir}/training_state_epoch_{epoch}.pth'
            torch.save(checkpoint, state_save_path)
            print(f"Checkpoint saved: model={model_save_path}, state={state_save_path}")

    # 保存最终模型
    final_checkpoint = {
        'epoch': args.epochs,
        'train_metrics': trainer.train_metrics,
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict()
    }

    final_save_path = f'{checkpoint_dir}/training_state_final.pth'
    torch.save(final_checkpoint, final_save_path)
    print(f"Training completed! Final model saved as '{final_save_path}'")  # 修复命名不一致问题
    
    # 绘制训练历史
    trainer.plot_training_history()
    print("Training history plot saved as 'training_history.png'")
    
    # 测试模型性能
    # test_model_performance(model, dataset, device)

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