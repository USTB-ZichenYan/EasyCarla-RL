🚀 [**New Dataset Released !**](#-download-dataset)

# EasyCarla-RL: A lightweight and beginner-friendly OpenAI Gym environment built on the CARLA simulator

## Overview

EasyCarla-RL provides a lightweight and easy-to-use Gym-compatible interface for the CARLA simulator, specifically tailored for reinforcement learning (RL) applications. It integrates essential observation components such as LiDAR scans, ego vehicle states, nearby vehicle information, and waypoints. The environment supports safety-aware learning with reward and cost signals, visualization of waypoints, and customizable parameters including traffic settings, number of vehicles, and sensor range. EasyCarla-RL is designed to help both researchers and beginners efficiently train and evaluate RL agents without heavy engineering overhead.

<div align="center">

<table>
  <tr>
    <td align="center"><b>🚗 Custom Traffic</b></td>
    <td align="center"><b>🔄 View Mode Switch</b></td>
    <td align="center"><b>🗺️ Multi-Town Maps</b></td>
  </tr>
  <tr>
    <td><img src="assets/part1.gif" width="100%"/></td>
    <td><img src="assets/part2.gif" width="100%"/></td>
    <td><img src="assets/part3.gif" width="100%"/></td>
  </tr>
</table>

</div>

## Installation

Clone the repository:

```bash
git clone https://github.com/silverwingsbot/EasyCarla-RL.git
cd EasyCarla-RL
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Install EasyCarla-RL as a local Python package:

```bash
pip install -e .
```

Make sure you have a running [CARLA simulator](https://carla.org/) server compatible with your environment.

For detailed installation instructions, please refer to the [official CARLA docs](https://carla.readthedocs.io/en/0.9.13/start_quickstart/)

## Quick Start

Run a simple demo to interact with the environment:

```bash
python easycarla_demo.py
```

This script demonstrates how to:
- Create and reset the environment
- Select random or autopilot actions
- Step through the environment and receive observations, rewards, costs, and done signals

Make sure your CARLA server is running before executing the demo.

## Advanced Example: Evaluation with Diffusion Q-Learning

For a more advanced usage, you can run a pre-trained [Diffusion Q-Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) agent in the EasyCarla-RL environment:

```bash
cd example
python run_dql_in_carla.py
```

Make sure you have downloaded or prepared a trained model checkpoint under the `example/params_dql/` directory.

This example demonstrates:
- Loading a pre-trained RL agent
- Interacting with EasyCarla-RL for evaluation
- Evaluating the performance of a real RL model on a simulated autonomous driving task

## 📥 Download Dataset

We provide an offline dataset used for training and evaluating RL agents in the EasyCarla-RL environment.

This dataset includes over **7,000 trajectories** and **1.1 million timesteps**, collected from a mix of expert and random policies (with an **8:2 ratio** of expert to random). The data is stored in **HDF5 format**.

You can download it from either of the following sources:

*  [Download from Hugging Face (direct link)](https://huggingface.co/datasets/silverwingsbot/easycarla/resolve/main/easycarla_offline_dataset.hdf5)
*  [Download from 百度网盘 (提取码: 2049)](https://pan.baidu.com/s/1yhCFzl4RFHzxfszebYnOIg?pwd=2049)

Filename: `easycarla_offline_dataset.hdf5` Size: \~2.76 GB Format: HDF5

### Observation Format

Each observation in the dataset is stored as a **307-dimensional flat vector**, constructed by concatenating several components in the following order:

```python
# Flattening function used during data generation

def flatten_obs(obs_dict):
    return np.concatenate([
        obs_dict['ego_state'],        # 9 dimensions
        obs_dict['lane_info'],        # 2 dimensions
        obs_dict['lidar'],            # 240 dimensions
        obs_dict['nearby_vehicles'],  # 20 dimensions
        obs_dict['waypoints']         # 36 dimensions
    ]).astype(np.float32)  # Total: 307 dimensions
```

This format allows for efficient training of neural networks while preserving critical spatial and semantic information.

## Project Structure

```
EasyCarla-RL/                    
├── easycarla/                 # Main environment module (Python package)
│   ├── envs/                     
│   │   ├── __init__.py           
│   │   └── carla_env.py       # Carla environment wrapper following the Gym API
│   └── __init__.py               
├── example/                   # Advanced example
│   ├── agents/                   
│   ├── params_dql/               
│   ├── utils/                    
│   └── run_dql_in_carla.py    # Script to run a pretrained RL model
├── easycarla_demo.py          # Quick Start demo script (basic Gym-style environment interaction)
├── requirements.txt              
├── setup.py                      
└── README.md                     
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

Created by [SilverWings](https://github.com/silverwingsbot)

## 💓 Acknowledgement

This project is made possible thanks to the following outstanding open-source contributions:

- [CARLA](https://github.com/carla-simulator/carla)
- [gym-carla](https://github.com/cjy1992/gym-carla)
- [Diffusion Q-Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)
