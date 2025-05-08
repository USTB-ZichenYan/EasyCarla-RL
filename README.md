# EasyCarla-RL

EasyCarla-RL is based on the [gym-carla](https://github.com/cjy1992/gym-carla) project, with significant improvements for ease of use and reinforcement learning tasks.

A lightweight and beginner-friendly OpenAI Gym environment built on the CARLA simulator.

## Overview

EasyCarla-RL offers a simple and efficient interface to use the CARLA simulator for reinforcement learning (RL) tasks.
It provides essential observation components such as LiDAR scans, ego vehicle states, nearby vehicle information, and waypoints,
allowing users to train and evaluate RL agents without complex engineering overhead. The environment is designed to be both accessible to beginners and powerful enough for advanced RL research.

## Features

- Lightweight and easy-to-integrate CARLA wrapper
- Specifically designed for reinforcement learning applications
- Rich observations: LiDAR, ego vehicle state, nearby vehicles, and waypoints
- Built-in support for safety-aware RL with reward and cost signals
- Configurable settings: traffic lights, number of vehicles, LiDAR range, and more
- Visualization support for waypoints and vehicle surroundings
- Fully compatible with the OpenAI Gym API

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

