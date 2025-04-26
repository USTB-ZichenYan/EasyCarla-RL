# EasyCarla-RL

EasyCarla-RL is based on the [gym-carla](https://github.com/cjy1992/gym-carla) project, with significant improvements for ease of use and reinforcement learning tasks.

A simple and easy-to-use OpenAI Gym environment based on the CARLA simulator.

## Overview

EasyCarla-RL provides a lightweight and beginner-friendly interface to use the CARLA simulator for reinforcement learning tasks.
It integrates essential observations such as Lidar scans, ego vehicle states, nearby vehicle information, and waypoints,
making it easy to train and evaluate RL agents without heavy engineering efforts.

## Features

- Lightweight and easy-to-use CARLA wrapper
- Designed specifically for reinforcement learning tasks
- Observations include Lidar, ego vehicle state, nearby vehicles, and waypoints
- Supports safety-aware RL with reward and cost outputs
- Configurable traffic lights, number of vehicles, and Lidar range
- Visualization support for waypoints
- Compatible with OpenAI Gym API

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

## Quick Start

Run a simple example with a pre-trained [Diffusion Q-Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) agent:

```bash
cd example
python run_dql_in_carla.py
```

Make sure to download or prepare a trained model checkpoint under the `example/params_dql/` directory.

## Project Structure

```
EasyCarla-RL/
├── easycarla/
│   ├── envs/
│   │   ├── __init__.py
│   │   └── carla_env.py
│   └── __init__.py
├── example/
│   ├── agent_dql/
│   ├── params_dql/
│   └── run_dql_in_carla.py
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

Created by [SilverWings](https://github.com/silverwingsbot)
