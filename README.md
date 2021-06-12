# Deep Reinforcement Learning - Target Reaching
Sawyer robot learning to reach a target with paralleled Soft Actor-Critic (SAC) algorithm, using CoppeliaSim and PyRep for robot simulation. The environment is wrapped into OpenAI Gym format.

## Installation

 - First, start by installing [CoppeliaSim](https://www.coppeliarobotics.com/) V4.2.0, previously called V-REP and [PyRep](https://github.com/stepjam/PyRep), a toolkit for robot learning research, built on top of CoppeliaSim.

 - To install the required packages for SAC agent, move to the repository folder and run,

```
pip install -r requirements.txt
```

 - To train and test the policy learnt, run

```
python sac_learn.py --train
python sac_learn.py --test
```
