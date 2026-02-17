# DRL-SPICE

DRL-SPICE is a modular infrastructure for automated analog circuit optimization using reinforcement learning and ngspice simulation.

This framework provides a complete abstraction layer that connects reinforcement learning algorithms with SPICE simulation, enabling automated circuit sizing, performance evaluation, and optimization.

It is designed for research and engineering applications in:

- analog / RF circuit optimization
- reinforcement learning-based circuit design
- SPICE simulation automation
- AI-driven electronic design automation (EDA)

---

## Core Components

DRL-SPICE consists of two main modules:

### 1. RL Environment Layer (`env/`)

Provides a fully modular reinforcement learning environment abstraction.

Features:

- Action modeling and decoding
- Observation modeling
- Reward and objective modeling
- Constraint handling
- Termination logic
- Modular environment composition
- Logging and simulation utilities

Key modules:

```
env/
├── action/           # action representation and decoding
├── observation/      # observation construction
├── reward/           # reward and objective models
├── termination/      # termination rules
├── logger/           # logging utilities
├── simulator.py      # simulator interface
├── factory.py        # environment builder
└── modular.py        # modular LNA environment
```

---

### 2. ngspice Integration Layer (`ngspice/`)

Provides a complete Python interface for ngspice simulation and circuit analysis.

Features:

- ngspice kernel execution
- automatic netlist generation and patching
- simulation execution and monitoring
- analysis result extraction
- performance metric parsing

Key modules:

```
ngspice/
├── kernel.py            # ngspice execution engine
├── netlist/
│   ├── circuit.py      # circuit representation
│   ├── designer.py     # netlist generation
│   └── patcher.py      # parameter injection
│
├── analysis/
│   ├── readers/        # performance metric extractors
│   │   ├── noise.py
│   │   ├── sparams.py
│   │   ├── stability.py
│   │   └── linearity.py
│
└── templates/
    ├── cs_lna/         # common-source LNA templates
    └── cgcs_lna/       # cascode LNA templates
```

---

## Supported Circuit Templates

Currently supported circuits:

- Common Source LNA
- Cascode LNA

Templates include:

- S-parameter analysis
- Noise figure analysis
- Frequency response analysis

Located in:

```
ngspice/templates/
```

---

## Architecture Overview

The framework connects reinforcement learning with circuit simulation:

```
RL Agent
   ↓
Environment (env/)
   ↓
Action Decoder
   ↓
Netlist Patcher
   ↓
ngspice Kernel
   ↓
Analysis Readers
   ↓
Performance Metrics
   ↓
Reward / Objective
   ↓
RL Agent Update
```

---

## Requirements

Linux environment required.

Tested on:

- Ubuntu 20.04
- Ubuntu 22.04

Dependencies:

Python:

```
python 3.9+
numpy
```

System:

```
ngspice
```

Install ngspice:

```
sudo apt install ngspice
```

Verify:

```
ngspice -v
```

---

## Installation

Clone repository:

```
git clone https://github.com/jyhong99/drl_spice.git
cd drl_spice
```

Set Python path:

```
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## Usage

This framework is designed to be integrated with external reinforcement learning algorithms.

Example workflow:

1. Create environment using `env.factory`
2. Initialize RL agent (e.g., SAC, TD3, DDPG, PPO)
3. Agent proposes circuit parameters
4. Parameters are injected into netlist
5. ngspice runs simulation
6. Performance metrics are extracted
7. Reward is computed
8. Agent updates policy

---

## Example Integration

Compatible with external RL libraries such as:

- SAC
- TD3
- DDPG
- PPO

Designed to integrate directly with:

https://github.com/jyhong99/rl_algorithms

---

## Design Goals

- Fully modular circuit optimization environment
- clean separation between RL and SPICE layers
- reusable EDA infrastructure
- scalable and extensible architecture
- research-grade implementation

---

## Applications

- analog circuit optimization
- RF circuit design automation
- reinforcement learning research
- EDA system development
- simulation-based optimization

---

## Platform

Linux only

Required:

- ngspice
- Python 3.9+

---

## Author

Junyoung Hong  
M.S. Student, AI Semiconductor Engineering  
Hanyang University

Research focus:

- Reinforcement learning
- Analog circuit optimization
- SPICE automation
- AI-driven EDA

GitHub:  
https://github.com/jyhong99

---

## License

MIT License
