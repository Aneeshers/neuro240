# neuro240

This final project attempts to cluster different agent types (random, deterministic, value-based) based on their trajectories. Unfortunately we can't differientiate between random and value-based 😓.

## Notes for Aneesh for Future Development

**Files tree:**

```
/Users/aneesh/Documents/neuro240
├── .gitattributes
├── README.md
├── agents
│   ├── d_agent.py
│   ├── r_agent.py
│   └── v_agent.py
├── dataset
│   ├── gen.py
│   └── storage.py
├── env
│   └── grid.py
├── model
│   ├── full.py
│   ├── ln.py
│   └── pp.py
├── train.py
├── train.sh
└── utils
    ├── log.py
    ├── loss.py
    └── visualize.py
```


- **agents**: Contains agent models implementing different strategies.
  - `d_agent.py`: Implements deterministic agents with predefined movement (we loop throug samples) patterns.
  - `r_agent.py`:random agents.
  - `v_agent.py`: value-iteration trained agents.

- **dataset**: Handles dataset generation and storage.

- **env**: 
  - `grid.py`: grid environment with reward config in init

- **model**:
  - `full.py`: The combined end-to-end model used to predict actions from latent representations.
  - `ln.py`: Latent space network for clustering trajectories.
  - `pp.py`: Prediction network that uses the latent space to determine the next state/action.

- **train.py**: train script

- **train.sh**: to run on HPC


### Running Experiments
`python train.py` should work. Be careful with Jax Cuda, future work should convert codebase to jax for major speed up.
- **Agent Types**:
  - Toggle experiments by modifying the `agent_types` variable in `train.py`.
.
