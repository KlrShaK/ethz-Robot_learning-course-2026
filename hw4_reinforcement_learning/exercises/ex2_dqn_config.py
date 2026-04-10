"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # Tune the following hyperparameters
    "lr": 1e-3,            # Unchanged
    "epsilon": 0.1,        # exploration rate; 10% gives enough random coverage early on
    "target_update": 50,   # 10 copies online -> target every N gradient steps; less frequent = more stable targets
    "hidden_dim": 128,     # 128 Unchanged
    
    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 42,
}