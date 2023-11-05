def minecraft_nav_config():
    return {
        "env": "MinecraftNav",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 500,
        "epochs": 4,
        "n_workers": 1,
        "worker_steps": 400,
        "n_mini_batch": 4,
        "value_loss_coefficient": 0.2,
        "hidden_layer_size": 128,
        "recurrence": 
            {
            "sequence_length": 8,
            "hidden_state_size": 64,
            "layer_type": "lstm",
            "reset_hidden_state": True
            },
        "learning_rate_schedule":
            {
            "initial": 3.0e-4,
            "final": 3.0e-6,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 1000
            }
    }
