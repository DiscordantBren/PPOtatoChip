"""
Predefined experiment recipes for hyperparameter sweeps.
"""

PREDEFINED_RECIPES = [
    {
        "name": "Learning Rate Sweep",
        "param": "lr",
        "values": [0.0001, 0.0003, 0.001],
        "description": "Compare convergence behavior across learning rates",
    },
    {
        "name": "Encoder Capacity Sweep",
        "param": "hidden_channels_e",
        "values": [64, 128, 256],
        "description": "Test how GNN encoder size affects placement quality",
    },
    {
        "name": "Encoder Ablation (Freeze vs Fine-tune vs Scratch)",
        "param": "freeze_encoder",
        "values": [True, False],
        "description": "Compare pretrained encoder frozen, fine-tuned, and no pretraining",
    },
    {
        "name": "Grid Resolution Sweep",
        "param": "num_rows",
        "values": [64, 128, 256],
        "description": "Test how grid granularity affects placement quality",
    },
    {
        "name": "PPO Clip Epsilon Sweep",
        "param": "clip_epsilon",
        "values": [0.1, 0.2, 0.3],
        "description": "Test PPO stability across clip epsilon values",
    },
    {
        "name": "Entropy Coefficient Sweep",
        "param": "entropy_coef",
        "values": [0.01, 0.1, 0.8],
        "description": "Test exploration vs exploitation balance",
    },
]