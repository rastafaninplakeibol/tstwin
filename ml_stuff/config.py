config = {
    "grid_width": 100,
    "grid_height": 100,
    "initial_energy": 100.0,
    "max_energy": 100.0,
    "energy_recovery": 0.5,   # energy recovered when walking
    "sprint_cost": 2.0,       # energy cost for sprinting
    "walk_distance": 1,       # cells moved when walking
    "sprint_distance": 2,     # cells moved when sprinting
    "max_steps": 100,         # max turns per episode
    "reward_reach_target": 10.0,
    "reward_distance_factor": -0.1,  # negative reward proportional to distance to target
    "negative_reward_no_energy": -2,  # negative reward proportional to distance to target

    # Actions: each action is ("move_type", "direction")
    # move_type in {walk, sprint}, direction in {up, down, left, right}
    "actions": [
        ("walk", "up"),
        ("walk", "down"),
        ("walk", "left"),
        ("walk", "right"),
        ("sprint", "up"),
        ("sprint", "down"),
        ("sprint", "left"),
        ("sprint", "right")
    ]
}
