import gymnasium as gym

gym.register(
    id="LeIsaac-XLeRobot-AmazonTruck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Update the class name here at the end of the string
        "env_cfg_entry_point": f"{__name__}.amazon_truck_env_cfg:AmazonTruckBiArmEnvCfg",
    },
)
