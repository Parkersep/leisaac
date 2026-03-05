"""Task configuration for teleoperating the XLeRobot in the Amazon Truck scene."""

from isaaclab.utils import configclass
from leisaac.tasks.pick_orange.pick_orange_env_cfg import PickOrangeEnvCfg
from leisaac.utils.registry import register_env

# Import your custom assets
# These imports assume the files are in the leisaac package path
from leisaac.assets.scenes.amazon_truck import amazon_truck_scene
from leisaac.assets.robots.lerobot import XLEROBOT_CFG


@configclass
class AmazonTruckXLeRobotEnvCfg(PickOrangeEnvCfg):
    """Configuration for the XLeRobot inside the Amazon Truck world."""

    def __post_init__(self):
        """Initializes the task and overrides the default assets."""
        super().__post_init__()

        # 1. Load the Amazon Truck Scene
        # We replace the default kitchen scene with your 'World_With_Collisions.usd'
        self.scene.world = amazon_truck_scene.replace(prim_path="{ENV_REGEX_NS}/World")

        # 2. Load the XLeRobot
        # We swap the default SO101 for the XLeRobot and set a starting position
        self.scene.robot = XLEROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=XLEROBOT_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.5),  # Adjust z to ensure it's on the floor
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )


# Register the environment so the teleop script can find it
register_env(
    "LeIsaac-XLeRobot-AmazonTruck-v0",
    entry_point=AmazonTruckXLeRobotEnvCfg,
)
