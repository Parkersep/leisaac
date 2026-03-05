"""Sandbox Configuration for the Dual-Arm XLeRobot in the Amazon Truck."""

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import JointPositionActionCfg

from leisaac.assets.scenes.amazon_truck import amazon_truck_scene
from leisaac.assets.robots.lerobot import XLEROBOT_CFG
from leisaac.utils.constant import BI_ARM_JOINT_NAMES

from leisaac.tasks.template.bi_arm_env_cfg import (
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmObservationsCfg,
    BiArmTerminationsCfg,
)


@configclass
class AmazonTruckBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration containing only the truck and the robot."""

    scene: AssetBaseCfg = amazon_truck_scene.replace(prim_path="{ENV_REGEX_NS}/Scene")

    def __post_init__(self):
        super().__post_init__()

        # Spawn the unified robot into the 'left_arm' slot to ensure it loads before cameras
        self.left_arm = XLEROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=XLEROBOT_CFG.init_state.replace(pos=(8.1020, 3.00, -1.6080)),
        )
        self.right_arm = None

        # Point cameras to the native USD links using /isaac_camera to avoid collisions
        self.left_wrist = self.left_wrist.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Fixed_Jaw_2/Left_Arm_Camera/isaac_camera"
        )
        self.left_wrist.offset.pos = (0.0, 0.0, 0.0)
        self.left_wrist.offset.rot = (1.0, 0.0, 0.0, 0.0)

        self.right_wrist = self.right_wrist.replace(
            prim_path="{ENV_REGEX_NS}/Robot/Fixed_Jaw/Right_Arm_Camera/isaac_camera"
        )
        self.right_wrist.offset.pos = (0.0, 0.0, 0.0)
        self.right_wrist.offset.rot = (1.0, 0.0, 0.0, 0.0)

        self.top = self.top.replace(
            prim_path="{ENV_REGEX_NS}/Robot/head_tilt_link/head_camera_link/isaac_camera"
        )
        self.top.offset.pos = (0.0, 0.0, 0.0)
        self.top.offset.rot = (1.0, 0.0, 0.0, 0.0)


@configclass
class AmazonTruckBiArmEnvCfg(BiArmTaskEnvCfg):
    """Sandbox environment for dual-arm teleoperation testing."""

    scene: AmazonTruckBiArmSceneCfg = AmazonTruckBiArmSceneCfg(env_spacing=8.0)
    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    # We leave terminations empty/default since we aren't tracking success conditions yet
    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    task_description: str = "Sandbox mode for testing dual-arm XLeRobot teleoperation."

    def __post_init__(self) -> None:
        super(BiArmTaskEnvCfg, self).__post_init__()

        # Disable the hardcoded gripper reset helper since we use a unified robot
        self.dynamic_reset_gripper_effort_limit = False

        # Restore necessary physics settings
        self.decimation = 1
        self.episode_length_s = 25.0
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True
        self.default_feature_joint_names = [
            f"{joint_name}.pos" for joint_name in BI_ARM_JOINT_NAMES
        ]

        self.viewer.eye = (2.5, -1.0, 1.3)
        self.viewer.lookat = (3.6, -0.4, 1.0)

        # Map actions to the specific joint names of the unified robot (held in "left_arm")
        self.actions.left_arm_action = JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.left_gripper_action = JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["Jaw"],
            scale=1.0,
            use_default_offset=True,
        )

        self.actions.right_arm_action = JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=[
                "Rotation_2",
                "Pitch_2",
                "Elbow_2",
                "Wrist_Pitch_2",
                "Wrist_Roll_2",
            ],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.right_gripper_action = JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["Jaw_2"],
            scale=1.0,
            use_default_offset=True,
        )

        # Redirect observations to the unified robot
        left_joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        right_joints = [
            "Rotation_2",
            "Pitch_2",
            "Elbow_2",
            "Wrist_Pitch_2",
            "Wrist_Roll_2",
            "Jaw_2",
        ]

        for obs_name in [
            "left_joint_pos",
            "left_joint_vel",
            "left_joint_pos_rel",
            "left_joint_vel_rel",
            "left_joint_pos_target",
        ]:
            getattr(self.observations.policy, obs_name).params["asset_cfg"] = (
                SceneEntityCfg("left_arm", joint_names=left_joints)
            )

        for obs_name in [
            "right_joint_pos",
            "right_joint_vel",
            "right_joint_pos_rel",
            "right_joint_vel_rel",
            "right_joint_pos_target",
        ]:
            getattr(self.observations.policy, obs_name).params["asset_cfg"] = (
                SceneEntityCfg("left_arm", joint_names=right_joints)
            )

    def use_teleop_device(self, teleop_device) -> None:
        """Overrides the template method to prevent init_action_cfg from overwriting our custom joint names."""
        self.task_type = teleop_device
