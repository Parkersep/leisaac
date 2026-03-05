"""Configuration for the Dual-Arm XLeRobot in the Amazon Truck."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import JointPositionActionCfg

# Device hardware calibration helper
from leisaac.devices.action_process import init_action_cfg

# Importing scene and robot configs
from leisaac.assets.scenes.amazon_truck import amazon_truck_scene
from leisaac.assets.robots.lerobot import XLEROBOT_CFG
from leisaac.utils.constant import BI_ARM_JOINT_NAMES

# Inside your main loop

# Importing base templates
from leisaac.tasks.template.bi_arm_env_cfg import (
    BiArmTaskEnvCfg,
    BiArmTaskSceneCfg,
    BiArmObservationsCfg,
    BiArmTerminationsCfg,
)


@configclass
class AmazonTruckBiArmSceneCfg(BiArmTaskSceneCfg):
    """
    Scene configuration containing the truck and the unified XLeRobot.
    """

    # Replace the default scene with the Amazon Truck USD
    scene: AssetBaseCfg = amazon_truck_scene.replace(prim_path="{ENV_REGEX_NS}/Scene")

    def __post_init__(self):
        """
        Initializes the scene by placing the robot and zeroing out camera offsets.
        """
        super().__post_init__()

        # We spawn the unified robot into the 'left_arm' slot so it loads early in the scene graph.
        # This prevents the cameras from crashing when they try to attach to links that don't exist yet.
        # We use the extracted global coordinates with a small Z-buffer (+0.05) so the robot drops onto the floor.
        self.left_arm = XLEROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=XLEROBOT_CFG.init_state.replace(
                pos=(2, 3.5, -1), rot=(0.7071, 0.0, 0.0, 0.7071)
            ),
        )

        # We disable the right_arm slot because both arms are contained in our single unified asset.
        self.right_arm = None

        # Attaching sensors to native USD links via child prims (/isaac_camera) to avoid naming collisions.
        # We explicitly zero out the offsets because the links in the custom USD are already perfectly placed.
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
    """
    Environment configuration for teleoperating the dual-arm XLeRobot.
    """

    scene: AmazonTruckBiArmSceneCfg = AmazonTruckBiArmSceneCfg(env_spacing=8.0)
    observations: BiArmObservationsCfg = BiArmObservationsCfg()
    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    task_description: str = "Sandbox mode for testing dual-arm XLeRobot teleoperation."

    def __post_init__(self) -> None:
        """
        Sets up physics, actions, and observation redirection for the unified robot.
        """
        # Call the grandparent init to avoid the BiArmTaskEnvCfg default setup which crashes on unified assets.
        super(BiArmTaskEnvCfg, self).__post_init__()

        # Disable dynamic gripper reset because the helper script expects two separate robot assets.
        self.dynamic_reset_gripper_effort_limit = False

        # Restore necessary physics and rendering settings for stability and visual fidelity.
        self.decimation = 1
        self.episode_length_s = 25.0
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True
        self.default_feature_joint_names = [f"{j}.pos" for j in BI_ARM_JOINT_NAMES]

        # Camera view for the simulation viewer focusing on the truck interior
        self.viewer.eye = (11.0, 1.0, 1.0)
        self.viewer.lookat = (8.1, 3.0, -1.0)

        # Map base actions to the unified robot asset.
        # Note: We configure standard settings here because the `init_action_cfg` helper
        # will automatically inject the correct hardware offsets/scales during runtime.
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

        # Redirect observation terms to look at the correct joint groups on the unified robot asset.
        left_joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        right_joints = [
            "Rotation_2",
            "Pitch_2",
            "Elbow_2",
            "Wrist_Pitch_2",
            "Wrist_Roll_2",
            "Jaw_2",
        ]

        for obs in [
            "left_joint_pos",
            "left_joint_vel",
            "left_joint_pos_rel",
            "left_joint_vel_rel",
            "left_joint_pos_target",
        ]:
            getattr(self.observations.policy, obs).params["asset_cfg"] = SceneEntityCfg(
                "left_arm", joint_names=left_joints
            )

        for obs in [
            "right_joint_pos",
            "right_joint_vel",
            "right_joint_pos_rel",
            "right_joint_vel_rel",
            "right_joint_pos_target",
        ]:
            getattr(self.observations.policy, obs).params["asset_cfg"] = SceneEntityCfg(
                "left_arm", joint_names=right_joints
            )

    def use_teleop_device(self, teleop_device) -> None:
        """
        Applies hardware calibration offsets from the LeIsaac library, then surgically
        patches the action mappings to target the unified robot and inverts backward joints.
        """
        self.task_type = teleop_device

        # 1. RUN THE HELPER: This injects the proper offset math so the robot starts in the correct base posture.
        self.actions = init_action_cfg(self.actions, device=teleop_device)

        # --- Left Arm Patch ---
        self.actions.left_arm_action.asset_name = "left_arm"
        self.actions.left_arm_action.joint_names = [
            "Rotation",
            "Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]

        # We override the scales here to fix the movement directions based on your testing.
        # -1.0 means the joint will move in the opposite direction of the leader arm command.
        self.actions.left_arm_action.scale = {
            "Rotation": 1.0,  # Shoulder pan is correct
            "Pitch": -1.0,  # Flipping inverted shoulder motion
            "Elbow": 1.0,  # Flipping inverted elbow motion
            "Wrist_Pitch": 1.0,  # Wrist pitch is correct
            "Wrist_Roll": -1.0,  # Flipping inverted wrist spin
        }

        self.actions.left_gripper_action.asset_name = "left_arm"
        self.actions.left_gripper_action.joint_names = ["Jaw"]

        # --- Right Arm Patch ---
        self.actions.right_arm_action.asset_name = "left_arm"
        self.actions.right_arm_action.joint_names = [
            "Rotation_2",
            "Pitch_2",
            "Elbow_2",
            "Wrist_Pitch_2",
            "Wrist_Roll_2",
        ]

        # We apply the exact same directional flips to the right arm.
        self.actions.right_arm_action.scale = {
            "Rotation_2": 1.0,
            "Pitch_2": -1.0,
            "Elbow_2": 1.0,
            "Wrist_Pitch_2": 1.0,
            "Wrist_Roll_2": -1.0,
        }

        self.actions.right_gripper_action.asset_name = "left_arm"
        self.actions.right_gripper_action.joint_names = ["Jaw_2"]
