import carb
import numpy as np
import torch
from leisaac.devices.keyboard import SO101Keyboard


class XLeRobotKeyboard(SO101Keyboard):
    """
    Keyboard controller for the unified XLeRobot base.
    Controls root_x_axis_joint, root_y_axis_joint, and root_z_rotation_joint.
    """

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, sensitivity=sensitivity)
        self.device_type = "xlerobot-keyboard"

        # Speed levels adjusted for the heavier XLeRobot model
        self._speed_levels = [
            {"xy_vel": 0.5, "theta_vel": 1.0},  # slow
            {"xy_vel": 1.5, "theta_vel": 2.0},  # medium
            {"xy_vel": 3.0, "theta_vel": 4.0},  # fast
        ]
        self._speed_index = 0

        self._create_key_bindings_for_base()
        self._vel_command = np.zeros(3)
        self._joint_names = self.env.scene["robot"].data.joint_names

    def get_device_state(self):
        """Returns the concatenated arm and base actions."""
        # Get the arm actions from the parent SO101Keyboard
        arm_action = super().get_device_state()

        # Scale velocity based on selected speed level
        current_speed = self._speed_levels[self._speed_index]
        scaled_vel = np.array(
            [
                self._vel_command[0] * current_speed["xy_vel"],
                self._vel_command[1] * current_speed["xy_vel"],
                self._vel_command[2] * current_speed["theta_vel"],
            ]
        )

        # For a unified base in Isaac Lab, we return the raw velocity targets
        # for the root joints (x, y, and theta).
        return np.concatenate([arm_action, scaled_vel])

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._BASE_KEY_MAPPING:
                self._vel_command += self._BASE_VEL_MAPPING[
                    self._BASE_KEY_MAPPING[event.input.name]
                ]

            # Speed level switching (1, 2, 3)
            if event.input.name in ["KEY_1", "KEY_2", "KEY_3"]:
                self._speed_index = int(event.input.name[-1]) - 1

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._BASE_KEY_MAPPING:
                self._vel_command[:] = 0.0

    def _create_key_bindings_for_base(self):
        self._BASE_VEL_MAPPING = {
            "forward": np.array([1.0, 0.0, 0.0]),
            "backward": np.array([-1.0, 0.0, 0.0]),
            "left": np.array([0.0, 1.0, 0.0]),
            "right": np.array([0.0, -1.0, 0.0]),
            "rot_l": np.array([0.0, 0.0, 1.0]),
            "rot_r": np.array([0.0, 0.0, -1.0]),
        }
        self._BASE_KEY_MAPPING = {
            "UP": "forward",
            "DOWN": "backward",
            "LEFT": "left",
            "RIGHT": "right",
            "Z": "rot_l",
            "X": "rot_r",
        }
