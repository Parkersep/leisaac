import carb
import numpy as np
import torch
from typing import Any, Dict

from leisaac.devices.device_base import Device
from ..lerobot.so101_leader import SO101Leader


class XLE_leader(Device):
    """
    A unified teleoperation interface that reads WASD keyboard inputs for an omnidirectional
    planar base and connects to two physical SO101 leader arms for manipulation.
    """

    def __init__(
        self,
        env: Any,
        left_port: str = "/dev/ttyACM0",
        right_port: str = "/dev/ttyACM1",
        recalibrate: bool = False,
    ):
        """
        Initializes the hybrid device, configures speed scaling, and connects to the physical arms.
        """
        # WHY: We call super to hook into the Omniverse carb input system for keyboard events.
        super().__init__(env, "xle-leader")

        # WHY: Defines speed levels so the operator can switch gears for fine-tuning movements in tight spaces.
        self._speed_levels = [
            {"xy_vel": 0.1, "theta_vel": 30 / 180.0 * np.pi},  # Slow
            {"xy_vel": 0.2, "theta_vel": 60 / 180.0 * np.pi},  # Medium
            {"xy_vel": 0.3, "theta_vel": 90 / 180.0 * np.pi},  # Fast
        ]
        self._speed_index = 0

        # WHY: Initializes a 3-element array to perfectly match the 3 root joints (X, Y, Z-rot) in your USD.
        self._vel_command = np.zeros(3)

        # WHY: We query 'left_arm' because that is the alias you used in your scene config to spawn the unified robot.
        self._joint_names = self.env.scene["left_arm"].data.joint_names

        print("Connecting to left SO101 leader...")
        self.left_leader = SO101Leader(
            env, left_port, recalibrate, "left_so101_leader.json"
        )
        print("Connecting to right SO101 leader...")
        self.right_leader = SO101Leader(
            env, right_port, recalibrate, "right_so101_leader.json"
        )

        # WHY: Disables the default keyboard listeners on the child arms so global keys (like 'R') don't fire three times.
        self.left_leader._stop_keyboard_listener()
        self.right_leader._stop_keyboard_listener()

        self._create_key_bindings_for_wheel()

    def _create_key_bindings_for_wheel(self) -> None:
        """
        Creates the mappings from physical keyboard keys to local velocity vectors.
        """
        # WHY: Separates the mathematical intent of the movement from the physical key pressed.
        self.translation = 10
        self.rotation = 5
        self._VEL_COMMAND_MAPPING = {
            "forward": np.asarray([self.translation, 0.0, 0.0]),
            "backward": np.asarray([-self.translation, 0.0, 0.0]),
            "left": np.asarray([0.0, self.translation, 0.0]),
            "right": np.asarray([0.0, -self.translation, 0.0]),
            "rotate_left": np.asarray([0.0, 0.0, self.rotation]),
            "rotate_right": np.asarray([0.0, 0.0, -self.rotation]),
        }

        # WHY: Maps string names to keys for easy remapping if you decide to switch from WASD to Arrow keys later.
        self._WHEEL_INPUT_KEY_MAPPING = {
            "W": "forward",
            "S": "backward",
            "A": "left",
            "D": "right",
            "Z": "rotate_left",
            "X": "rotate_right",
            "KEY_W": "forward",
            "KEY_S": "backward",
            "KEY_A": "left",
            "KEY_D": "right",
            "KEY_Z": "rotate_left",
            "KEY_X": "rotate_right",
        }

    def _add_device_control_description(self) -> None:
        """
        Populates the terminal printout table with keyboard instructions.
        """
        self._display_controls_table.add_row(["W / S", "Drive base forward / backward"])
        self._display_controls_table.add_row(["A / D", "Strafe base left / right"])
        self._display_controls_table.add_row(["Z / X", "Rotate base left / right"])
        self._display_controls_table.add_row(["1, 2, 3", "Change speed level"])

    def _on_keyboard_event(self, event: Any, *args, **kwargs) -> None:
        """
        Listens for key presses, scales the input by the speed profile, and updates the velocity buffer.
        """
        super()._on_keyboard_event(event, *args, **kwargs)

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in ["KEY_1", "KEY_2", "KEY_3"]:
                self._speed_index = int(event.input.name.split("_")[-1]) - 1
                print(f"Speed level changed to: {self._speed_index + 1}")

            if event.input.name in self._WHEEL_INPUT_KEY_MAPPING.keys():
                vel_key = self._WHEEL_INPUT_KEY_MAPPING[event.input.name]
                scale_key = "theta_vel" if "rotate" in vel_key else "xy_vel"

                # WHY: Using '+=' allows the operator to press 'W' and 'A' simultaneously to strafe diagonally.
                self._vel_command += (
                    self._VEL_COMMAND_MAPPING[vel_key]
                    * self._speed_levels[self._speed_index][scale_key]
                )

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # WHY: Acts as a dead-man's switch. Zeroing the buffer instantly stops the robot when the key is released.
            if event.input.name in self._WHEEL_INPUT_KEY_MAPPING.keys():
                vel_key = self._WHEEL_INPUT_KEY_MAPPING[event.input.name]
                scale_key = "theta_vel" if "rotate" in vel_key else "xy_vel"
                self._vel_command -= (
                    self._VEL_COMMAND_MAPPING[vel_key]
                    * self._speed_levels[self._speed_index][scale_key]
                )
                # Ensure we don't have small floating point errors near zero
                if np.all(np.abs(self._vel_command) < 1e-5):
                    self._vel_command[:] = 0.0

    def get_device_state(self) -> Dict[str, np.ndarray]:
        """
        Returns the raw velocity command for the base and packages it with arm states.
        """
        # WHY: The prismatic and revolute joints at the articulation root are
        # intended to move the entire robot body directly relative to its own axes.
        base_action = self._vel_command.copy()

        raw_left = self.left_leader.get_device_state()
        raw_right = self.right_leader.get_device_state()

        return {
            "base_action": base_action,
            "left_arm": raw_left,
            "right_arm": raw_right,
        }

    def input2action(self) -> Dict[str, Any]:
        """
        Adds hardware motor limits required for action processing.
        """
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = {
            "left_arm": self.left_leader.motor_limits,
            "right_arm": self.right_leader.motor_limits,
        }
        return ac_dict

    def reset(self) -> None:
        """
        Clears the velocity buffer and triggers the reset sequence on physical hardware.
        """
        self._vel_command[:] = 0.0
        self._speed_index = 0
        self.left_leader.reset()
        self.right_leader.reset()
        super().reset()

    def __del__(self) -> None:
        """
        Safely disconnects from the physical serial ports.
        """
        # WHY: Disconnecting gracefully prevents the OS from locking the USB ports, which would require a physical replug.
        if hasattr(self, "left_leader") and self.left_leader.is_connected:
            self.left_leader.disconnect()
        if hasattr(self, "right_leader") and self.right_leader.is_connected:
            self.right_leader.disconnect()
        super().__del__()
