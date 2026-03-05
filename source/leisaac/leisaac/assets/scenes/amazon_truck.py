from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Toy Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

amazon_truck_scene_path = "/home/parker/Nvidia/Go2W/source/Go2W/Go2W/tasks/manager_based/go2w/custom_assets/Collected_World_With_Collisions/World_With_Collisions.usd"

amazon_truck_scene = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=amazon_truck_scene_path,
    )
)
