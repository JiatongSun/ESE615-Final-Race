#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


def generate_launch_description():
    ld = LaunchDescription()
    print(os.getcwd())
    config = os.path.join(
        'src',
        'opponent_predictor',
        'config',
        'params.yaml'
    )
    config_dict = yaml.safe_load(open(config, 'r'))
    opponent_predictor_node = Node(
        package="opponent_predictor",
        executable="opponent_predictor_node.py",
        name="opponent_predictor_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            # YAML Params
            config_dict,

            # RVIZ Params
            {"visualize": True},
            {"visualize_grid": False},
            {"visualize_obstacle": False},
            {"visualize_opp_pose": True},
            {"visualize_opp_bbox": True},

            # Grid Params
            {"grid_xmin": 0.0},
            {"grid_xmax": 8.0},
            {"grid_ymin": -2.5},
            {"grid_ymax": 2.5},
            {"grid_resolution": 0.05},
            {"plot_resolution": 0.25},
            {"grid_safe_dist": 0.1},

            # Wall Params
            {"wall_safe_dist": 0.17},

            # Obstacle Params
            {"cluster_dist_tol": WIDTH + 2 * WHEEL_LENGTH},
            {"cluster_size_tol": 15},
        ]
    )
    ld.add_action(opponent_predictor_node)

    return ld
