#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)
MAX_SPEED = 12.0  # (m/s)


def generate_launch_description():
    ld = LaunchDescription()
    print(os.getcwd())
    config = os.path.join(
        'src',
        'lane_follow',
        'config',
        'params.yaml'
    )
    config_dict = yaml.safe_load(open(config, 'r'))
    lane_follow_node = Node(
        package="lane_follow",
        executable="lane_follow_node.py",
        name="lane_follow_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            # YAML Params
            config_dict,

            # RVIZ Params
            {"visualize": True},

            # Obstacle Params
            {"obs_dist_thresh": WIDTH},

            # Pure Pursuit Params
            {"lookahead_distance": 1.5},
            {"lookahead_attenuation": 0.6},
            {"lookahead_idx": 40},
            {"lookbehind_idx": 0},

            # PID Control Params
            {"kp_steer": 0.3},
            {"ki_steer": 0.0},
            {"kd_steer": 5.0},
            {"max_steer": MAX_STEER},
            {"alpha_steer": 1.0},

            {"kp_speed": 2.5},
            {"ki_speed": 0.0},
            {"kd_speed": 3.0},
            {"max_speed": MAX_SPEED},
            {"alpha_speed": 1.0},

            {"kp_pos": 0.5},
            {"ki_pos": 0.0},
            {"kd_pos": 0.0},

            # Speed Params
            {"follow_speed": 2.0},
            {"lane_dist_thresh": 1.0},
        ]
    )
    ld.add_action(lane_follow_node)

    lane_visualize_node = Node(
        package="lane_follow",
        executable="lane_visualize.py",
        name="lane_visualize",
        output="screen",
        emulate_tty=True,
        parameters=[
            # YAML Params
            config_dict,

            # RVIZ Params
            {"visualize": True},
        ]
    )
    ld.add_action(lane_visualize_node)

    return ld
