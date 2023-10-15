#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Int16

"""
Constant Definition
"""
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


class LaneVisualize(Node):
    """
    Class for lane visualization
    """

    def __init__(self):
        super().__init__("lane_visualize_node")

        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")

        # Global Map Params
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value
        self.num_lanes = self.get_parameter("num_lanes").get_parameter_value().integer_value
        self.lane_files = self.get_parameter("lane_files").get_parameter_value().string_array_value

        self.num_pts = []
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_pos = []
        self.waypoint_v = []
        self.waypoint_yaw = []
        self.waypoint_curv = []

        assert len(self.lane_files) == self.num_lanes

        for i in range(self.num_lanes):
            csv_loc = os.path.join("src", "lane_follow", "csv", self.lane_files[i] + ".csv")
            waypoints = np.loadtxt(csv_loc, delimiter=";", skiprows=1)
            self.num_pts.append(len(waypoints))
            self.waypoint_x.append(waypoints[:, 1])
            self.waypoint_y.append(waypoints[:, 2])
            self.waypoint_pos.append(waypoints[:, 1:3])
            self.waypoint_yaw.append(waypoints[:, 3])
            self.waypoint_curv.append(waypoints[:, 4])
            self.waypoint_v.append(waypoints[:, 5])

        # Topics & Subs, Pubs
        self.timer = self.create_timer(1.0, self.timer_callback)

        path_topic = []
        self.path_pub_ = []
        for i in range(self.num_lanes):
            path_topic.append("/global_path/lane_" + str(i))
            self.path_pub_.append(self.create_publisher(Marker, path_topic[i], 10))

    def timer_callback(self):
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        if visualize:
            self.visualize_global_path()

    def visualize_global_path(self):
        # Publish all waypoints
        for lane_idx in range(self.num_lanes):
            num_pts = self.num_pts[lane_idx]
            target_x = self.waypoint_x[lane_idx]
            target_y = self.waypoint_y[lane_idx]
            target_v = self.waypoint_v[lane_idx]
            v_min = np.min(target_v)
            v_max = np.max(target_v)

            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = 0
            marker.ns = "global_planner"
            marker.type = 4
            marker.action = 0
            marker.points = []
            marker.colors = []
            for i in range(num_pts + 1):
                this_point = Point()
                this_point.x = target_x[i % num_pts]
                this_point.y = target_y[i % num_pts]
                marker.points.append(this_point)

                this_color = ColorRGBA()
                speed_ratio = (target_v[i % num_pts] - v_min) / (v_max - v_min)
                this_color.a = 1.0
                this_color.r = (1 - speed_ratio)
                this_color.g = speed_ratio
                marker.colors.append(this_color)

            this_scale = 0.1
            marker.scale.x = this_scale
            marker.scale.y = this_scale
            marker.scale.z = this_scale

            marker.pose.orientation.w = 1.0

            self.path_pub_[lane_idx].publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("Lane Visualize Initialized")
    lane_visualize_node = LaneVisualize()
    rclpy.spin(lane_visualize_node)

    lane_visualize_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
