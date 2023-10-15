#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union
import scipy.spatial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, Pose, PoseArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

"""
Constant Definition
"""
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


class LaneFollow(Node):
    """
    Class for lane follow
    """

    def __init__(self):
        super().__init__("lane_follow_node")

        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("obs_dist_thresh")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")

        self.declare_parameter("lookahead_distance")
        self.declare_parameter("lookahead_attenuation")
        self.declare_parameter("lookahead_idx")
        self.declare_parameter("lookbehind_idx")

        self.declare_parameter("kp_steer")
        self.declare_parameter("ki_steer")
        self.declare_parameter("kd_steer")
        self.declare_parameter("max_steer")
        self.declare_parameter("alpha_steer")

        self.declare_parameter("kp_speed")
        self.declare_parameter("ki_speed")
        self.declare_parameter("kd_speed")
        self.declare_parameter("max_speed")
        self.declare_parameter("alpha_speed")

        self.declare_parameter("kp_pos")
        self.declare_parameter("ki_pos")
        self.declare_parameter("kd_pos")

        self.declare_parameter("follow_speed")
        self.declare_parameter("lane_dist_thresh")

        # PID Control Params
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0
        self.prev_steer = 0.0

        self.prev_speed_error = 0.0
        self.speed_integral = 0.0
        self.prev_speed = 0.0

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

        # Car Status Variables
        self.lane_idx = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None

        # Obstacle Variables
        self.obstacles = None
        self.opponent = None
        self.lane_free = [True] * self.num_lanes

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if self.real_test else "/ego_racecar/odom"
        obstacle_topic = "/opp_predict/bbox"
        opponent_topic = "/opp_predict/pose"
        drive_topic = "/drive"
        waypoint_topic = "/waypoint"

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.obstacle_sub_ = self.create_subscription(PoseArray, obstacle_topic, self.obstacle_callback, 1)
        self.opponent_sub_ = self.create_subscription(Pose, opponent_topic, self.opponent_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)

    def odom_callback(self, odom_msg: Odometry):
        self.curr_vel = odom_msg.twist.twist.linear.x

    def obstacle_callback(self, obstacle_msg: PoseArray):
        obstacle_list = []
        for obstacle in obstacle_msg.poses:
            x = obstacle.position.x
            y = obstacle.position.y
            obstacle_list.append([x, y])
        self.obstacles = np.array(obstacle_list) if obstacle_list else None

        if self.obstacles is None:
            return

        obs_dist_thresh = self.get_parameter("obs_dist_thresh").get_parameter_value().double_value
        for i in range(self.num_lanes):
            d = scipy.spatial.distance.cdist(self.waypoint_pos[i], self.obstacles)
            self.lane_free[i] = (np.min(d) > obs_dist_thresh)

    def opponent_callback(self, opponent_msg: Pose):
        opponent_x = opponent_msg.position.x
        opponent_y = opponent_msg.position.y
        self.opponent = np.array([opponent_x, opponent_y])

    def pose_callback(self, pose_msg: Union[PoseStamped, Odometry]):
        """
        The pose callback when subscribed to particle filter"s inferred pose
        Here is where the main RRT loop happens

        Args:
            pose_msg (PoseStamped / Odometry): incoming message from subscribed topic
        Returns:

        """

        # Read pose data
        if self.real_test:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.pose.orientation

        curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                              1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        # Find index of the current point
        distances = []
        dist_to_lanes = []
        for i in range(self.num_lanes):
            distance = np.linalg.norm(self.waypoint_pos[i] - curr_pos, axis=1)
            distances.append(distance)
            dist_to_lane = np.min(distance)
            dist_to_lanes.append(dist_to_lane)

        # Select lane based on obstacle info
        # Only switch lane when current lane is occupied but another lane is free
        lane_available = np.array(self.lane_free).any()
        if not self.lane_free[self.lane_idx] and lane_available:
            self.lane_idx = np.argmax(self.lane_free)

        distance = distances[self.lane_idx]
        self.curr_idx = np.argmin(distance)

        lane_dist_thresh = self.get_parameter("lane_dist_thresh").get_parameter_value().double_value
        on_lane = np.min(distance) < lane_dist_thresh

        num_pts = self.num_pts[self.lane_idx]
        waypoint_pos = self.waypoint_pos[self.lane_idx]
        waypoint_v = self.waypoint_v[self.lane_idx]

        # Find the closest point on current lane
        # Calculate distance to each point on the lane
        closest_point = waypoint_pos[self.curr_idx]
        distance = np.linalg.norm(waypoint_pos - closest_point, axis=1)

        # Get lookahead distance
        L = self.get_lookahead_dist(self.curr_idx)

        # Binary search goal waypoint to track
        self.goal_idx = self.curr_idx
        while distance[self.goal_idx] <= L:
            self.goal_idx = (self.goal_idx + 1) % num_pts

        left = waypoint_pos[(self.goal_idx - 1) % num_pts, :]
        right = waypoint_pos[self.goal_idx % num_pts, :]

        while True:
            mid = (left + right) / 2
            dist = np.linalg.norm(mid - closest_point)
            if abs(dist - L) < 1e-2:
                self.target_point = mid
                break
            elif dist > L:
                right = mid
            else:
                left = mid

        # Transform goal point to vehicle frame of reference
        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])

        # Get desired speed and steering angle
        gamma = 2 / L ** 2
        error = gamma * target_y
        steer = self.get_steer(error)

        target_speed = waypoint_v[self.curr_idx % num_pts]
        if not on_lane:  # ego car too far from any lane, needs to merge into a lane
            error = np.linalg.norm(self.target_point - curr_pos)
            speed = self.get_speed(error, use_pos=True)
        elif not lane_available:  # all lanes occupied by opponent, follow opponent until any lane clear
            speed = self.get_parameter("follow_speed").get_parameter_value().double_value
        else:  # keep up with desired speed
            error = target_speed - self.curr_vel
            speed = self.get_speed(error, use_pos=False)

        print("vel: %.2f\t tar_vel: %.2f\t cmd_vel: %.2f\t steer: %.2f" % (self.curr_vel,
                                                                           target_speed,
                                                                           speed,
                                                                           np.rad2deg(steer)))

        # Publish drive message
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)

        # Visualize waypoints
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        if visualize:
            self.visualize_target()

        return None

    def get_lookahead_dist(self, curr_idx):
        """
        This method should calculate the lookahead distance based on past and future waypoints

        Args:
            curr_idx (ndarray[int]): closest waypoint index
        Returns:
            lookahead_dist (float): lookahead distance

        """
        L = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        lookahead_idx = self.get_parameter("lookahead_idx").get_parameter_value().integer_value
        lookbehind_idx = self.get_parameter("lookbehind_idx").get_parameter_value().integer_value
        slope = self.get_parameter("lookahead_attenuation").get_parameter_value().double_value

        num_pts = self.num_pts[self.lane_idx]
        waypoint_yaw = self.waypoint_yaw[self.lane_idx]

        yaw_before = waypoint_yaw[(curr_idx - lookbehind_idx) % num_pts]
        yaw_after = waypoint_yaw[(curr_idx + lookahead_idx) % num_pts]
        yaw_diff = abs(yaw_after - yaw_before)
        if yaw_diff > np.pi:
            yaw_diff = yaw_diff - 2 * np.pi
        if yaw_diff < -np.pi:
            yaw_diff = yaw_diff + 2 * np.pi
        yaw_diff = abs(yaw_diff)
        if yaw_diff > np.pi / 2:
            yaw_diff = np.pi / 2
        L = max(0.5, L * (np.pi / 2 - yaw_diff * slope) / (np.pi / 2))

        return L

    def get_steer(self, error):
        """ Get desired steering angle by PID
        """
        kp = self.get_parameter("kp_steer").get_parameter_value().double_value
        ki = self.get_parameter("ki_steer").get_parameter_value().double_value
        kd = self.get_parameter("kd_steer").get_parameter_value().double_value
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        alpha = self.get_parameter("alpha_steer").get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        self.prev_steer_error = error
        self.steer_integral += error
        steer = kp * error + ki * self.steer_integral + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        new_steer = alpha * new_steer + (1 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer

    def get_speed(self, error, use_pos=False):
        """ Get desired speed by PID
        """
        if use_pos:
            kp = self.get_parameter("kp_pos").get_parameter_value().double_value
            ki = self.get_parameter("ki_pos").get_parameter_value().double_value
            kd = self.get_parameter("kd_pos").get_parameter_value().double_value
        else:
            kp = self.get_parameter("kp_speed").get_parameter_value().double_value
            ki = self.get_parameter("ki_speed").get_parameter_value().double_value
            kd = self.get_parameter("kd_speed").get_parameter_value().double_value
        max_control = self.get_parameter("max_speed").get_parameter_value().double_value
        alpha = self.get_parameter("alpha_speed").get_parameter_value().double_value

        d_error = error - self.prev_speed_error
        self.prev_speed_error = error
        self.speed_integral += error
        speed = kp * error + ki * self.speed_integral + kd * d_error
        new_speed = np.clip(speed, 0, max_control)
        new_speed = alpha * new_speed + (1 - alpha) * self.prev_steer
        self.prev_speed = new_speed

        return new_speed

    def visualize_target(self):
        # Publish target waypoint
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "target_waypoint"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.target_point[0]
        marker.pose.position.y = self.target_point[1]

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        marker.lifetime.nanosec = int(1e8)

        self.waypoint_pub_.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("Lane Follow Initialized")
    lane_follow_node = LaneFollow()
    rclpy.spin(lane_follow_node)

    lane_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
