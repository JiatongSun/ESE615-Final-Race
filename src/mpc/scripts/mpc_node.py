#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Int16
import mpc.mpc_class as mpc
import yaml

import numpy as np
import math
import time

from os.path import join
#car_map = 'race3_2'
# Read map name from config file
config_file = "./src/mpc/config/params.yaml"
with open(config_file, 'r') as stream:
    parsed_yaml = yaml.safe_load(stream)
real_test = parsed_yaml["real_test"]
car_map = parsed_yaml["map_name"]
csv_loc = join('src', 'mpc','csv', car_map + '_traj_ltpl_cl.csv')

sleep_time = 1 / 10

class MPC(Node):
    """
    Implement MPC on the car
    """

    def __init__(self):
        super().__init__('mpc_node')

        self.odom = Odometry()
        self.pf = PoseStamped()

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if real_test else "/ego_racecar/odom"
        scan_topic = "/scan"
        drive_topic = "/drive"

        opp_pose_topic = "/opp_racecar/odom"
        opp_drive_topic = "/opp_drive"

        glob_path_topic = "/global_path"
        mpc_path_topic = "/mpc_path"


        # two more marker paths for miscellaneous publishing
        misc_topic_1 = "/misc_path_1"
        misc_topic_2 = "/misc_path_2"

        if real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.scan_sub_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.opp_pose_sub_ = self.create_subscription(Odometry, opp_pose_topic, self.opp_pose_callback, 10)

        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.opp_drive_pub_ = self.create_publisher(AckermannDriveStamped, opp_drive_topic, 10)
        self.glob_path_pub_ = self.create_publisher(Marker, glob_path_topic, 10)
        self.mpc_path_pub_  = self.create_publisher(Marker, mpc_path_topic, 10)
        self.misc_path_pub1_ = self.create_publisher(Marker, misc_topic_1, 10)
        self.misc_path_pub2_ = self.create_publisher(Marker, misc_topic_2, 10)


        # reference trajectory
        waypoints = np.loadtxt(csv_loc, delimiter=';')
        self.cs = waypoints[:, 7].flatten()
        # these two are the centerline
        # so calculate it with alpha later
        self.cenx = waypoints[:, 0].flatten()
        self.ceny = waypoints[:, 1].flatten()
        self.cv = waypoints[:, 10].flatten()
        self.wr = waypoints[:, 2].flatten() # width to the right
        self.wl = waypoints[:, 3].flatten() # width to the left
        self.ux = waypoints[:, 4].flatten() # x component of the norm vector of the ref point
        self.uy = waypoints[:, 5].flatten() # y component of the norm vector of the ref point

        self.calpha = waypoints[:, 6].flatten()
        self.cx = self.cenx # - self.calpha * self.ux # the "reference" line is the raceline?
        self.cy = self.ceny # - self.calpha * self.uy

        self.wall_right_x = self.cenx + self.wr * self.ux - mpc.SAFETY * self.ux
        self.wall_right_y = self.ceny + self.wr * self.uy - mpc.SAFETY * self.uy
        self.wall_left_x = self.cenx - self.wl * self.ux + mpc.SAFETY * self.ux
        self.wall_left_y = self.ceny - self.wl * self.uy + mpc.SAFETY * self.uy

        # pre-calculate the half-plane constraints: THESE WILL NOT CHANGE WITH THE STATE VECTOR
        # fits F*[x, y].T <= f
        self.F = [np.array([[ self.ux[i], self.uy[i]]/self.wr[i], [-self.ux[i], -self.uy[i]]/self.wl[i]]) for i in range(len(self.cs))]
        self.f = [np.array([self.wr[i] - mpc.SAFETY + (self.ux[i] * self.cenx[i] + self.uy[i] * self.ceny[i])/self.wr[i], self.wl[i] - mpc.SAFETY - (self.ux[i] * self.cenx[i] + self.uy[i] * self.ceny[i])/self.wl[i]]) for i in range(len(self.cs))]

        # self.cyaw = waypoints[:,3].flatten()
        # # have to convert the yaws so that 0 is east instead of the north the .csv assumes
        # self.cyaw = self.cyaw - np.pi / 2.0
        # for i in range(len(self.cyaw)):
        #     if self.cyaw[i] < -np.pi:
        #         self.cyaw[i] = self.cyaw[i] + np.pi * 2.0
        # self.cyaw = -self.cyaw

        # manually calculate the yaws and distances between points
        self.cyaw = [None] * len(self.cx)
        arrsiz = len(self.cx)
        for i in range(arrsiz):
            self.cyaw[i] = np.arctan2(self.cy[(i + 1) % arrsiz] - self.cy[i % arrsiz],
                                      self.cx[(i + 1) % arrsiz] - self.cx[i % arrsiz])
        # also calculate the distance traveled for each point
        self.cds = self.cs.copy()
        for i in range(arrsiz - 1):
            self.cds[i] = self.cs[(i + 1) % arrsiz] - self.cs[i]
        self.cds[-1] = np.mean(self.cds[0:-1])

        # MPC variables
        self.target_ind = None # first start near the beginning
        self.dl = 0.5 # course tick
        self.mpc_prob = mpc.MPC()


        # visualization variables
        self.v_min_color = np.min(self.cv)
        self.v_max_color = np.max(self.cv)
        self.num_points = len(waypoints)

        # command horizon predictions
        self.cmd_as = None     # commanded accelerations (m/s^2)
        self.cmd_steers = None # commanded steering angle (rad)
        self.cmd_vs = None     # commanded velocities (m/s)

        # debugging
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.frame_cnt = 0


    def timer_callback(self):
        fps = Int16()
        fps.data = self.frame_cnt
        self.frame_cnt = 0
        self.get_logger().info('mpc fps: %d' % fps.data)

    def odom_callback(self, odom_msg):
        self.odom = odom_msg


    def scan_callback(self, scan_msg):
        # TODO: update state information here for actual race implementation

        pass

    def pose_callback(self, pose_msg):
        if real_test:
            self.pf = pose_msg
            curr_x = self.pf.pose.position.x
            curr_y = self.pf.pose.position.y
            curr_quat = self.pf.pose.orientation
            curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                                  1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))
        else:
            curr_x = self.odom.pose.pose.position.x
            curr_y = self.odom.pose.pose.position.y
            curr_quat = self.odom.pose.pose.orientation
            curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                                  1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))
        curr_vx = self.odom.twist.twist.linear.x
        curr_vy = self.odom.twist.twist.linear.y
        curr_v = np.linalg.norm([curr_vx, curr_vy])

        # add lookahead distance
        running_time = 0.0
        curr_x = curr_x + curr_vx * np.cos(curr_yaw) * running_time
        curr_y = curr_y + curr_vx * np.sin(curr_yaw) * running_time

        self.mpc_state = mpc.State(curr_x, curr_y, curr_yaw, curr_v)


        # TODO: implement MPC

        # variables to command to the vehicle
        cmd_a = 0.0     # m/s^2
        cmd_steer = 0.0 # rad
        cmd_v = 0.0     # m/s


        xref, self.target_ind, dref, Fref, fref = mpc.calc_ref_trajectory(self.mpc_state, self.cds, 
                self.cx, self.cy, self.cyaw, self.cv, self.cmd_vs,
                self.F, self.f, self.target_ind)

        # print(self.target_ind)

        if (self.mpc_state.yaw - xref[3, :] > 1.5 * np.pi).any():
            self.mpc_state.yaw -= 2.0 * np.pi
        elif (self.mpc_state.yaw - xref[3, :] < -1.5 * np.pi).any():
            self.mpc_state.yaw += 2.0 * np.pi
        x0 = [self.mpc_state.x, self.mpc_state.y, self.mpc_state.v, self.mpc_state.yaw]
        # store horizon desires as arrays, too, for visualization
        self.cmd_as, self.cmd_steers, cmd_xs, cmd_ys, cmd_yaws, self.cmd_vs = mpc.iterative_linear_mpc_control(
                self.mpc_prob, xref, x0, dref, Fref, fref, self.cmd_as, self.cmd_steers)

        if self.cmd_steers is not None:
            cmd_steer, cmd_a, cmd_v = self.cmd_steers[0], self.cmd_as[0], self.cmd_vs[1]

        # self.mpc_state = mpc.update_state(self.mpc_state, cmd_a, cmd_steer)
        # print('a:\t%.3f' % cmd_a)
        # print('d:\t%.3f' % cmd_steer)
        # print('v:\t%.3f' % cmd_v)

        # print('vel_err:\t%+.3f' % (curr_v - cmd_v))

        # print('des_yaw:\t%.3f' % self.cyaw[self.target_ind])
        # print('cmd_yaw:\t%.3f' % cmd_yaws[0])
        # print('cur_yaw:\t%.3f' % self.mpc_state.yaw)

        # print('\n')

        message_out = AckermannDriveStamped()
        message_out.drive.acceleration = cmd_a
        message_out.drive.speed = cmd_v
        message_out.drive.steering_angle = cmd_steer
        self.drive_pub_.publish(message_out)

        # also handle visualization
        mpc_waypoints = self.visualize_waypoints(cmd_xs, cmd_ys, self.cmd_vs, 0, 'mpc', 7)
        self.mpc_path_pub_.publish(mpc_waypoints)

        # visualize the whole path
        #glob_waypoints = self.visualize_waypoints(self.cx, self.cy, self.cv, 1, 'global', 4)
        # visualize just the reference trajectory
        glob_waypoints = self.visualize_waypoints(xref[0], xref[1], xref[2], 1, 'global', 4)
        self.glob_path_pub_.publish(glob_waypoints)


        # visualize miscellaneous waypoints
        misc_waypoints_1 = self.visualize_waypoints(self.wall_right_x, self.wall_right_y, self.cv, 2, 'global', 4)
        self.misc_path_pub1_.publish(misc_waypoints_1)
        misc_waypoints_2 = self.visualize_waypoints(self.wall_left_x, self.wall_left_y, self.cv, 2, 'global', 4)
        self.misc_path_pub2_.publish(misc_waypoints_2)

        self.frame_cnt += 1

        time.sleep(sleep_time)
        pass

    def opp_pose_callback(self, opp_pose_msg):
        # TODO: update opponent state info
        pass

    def visualize_waypoints(self, x, y, vel, idnum, idname, mark_type):
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = idnum
        marker.ns = idname
        marker.type = mark_type
        marker.action = 0
        marker.points = []
        marker.colors = []
        for i in range(len(x)):
            this_point = Point()
            this_point.x = x[i]
            this_point.y = y[i]
            marker.points.append(this_point)

            this_color = ColorRGBA()
            speed_ratio = (vel[i] - self.v_min_color) / (self.v_max_color - self.v_min_color)
            this_color.a = 1.0
            this_color.r = (1 - speed_ratio)
            this_color.g = speed_ratio
            marker.colors.append(this_color)

        this_scale = 0.1
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        return marker





def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
