#!/usr/bin/env python
import numpy as np
from numpy import linalg as la
import os
import pandas as pd
import math

import roslib
import rospy
import rospkg

from std_msgs.msg import Int16
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from tf.transformations import euler_from_quaternion

sin = math.sin
cos = math.cos
atan2 = math.atan2
sqrt = math.sqrt

# Calculate distance
def calc_dist(tx, ty, ix, iy):
    return math.sqrt( (tx-ix)**2 + (ty-iy)**2 )

# Normalize angle [-pi, +pi]
def normalize_angle(angle):
    if angle > math.pi:
        norm_angle = angle - 2*math.pi
    elif angle < -math.pi:
        norm_angle = angle + 2*math.pi
    else:
        norm_angle = angle
    return norm_angle

# Global2Local
def global2local(ego_x, ego_y, ego_yaw, x_list, y_list):
    """
    TODO 1.
    Transform from global to local coordinate w.r.t. ego vehicle's current pose (x, y, yaw).
        - x_list, y_list               : global coordinate trajectory.
        - ego_x, ego_y, ego_yaw        : ego vehicle's current pose.
        - output_x_list, output_y_list : transformed local coordinate trajectory.
    """
    R = np.mat(
       [[cos(-ego_yaw),  -sin(-ego_yaw), -(cos(-ego_yaw)*ego_x-sin(-ego_yaw)*ego_y)],
        [sin(-ego_yaw),   cos(-ego_yaw), -(sin(-ego_yaw)*ego_x+cos(-ego_yaw)*ego_y)],
        [            0,               0,                                          1]]
    )
    
    global_traj = np.vstack(
       [np.mat(x_list),
        np.mat(y_list),
        np.ones_like(x_list)]
    )

    local_traj = np.matmul(R, global_traj)

    output_x_list = local_traj[0,:]
    output_y_list = local_traj[1,:]

    return output_x_list, output_y_list

# Find nearest point
def find_nearest_point(ego_x, ego_y, x_list, y_list):
    """
    TODO 2.
    Find the nearest distance(near_dist) and its index(near_ind) w.r.t.
    current position (ego_x,y) and given trajectory (x_list,y_list).
        - Use 'calc_dist' function to calculate distance.
        - Use np.argmin for finding the index whose value is minimum.
    """
    diff = np.mat([(x_list-ego_x),
                   (y_list-ego_y)])
    dist = la.norm(diff, axis=0)
   
    near_ind = np.argmin(dist)
    near_dist = dist[near_ind]

    return near_dist, near_ind

# Calculate Error
def calc_error(ego_x, ego_y, ego_yaw, x_list, y_list, wpt_look_ahead=0):
    """
    TODO 3.
    1. Transform from global to local coordinate trajectory.
    2. Find the nearest waypoint.
    3. Set lookahead waypoint.
        - (hint) use the index of the nearest waypoint (near_ind) from 'find_nearest_point'.
        - (hint) consider that the next index of the terminal waypoint is 0, which is not terminal_index + 1.
    4. Calculate errors
        - error_yaw (yaw error)
            : (hint) use math.atan2 for calculating angle btw lookahead and next waypoints.
            : (hint) consider that the next index of the terminal waypoint is 0, which is not terminal_index + 1.
        - error_y (crosstrack error)
            : (hint) y value of the lookahead point in local coordinate waypoint trajectory.
    """
    # 1. Global to Local coordinate
    local_x_list, local_y_list = global2local(ego_x, ego_y, ego_yaw, x_list, y_list)
    
    # 2. Find the nearest waypoint
    _, near_ind = find_nearest_point(ego_x, ego_y, x_list, y_list)
    # near_wpt = local_x_list.item(near_ind), local_y_list.item(near_ind)

    # 3. Set lookahead waypoint (index of waypoint trajectory)
    lookahead_wpt_ind = (near_ind+wpt_look_ahead)%len(x_list)
    lookahead_wpt = (local_x_list.item(lookahead_wpt_ind), local_y_list.item(lookahead_wpt_ind))
    next_lookahead_wpt_ind = (near_ind+wpt_look_ahead+1)%len(x_list)
    next_lookahead_wpt = (local_x_list.item(next_lookahead_wpt_ind), local_y_list.item(next_lookahead_wpt_ind))

    # 4. Calculate errors
    error_yaw = math.atan2(next_lookahead_wpt[1]-lookahead_wpt[1], next_lookahead_wpt[0]-lookahead_wpt[0])
    error_yaw = normalize_angle(error_yaw) # Normalize angle to [-pi, +pi]

    error_y   = lookahead_wpt[1]

    return error_y, error_yaw

class WaypointFollower():
    def __init__(self):
        # ROS init
        rospy.init_node('wpt_follwer')
        self.control_freq = 10
        self.rate = rospy.Rate(self.control_freq)

        # Params
        self.target_speed = 10/3.6              # 10km/h??
        self.MAX_STEER    = np.deg2rad(17.75)

        self.THR_P = 0.75
        self.THR_I = 0.25
        self.THR_D = 0.25

        # integral term
        self.THR_IMAX = 250
        self.error_v_i = 0
        # derivative term
        self.error_v_prev = 0

        # vehicle state
        self.ego_x   = 0
        self.ego_y   = 0
        self.ego_yaw = 0
        self.ego_vx  = 0

        self.wpt_look_ahead = 1   # [index]

        # Pub/Sub
        self.pub_command = rospy.Publisher('/control', AckermannDriveStamped, queue_size=5)
        self.sub_odom    = rospy.Subscriber('/simulation/bodyOdom', Odometry, self.callback_odom)

    def callback_odom(self, msg):
        """
        Subscribe Odometry message
        ref: http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/Odometry.html
        """
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        self.ego_vx = msg.twist.twist.linear.x
        # get euler from quaternion
        q = msg.pose.pose.orientation
        q_list = [q.x, q.y, q.z, q.w]
        _, _, self.ego_yaw = euler_from_quaternion(q_list)

    # Controller
    def steer_control(self, error_y, error_yaw):
        """
        TODO 4.
        Implement a steering controller (PID controller or Pure pursuit or Stanley method).
        You can use not only error_y, error_yaw, but also other input arguments for this controller if you want.
        """

        k = 3
        steer_cmd = error_yaw 
        if self.ego_vx != 0:
            steer_cmd += atan2(k*error_y, self.ego_vx)

        # Control limit
        steer_cmd = np.clip(steer_cmd, -self.MAX_STEER, self.MAX_STEER)

        steer_cmd = steer_cmd * self.MAX_STEER

        return steer_cmd

    def speed_control(self, error_v):
        """
        TODO 5.
        Implement a speed controller (PID controller).
        You can use not only error_v, but also other input arguments for this controller if you want.
        """

        throttle_cmd = error_v*self.THR_P \
                    + min(self.THR_IMAX, self.error_v_i)/self.control_freq*self.THR_I \
                    + (error_v-self.error_v_prev)/self.control_freq*self.THR_P

        self.error_v_i += error_v
        self.error_v_i = min(self.THR_IMAX, self.error_v_i)
        self.error_v_prev = error_v

        return throttle_cmd

    def publish_command(self, steer, accel):
        """
        Publish command as AckermannDriveStamped
        ref: http://docs.ros.org/en/jade/api/ackermann_msgs/html/msg/AckermannDriveStamped.html
        """
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer
        msg.drive.acceleration = accel
        self.pub_command.publish(msg)

def main():
    # Load Waypoint
    rospack = rospkg.RosPack()
    WPT_CSV_PATH = rospack.get_path('waypoint_follower') + "/wpt_data/wpt_data_dense.csv"
    csv_data = pd.read_csv(WPT_CSV_PATH, sep=',', header=None)
    wpts_x = csv_data.values[:,0]
    wpts_y = csv_data.values[:,1]

    print("loaded wpt :", wpts_x.shape, wpts_y.shape)

    # Define controller
    wpt_control = WaypointFollower()

    avg_error_yy = 0
    avg_error_vv = 0
    i = 0

    while not rospy.is_shutdown():
        i += 1
        # Get current state
        ego_x = wpt_control.ego_x
        ego_y = wpt_control.ego_y
        ego_yaw = wpt_control.ego_yaw
        ego_vx = wpt_control.ego_vx

        # for stanely method, geometry update
        ego_x = ego_x + 0.257 * cos(ego_yaw) 
        ego_y = ego_y + 0.257 * sin(ego_yaw)

        # Lateral error calculation (cross-track error, yaw error)
        error_y, error_yaw = calc_error(ego_x, ego_y, ego_yaw, wpts_x, wpts_y, wpt_look_ahead=wpt_control.wpt_look_ahead)

        avg_error_yy += error_y**2

        # Longitudinal error calculation (speed error)
        error_v = wpt_control.target_speed - ego_vx

        avg_error_vv += error_v**2

        # Control
        steer_cmd = wpt_control.steer_control(error_y, error_yaw)
        throttle_cmd = wpt_control.speed_control(error_v)

        # Publish command
        wpt_control.publish_command(steer_cmd, throttle_cmd)

        rospy.loginfo("Commands: (steer=%.3f, accel=%.3f). Errors: (CrossTrackError=%.3f, YawError=%.3f, SpeedError=%.3f)." %(steer_cmd, throttle_cmd, error_y, error_yaw, error_v))
        rospy.loginfo("Average Error: CrossTrackError=%.3f, SpeedError=%.3f" %(sqrt(avg_error_yy/i), sqrt(avg_error_vv/i)))
        wpt_control.rate.sleep()

if __name__ == '__main__':
    main()
