#!/usr/bin/env python3

import rospy
import numpy as np
from autoware_msgs.msg import Lane, VehicleCmd
from geometry_msgs.msg import PoseStamped
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):
        # Parameters
        self.current_pose = None
        self.path_linestring = None
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 4.0)
        self.wheel_base = rospy.get_param('/vehicle/wheel_base', 2.7)
        self.distance_to_velocity_interpolator = None

        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=1)

        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        if len(msg.waypoints) < 2:
            # If the path has less than 2 waypoints, stop the vehicle
            rospy.logwarn("Received an empty or too-short path. Stopping the vehicle.")
            self.path_linestring = None
            self.distance_to_velocity_interpolator = None
            self.publish_vehicle_cmd(0.0, 0.0)  # Stop the vehicle
            return

        # Convert waypoints to shapely linestring
        self.path_linestring = LineString([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        prepare(self.path_linestring)

        # Collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        
        # Calculate distances between points
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)  # Add 0 distance at the beginning
        
        # Extract velocity values at waypoints
        velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
        
        # Create the interpolator
        self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)
        
        rospy.loginfo("Path received and velocity interpolator created")

    def current_pose_callback(self, msg):
        self.current_pose = msg
        
        if self.path_linestring is None or self.distance_to_velocity_interpolator is None:
            rospy.logwarn("No valid path available, stopping the vehicle.")
            self.publish_vehicle_cmd(0.0, 0.0)  # Stop the vehicle
            return

        current_pose_point = Point(msg.pose.position.x, msg.pose.position.y)
        d_ego_from_path_start = self.path_linestring.project(current_pose_point)
        
        # Calculate steering angle
        steering_angle = self.calculate_steering_angle(current_pose_point, d_ego_from_path_start)
        
        # Interpolate velocity
        velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)

        self.publish_vehicle_cmd(steering_angle, velocity)

    def calculate_steering_angle(self, current_pose_point, d_ego_from_path_start):
        # Find lookahead point
        lookahead_point = self.path_linestring.interpolate(d_ego_from_path_start + self.lookahead_distance)
        
        # Get current vehicle orientation (heading)
        _, _, heading = euler_from_quaternion([
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w
        ])
        
        # Calculate lookahead point heading
        lookahead_heading = np.arctan2(lookahead_point.y - current_pose_point.y, 
                                       lookahead_point.x - current_pose_point.x)
        
        # Calculate α (alpha) - the difference in car heading and lookahead point heading
        alpha = lookahead_heading - heading
        
        # Normalize alpha to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        
        # Recalculate the actual lookahead distance (ld)
        ld = distance(current_pose_point, Point(lookahead_point.x, lookahead_point.y))
        
        # Calculate steering angle
        steering_angle = np.arctan2(2 * self.wheel_base * np.sin(alpha), ld)
        
        return steering_angle

    def publish_vehicle_cmd(self, steering_angle, velocity):
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.header.stamp = self.current_pose.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        
        vehicle_cmd.ctrl_cmd.linear_velocity = velocity
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle

        self.vehicle_cmd_pub.publish(vehicle_cmd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()

