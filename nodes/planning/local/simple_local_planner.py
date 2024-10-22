#!/usr/bin/env python3

import rospy
import math
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Vector3Stamped
from shapely.geometry import LineString, Point, Polygon
from shapely import prepare
from tf2_geometry_msgs import do_transform_vector3
from scipy.interpolate import interp1d
from numpy.lib.recfunctions import unstructured_to_structured

class SimpleLocalPlanner:

    def __init__(self):
        # Parameters
        self.output_frame = rospy.get_param("~output_frame")
        self.local_path_length = rospy.get_param("~local_path_length")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.default_deceleration = rospy.get_param("default_deceleration")

        # Variables
        self.lock = threading.Lock()
        self.global_path_linestring = None
        self.global_path_distances = None
        self.distance_to_velocity_interpolator = None
        self.current_speed = None
        self.current_position = None
        self.goal_point = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('global_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def path_callback(self, msg):
        if len(msg.waypoints) == 0:
            global_path_linestring = None
            global_path_distances = None
            distance_to_velocity_interpolator = None
            goal_point = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())
        else:
            waypoints_xyz = np.array([(w.pose.pose.position.x, w.pose.pose.position.y, w.pose.pose.position.z) for w in msg.waypoints])
            global_path_linestring = LineString(waypoints_xyz)
            prepare(global_path_linestring)

            global_path_distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xyz[:,:2], axis=0)**2, axis=1)))
            global_path_distances = np.insert(global_path_distances, 0, 0)

            velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
            distance_to_velocity_interpolator = interp1d(global_path_distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

            # Store the last point as the goal
            goal_point = Point(waypoints_xyz[-1])

            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(msg.waypoints))

        with self.lock:
            self.global_path_linestring = global_path_linestring
            self.global_path_distances = global_path_distances
            self.distance_to_velocity_interpolator = distance_to_velocity_interpolator
            self.goal_point = goal_point

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = Point([msg.pose.position.x, msg.pose.position.y])

    def detected_objects_callback(self, msg):
        with self.lock:
            global_path_linestring = self.global_path_linestring
            global_path_distances = self.global_path_distances
            distance_to_velocity_interpolator = self.distance_to_velocity_interpolator
            current_position = self.current_position
            current_speed = self.current_speed
            goal_point = self.goal_point

        # Check if all necessary variables are set
        if (global_path_linestring is None or global_path_distances is None or
            distance_to_velocity_interpolator is None or current_position is None or
            current_speed is None or goal_point is None):
            # Publish empty local path
            self.publish_local_path_wp([], rospy.Time.now(), self.output_frame)
            return

        # Find ego vehicle location on the global path
        d_ego_from_path_start = global_path_linestring.project(current_position)

        # Extract local path
        local_path = self.extract_local_path(global_path_linestring, global_path_distances, 
                                           d_ego_from_path_start, self.local_path_length)

        if local_path is None:
            # Publish empty local path
            self.publish_local_path_wp([], rospy.Time.now(), self.output_frame)
            return

        # Calculate map-based target velocity
        map_target_velocity = distance_to_velocity_interpolator(d_ego_from_path_start)

        # Create a buffer around the local path
        local_path_buffer = local_path.buffer(self.stopping_lateral_distance, cap_style="flat")
        prepare(local_path_buffer)

        # Check for intersections with detected objects
        local_path_blocked = False
        object_distances = []
        object_velocities = []
        object_braking_distances = []

        print(f"------ detected objects callback, number of objects: {len(msg.objects)}")

        for obj in msg.objects:
            # Create a polygon from the object's convex hull
            obj_points = [(p.x, p.y) for p in obj.convex_hull.polygon.points]
            obj_polygon = Polygon(obj_points)

            # Check if the object intersects with the local path buffer
            if local_path_buffer.intersects(obj_polygon):
                local_path_blocked = True

                # Calculate distances for all the object points that intersect with local_path_buffer
                obj_distances = [local_path.project(Point(coords)) for coords in obj_polygon.exterior.coords]
                min_obj_distance = min(obj_distances)
                
                # Calculate the actual distance to the object
                actual_distance = min_obj_distance - self.current_pose_to_car_front
                object_distances.append(actual_distance)
                object_braking_distances.append(self.braking_safety_distance_obstacle)

                # Get the transform from map to base_link
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_link',
                        'map',
                        rospy.Time(0),
                        rospy.Duration(self.transform_timeout)
                    )
                except (TransformException, rospy.ROSException) as ex:
                    rospy.logwarn(f"Could not transform velocity: {ex}")
                    transform = None

                # Project object velocity to base_link frame to get longitudinal speed
                if transform is not None:
                    vector3_stamped = Vector3Stamped(vector=obj.velocity.linear)
                    velocity = do_transform_vector3(vector3_stamped, transform).vector
                else:
                    velocity = Vector3()
                object_velocity = velocity.x
                object_velocities.append(object_velocity)

                print(f"object velocity: {math.sqrt(obj.velocity.linear.x**2 + obj.velocity.linear.y**2):.6f} transformed velocity: {object_velocity:.6f}")

        # Consider the goal point if it's within local path length
        goal_distance = global_path_linestring.project(goal_point) - d_ego_from_path_start
        if goal_distance <= self.local_path_length:
            # Create a perpendicular line at the goal point to represent the stopping wall
            goal_coords = np.array(goal_point.coords[0])
            path_direction = np.array(local_path.coords[-1]) - np.array(local_path.coords[-2])
            path_direction = path_direction[:2] / np.linalg.norm(path_direction[:2])
            perpendicular = np.array([-path_direction[1], path_direction[0]])
            
            # Create a wall polygon perpendicular to the path
            wall_length = self.stopping_lateral_distance * 2
            wall_points = [
                (goal_coords[0] - perpendicular[0] * wall_length, goal_coords[1] - perpendicular[1] * wall_length),
                (goal_coords[0] + perpendicular[0] * wall_length, goal_coords[1] + perpendicular[1] * wall_length)
            ]
            wall_line = LineString(wall_points)
            
            # Calculate the distance to the goal considering the perpendicular wall
            goal_intersection = local_path.intersection(wall_line)
            if not goal_intersection.is_empty:
                goal_distance = local_path.project(goal_intersection)
            
            object_distances.append(goal_distance - self.current_pose_to_car_front)
            object_velocities.append(0.0)  # Goal point is stationary
            object_braking_distances.append(self.braking_safety_distance_goal)
            print(f"Goal point added as obstacle. Distance: {goal_distance:.6f}")

        # Calculate target velocity based on obstacles and goal
        if object_distances:
            object_distances = np.array(object_distances)
            object_velocities = np.array(object_velocities)
            object_braking_distances = np.array(object_braking_distances)

            # Calculate target distances considering safe following distance
            target_distances = object_distances - self.braking_reaction_time * np.abs(object_velocities)
            target_distances = np.maximum(0, target_distances - object_braking_distances)

            # v = sqrt(v0^2 + 2as)
            target_velocities = np.sqrt(np.maximum(0, np.maximum(0, object_velocities)**2 + 2 * self.default_deceleration * target_distances))
            
            min_index = np.argmin(target_velocities)
            obstacle_target_velocity = target_velocities[min_index]
            closest_object_distance = object_distances[min_index]
            closest_object_velocity = object_velocities[min_index]
            stopping_point_distance = target_distances[min_index] + self.current_pose_to_car_front
        else:
            obstacle_target_velocity = float('inf')
            closest_object_distance = float('inf')
            closest_object_velocity = 0.0
            stopping_point_distance = 0.0

        # Choose the minimum of map-based and obstacle-based target velocities
        target_velocity = min(map_target_velocity, obstacle_target_velocity)

        # Convert local path to waypoints
        local_path_waypoints = self.convert_local_path_to_waypoints(local_path, target_velocity)

        # Publish local path
        self.publish_local_path_wp(local_path_waypoints, rospy.Time.now(), self.output_frame,
                                 closest_object_distance, closest_object_velocity,
                                 local_path_blocked, stopping_point_distance)

        print(f"Target velocity: {target_velocity:.6f}")
        print(f"Closest object distance: {closest_object_distance:.6f}")
        print(f"Closest object velocity: {closest_object_velocity:.6f}")
        print(f"Stopping point distance: {stopping_point_distance:.6f}")
        print(f"Local path blocked: {local_path_blocked}")

    def extract_local_path(self, global_path_linestring, global_path_distances, d_ego_from_path_start, local_path_length):
        if math.isclose(d_ego_from_path_start, global_path_linestring.length):
            return None

        d_to_local_path_end = d_ego_from_path_start + local_path_length

        index_start = np.argmax(global_path_distances >= d_ego_from_path_start)
        index_end = np.argmax(global_path_distances >= d_to_local_path_end)

        if index_end == 0:
            index_end = len(global_path_linestring.coords) - 1

        start_point = global_path_linestring.interpolate(d_ego_from_path_start)
        end_point = global_path_linestring.interpolate(d_to_local_path_end)
        local_path = LineString([start_point] + list(global_path_linestring.coords[index_start:index_end]) + [end_point])

        return local_path

    def convert_local_path_to_waypoints(self, local_path, target_velocity):
        local_path_waypoints = []
        for point in local_path.coords:
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = point[0]
            waypoint.pose.pose.position.y = point[1]
            waypoint.pose.pose.position.z = point[2]
            waypoint.twist.twist.linear.x = target_velocity
            local_path_waypoints.append(waypoint)
        return local_path_waypoints

    def publish_local_path_wp(self, local_path_waypoints, stamp, output_frame, closest_object_distance=0.0, closest_object_velocity=0.0, local_path_blocked=False, stopping_point_distance=0.0):
        lane = Lane()
        lane.header.frame_id = output_frame
        lane.header.stamp = stamp
        lane.waypoints = local_path_waypoints
        lane.closest_object_distance = closest_object_distance
        lane.closest_object_velocity = closest_object_velocity
        lane.is_blocked = local_path_blocked
        lane.cost = stopping_point_distance
        self.local_path_pub.publish(lane)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('simple_local_planner')
    node = SimpleLocalPlanner()
    node.run()
