#!/usr/bin/env python3

# Core ROS and geometric processing imports
import rospy
import threading
import numpy as np
from tf2_ros import Buffer, TransformListener, TransformException
from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint, TrafficLightResultArray, TrafficLightResult
from geometry_msgs.msg import PoseStamped, TwistStamped, PointStamped
from shapely.geometry import LineString, Point, Polygon
from shapely import prepare
from tf2_geometry_msgs import do_transform_pose
from scipy.interpolate import interp1d
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector

class SimpleLocalPlanner:
    def __init__(self):
        # Load all planning and safety parameters from ROS parameter server
        self.output_frame = rospy.get_param("~output_frame")
        self.local_path_length = rospy.get_param("~local_path_length")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        
        # Safety distance parameters for different scenarios
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        
        # Vehicle behavior parameters
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.tfl_maximum_deceleration = rospy.get_param("~tfl_maximum_deceleration")

        # Lanelet2 map parameters
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # Initialize internal state variables
        self.lock = threading.Lock()
        self.global_path_linestring = None
        self.global_path_distances = None
        self.distance_to_velocity_interpolator = None
        self.current_speed = None
        self.current_position = None
        self.goal_point = None
        self.goal_reached = False
        self.red_stoplines = set()

        # Set up TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Load and process Lanelet2 map
        try:
            if coordinate_transformer == "utm":
                projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
            else:
                raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
            
            lanelet2_map = load(lanelet2_map_name, projector)
            self.stoplines = self.get_stoplines(lanelet2_map)
            rospy.loginfo(f"Loaded {len(self.stoplines)} stop lines from lanelet2 map")
            
        except Exception as e:
            rospy.logerr(f"Failed to load lanelet2 map: {str(e)}")
            raise

        # Set up ROS publishers and subscribers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # Subscribe to all required topics
        rospy.Subscriber('global_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_callback, queue_size=1, tcp_nodelay=True)

    @staticmethod
    def get_stoplines(lanelet2_map):
        """Extract all stop lines from the Lanelet2 map and convert to Shapely LineStrings"""
        stoplines = {}
        for line in lanelet2_map.lineStringLayer:
            if line.attributes:
                if line.attributes["type"] == "stop_line":
                    stoplines[line.id] = LineString([(p.x, p.y) for p in line])
        return stoplines

    @staticmethod
    def calculate_deceleration(v, v0, s):
        """
        Calculate required deceleration for safe stopping
        v: target velocity (usually 0 for stopping)
        v0: current velocity
        s: distance to stopping point
        """
        if s <= 0:
            return float('inf')
        return (v**2 - v0**2) / (2 * s)

    def traffic_light_callback(self, msg):
        """Process traffic light states and store red light stop line IDs"""
        with self.lock:
            self.red_stoplines.clear()
            for result in msg.results:
                if result.recognition_result == 0:  # RED or YELLOW
                    self.red_stoplines.add(result.lane_id)
                    rospy.logdebug(f"RED light detected at stop line {result.lane_id}")

    def path_callback(self, msg):
        """Process new global path and create velocity profile"""
        with self.lock:
            if len(msg.waypoints) == 0:
                # Don't set goal_reached or clear path when receiving empty path
                rospy.loginfo("%s - Empty global path received, continuing to track current goal", rospy.get_name())
                return  # Keep using existing path data
            else:
                # Convert waypoints to numpy array for efficient processing
                waypoints_xyz = np.array([(w.pose.pose.position.x, w.pose.pose.position.y, w.pose.pose.position.z) for w in msg.waypoints])
                self.global_path_linestring = LineString(waypoints_xyz)
                prepare(self.global_path_linestring)

                # Calculate cumulative distances along path
                self.global_path_distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xyz[:,:2], axis=0)**2, axis=1)))
                self.global_path_distances = np.insert(self.global_path_distances, 0, 0)

                # Create velocity profile
                velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
                self.distance_to_velocity_interpolator = interp1d(self.global_path_distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

                self.goal_point = Point(waypoints_xyz[-1])
                self.goal_reached = False

                rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(msg.waypoints))

    def current_velocity_callback(self, msg):
        """Store current vehicle velocity"""
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        """Store current vehicle position"""
        self.current_position = Point([msg.pose.position.x, msg.pose.position.y])

    def detected_objects_callback(self, msg):
        """
        Main planning callback that handles:
        - Local path generation
        - Obstacle detection and avoidance
        - Traffic light response
        - Velocity planning
        """
        if self.goal_reached:
            return

        # Get current state with thread safety
        with self.lock:
            global_path_linestring = self.global_path_linestring
            global_path_distances = self.global_path_distances
            distance_to_velocity_interpolator = self.distance_to_velocity_interpolator
            current_position = self.current_position
            current_speed = self.current_speed
            goal_point = self.goal_point
            goal_reached = self.goal_reached
            red_stoplines = self.red_stoplines.copy()

        # Check if we have all required data
        if global_path_linestring is None or global_path_distances is None or \
           distance_to_velocity_interpolator is None or current_position is None or \
           current_speed is None:
            self.publish_local_path_wp([], rospy.Time.now(), self.output_frame)
            return

        # Calculate distance along path
        d_ego_from_path_start = global_path_linestring.project(current_position)

        # Extract local path segment
        local_path = self.extract_local_path(
            global_path_linestring,
            global_path_distances,
            d_ego_from_path_start,
            self.local_path_length
        )

        if local_path is None:
            self.publish_local_path_wp([], rospy.Time.now(), self.output_frame)
            return

        # Initialize velocity from path
        map_target_velocity = float(distance_to_velocity_interpolator(d_ego_from_path_start))
        target_velocity = map_target_velocity

        # Create buffer for collision checking
        local_path_buffer = local_path.buffer(self.stopping_lateral_distance, cap_style='flat')
        prepare(local_path_buffer)

        # Initialize lists for tracking objects and constraints
        object_distances = []
        adjusted_object_distances = []
        object_velocities = []
        object_braking_distances = []
        object_types = []
        local_path_blocked = False

        # Process RED traffic lights
        for stopline_id in red_stoplines:
            stopline = self.stoplines.get(stopline_id)
            if stopline is not None and local_path.intersects(stopline):
                intersection_point = local_path.intersection(stopline)
                if not intersection_point.is_empty:
                    d_stopline = local_path.project(intersection_point)
                    
                    # Calculate stopping distance using consistent logic
                    stopping_distance = d_stopline - self.current_pose_to_car_front
                    required_decel = abs(self.calculate_deceleration(0, current_speed, max(0.1, stopping_distance)))
                    
                    if required_decel <= self.tfl_maximum_deceleration:
                        # Add stopline using same logic as obstacles
                        object_distances.append(d_stopline)
                        adjusted_s_stopline = d_stopline - self.current_pose_to_car_front - self.braking_safety_distance_stopline
                        adjusted_s_stopline = max(0.0, adjusted_s_stopline)
                        adjusted_object_distances.append(adjusted_s_stopline)
                        object_velocities.append(0.0)
                        object_braking_distances.append(self.braking_safety_distance_stopline)
                        object_types.append('stopline')
                        local_path_blocked = True
                    else:
                        rospy.logwarn_throttle(3, 
                            f"Ignoring RED light at stop line {stopline_id}. "
                            f"Required deceleration ({required_decel:.2f} m/s^2) exceeds "
                            f"maximum allowed ({self.tfl_maximum_deceleration:.2f} m/s^2)")

        # Handle goal point if it exists
        if goal_point is not None:
            d_goal_from_local_start = local_path.project(goal_point)
            
            if d_goal_from_local_start <= local_path.length:
                # Add the actual goal point
                object_distances.append(d_goal_from_local_start)
                
                # Only subtract current_pose_to_car_front to align with actual goal
                adjusted_s_goal = d_goal_from_local_start - self.current_pose_to_car_front
                adjusted_s_goal = max(0.0, adjusted_s_goal)

                # Check both distance and speed for goal reached
                if adjusted_s_goal <= 0.1 and current_speed < 0.1:
                    rospy.loginfo("%s - Local goal reached, vehicle stopped", rospy.get_name())
                    waypoint = Waypoint()
                    current_point = list(local_path.coords)[0]
                    waypoint.pose.pose.position.x = current_point[0]
                    waypoint.pose.pose.position.y = current_point[1]
                    waypoint.pose.pose.position.z = current_point[2] if len(current_point) > 2 else 0.0
                    waypoint.twist.twist.linear.x = 0.0
                    
                    # Now we can set goal reached
                    self.goal_reached = True
                    self.publish_local_path_wp(
                        [waypoint],
                        rospy.Time.now(),
                        self.output_frame,
                        closest_object_distance=0.0,
                        closest_object_velocity=0.0,
                        local_path_blocked=True,
                        stopping_point_distance=0.0
                    )
                    return

                adjusted_object_distances.append(adjusted_s_goal)
                object_velocities.append(0.0)  # Stationary
                object_braking_distances.append(0.0)  # No safety distance for goal
                object_types.append('goal')

        # Handle obstacle detection and transformation
        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame,
                msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(self.transform_timeout))
        except (TransformException, rospy.exceptions.ROSTimeMovedBackwardsException) as ex:
            transform = None

        # Process detected obstacles
        for obj in msg.objects:
            if transform is not None:
                try:
                    pose_stamped = PoseStamped()
                    pose_stamped.header = obj.header
                    pose_stamped.pose = obj.pose
                    transformed_pose = do_transform_pose(pose_stamped, transform)
                except TransformException:
                    continue
            else:
                continue

            # Check object footprint
            footprint = obj.convex_hull.polygon
            if len(footprint.points) < 3:
                continue
            object_polygon_coords = [(p.x, p.y) for p in footprint.points]
            object_polygon = Polygon(object_polygon_coords)

            # Process intersecting obstacles
            if local_path_buffer.intersects(object_polygon):
                object_centroid = object_polygon.centroid
                d_object_from_local_start = local_path.project(object_centroid)
                object_distances.append(d_object_from_local_start)

                adjusted_s_object = d_object_from_local_start - self.current_pose_to_car_front - self.braking_safety_distance_obstacle
                adjusted_s_object = max(0.0, adjusted_s_object)
                adjusted_object_distances.append(adjusted_s_object)

                object_velocity = np.hypot(obj.velocity.linear.x, obj.velocity.linear.y)
                object_velocities.append(object_velocity)
                object_braking_distances.append(self.braking_safety_distance_obstacle)
                object_types.append('obstacle')

        # Calculate velocity adjustments for obstacles and goal
        if adjusted_object_distances:
            adjusted_object_distances = np.array(adjusted_object_distances)
            object_velocities = np.array(object_velocities)
            object_braking_distances = np.array(object_braking_distances)
            object_types = np.array(object_types)

            # Calculate safe velocities considering reaction time
            target_distances = adjusted_object_distances - self.braking_reaction_time * np.abs(object_velocities)
            target_distances = np.maximum(0.0, target_distances)

            # Calculate safe velocities based on physics
            v_obj_for_calculation = np.maximum(0.0, object_velocities)
            delta = v_obj_for_calculation**2 + 2 * abs(self.default_deceleration) * target_distances
            delta = np.maximum(0.0, delta)
            target_velocities = np.sqrt(delta)

            # Find most restrictive velocity constraint
            min_target_velocity = np.min(target_velocities)
            min_index = np.argmin(target_velocities)
            target_velocity = min(min_target_velocity, target_velocity)

            # Handle different types of constraints
            is_goal = object_types[min_index] == "goal"
            is_stopline = object_types[min_index] == "stopline"
            
            if is_goal:
                # Goal point handling - don't mark path as blocked
                stopping_point_distance = object_distances[min_index]
                closest_object_distance = adjusted_object_distances[min_index]
                closest_object_velocity = object_velocities[min_index]
                local_path_blocked = False
            elif is_stopline:
                # Handle stop line constraint
                stopping_point_distance = object_distances[min_index]
                closest_object_distance = adjusted_object_distances[min_index]
                closest_object_velocity = object_velocities[min_index]
                local_path_blocked = True
            else:
                # Handle obstacle constraint
                stopping_point_distance = object_distances[min_index] - self.current_pose_to_car_front
                closest_object_distance = object_distances[min_index]
                closest_object_velocity = object_velocities[min_index]
                local_path_blocked = True
        else:
            # No constraints, use default values
            stopping_point_distance = 0.0
            closest_object_distance = float('inf')
            closest_object_velocity = 0.0
            local_path_blocked = False

        # Generate and publish final path
        local_path_waypoints = self.convert_local_path_to_waypoints(local_path, target_velocity)
        self.publish_local_path_wp(
            local_path_waypoints,
            rospy.Time.now(),
            self.output_frame,
            closest_object_distance=closest_object_distance,
            closest_object_velocity=closest_object_velocity,
            local_path_blocked=local_path_blocked,
            stopping_point_distance=stopping_point_distance
        )

    def extract_local_path(self, global_path_linestring, global_path_distances, d_ego_from_path_start, local_path_length):
        """
        Extract a local section of the global path starting from current position
        Returns None if beyond path end
        """
        if d_ego_from_path_start >= global_path_linestring.length:
            return None

        # Calculate end point of local path
        d_to_local_path_end = d_ego_from_path_start + local_path_length
        
        # Find relevant waypoint indices
        index_start = np.argmax(global_path_distances >= d_ego_from_path_start)
        index_end = np.argmax(global_path_distances >= d_to_local_path_end)

        if index_end == 0:
            index_end = len(global_path_linestring.coords) - 1

        # Create local path with interpolated endpoints
        start_point = global_path_linestring.interpolate(d_ego_from_path_start)
        end_point = global_path_linestring.interpolate(min(d_to_local_path_end, global_path_linestring.length))
        local_path = LineString([start_point] + list(global_path_linestring.coords[index_start:index_end]) + [end_point])

        return local_path

    def convert_local_path_to_waypoints(self, local_path, target_velocity):
        """Convert geometric path to ROS waypoints with velocity"""
        local_path_waypoints = []
        for point in local_path.coords:
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = point[0]
            waypoint.pose.pose.position.y = point[1]
            waypoint.pose.pose.position.z = point[2] if len(point) > 2 else 0.0
            waypoint.twist.twist.linear.x = target_velocity
            local_path_waypoints.append(waypoint)
        return local_path_waypoints

    def publish_local_path_wp(self, local_path_waypoints, stamp, output_frame, 
                            closest_object_distance=0.0, closest_object_velocity=0.0, 
                            local_path_blocked=False, stopping_point_distance=0.0):
        """Publish local path with additional metadata"""
        lane = Lane()
        lane.header.frame_id = output_frame
        lane.header.stamp = stamp
        lane.waypoints = local_path_waypoints
        
        if local_path_blocked:
            lane.closest_object_distance = stopping_point_distance
        else:
            lane.closest_object_distance = closest_object_distance
            
        lane.closest_object_velocity = closest_object_velocity
        lane.is_blocked = local_path_blocked
        lane.cost = stopping_point_distance
        self.local_path_pub.publish(lane)

    def run(self):
        """Main run loop"""
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node('simple_local_planner')
        node = SimpleLocalPlanner()
        node.run()
    except rospy.ROSInterruptException:
        pass
