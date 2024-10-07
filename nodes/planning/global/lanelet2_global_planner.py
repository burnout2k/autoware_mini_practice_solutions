#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane, Waypoint
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest
import math
from shapely.geometry import Point, LineString

# ANSI color codes for terminal output
GREEN = '\033[92m'
ENDC = '\033[0m'

class Lanelet2GlobalPlanner:
    def __init__(self):
        rospy.init_node('lanelet2_global_planner', anonymous=True)
        
        rospy.loginfo("Initializing Lanelet2GlobalPlanner...")
        
        # Load parameters directly in __init__
        self.map_frame = rospy.get_param('~output_frame', 'map')
        self.lanelet2_map_name = rospy.get_param('~lanelet2_map_name', '')
        self.origin_lat = rospy.get_param('/localization/utm_origin_lat', 0.0)
        self.origin_lon = rospy.get_param('/localization/utm_origin_lon', 0.0)
        self.speed_limit = rospy.get_param('~speed_limit', 40.0) / 3.6  # Convert km/h to m/s
        self.distance_to_goal_limit = rospy.get_param('~distance_to_goal_limit', 5.0)
        
        rospy.loginfo("Parameters loaded: map_frame=%s, lanelet2_map_name=%s, origin_lat=%f, origin_lon=%f, speed_limit=%f, distance_to_goal_limit=%f",
                      self.map_frame, self.lanelet2_map_name, self.origin_lat, self.origin_lon, self.speed_limit, self.distance_to_goal_limit)
        
        if not self.lanelet2_map_name:
            rospy.logerr("%s - lanelet2_map_name parameter is not set!", rospy.get_name())
            rospy.signal_shutdown("Required parameter 'lanelet2_map_name' not set")
        
        # Load Lanelet2 map
        rospy.loginfo("Loading Lanelet2 map...")
        projector = UtmProjector(Origin(self.origin_lat, self.origin_lon))
        self.lanelet2_map = load(self.lanelet2_map_name, projector)
        rospy.loginfo("%s - Lanelet2 map loaded successfully", rospy.get_name())
        
        # Set up traffic rules and routing graph
        rospy.loginfo("Setting up traffic rules and routing graph...")
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)
        rospy.loginfo("Traffic rules and routing graph set up successfully.")
        
        # Initialize state variables
        self.current_pose = None
        self.goal_point = None
        self.path_published = False
        self.goal_reached = False
        
        # Set up publisher and subscribers
        self.waypoints_pub = rospy.Publisher('global_path', Lane, queue_size=1, latch=True)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback)
        
        rospy.loginfo("Lanelet2GlobalPlanner initialized successfully.")
        rospy.spin()

    def current_pose_callback(self, msg):
        # Update current pose and check if goal is reached
        self.current_pose = msg
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        
        if self.goal_point and self.path_published and not self.goal_reached:
            distance_to_goal = self.distance(self.current_location, self.goal_point)
            
            if distance_to_goal <= self.distance_to_goal_limit:
                rospy.loginfo(f"{GREEN}%s - Goal reached!{ENDC}", rospy.get_name())
                self.goal_reached = True
                self.publish_waypoints([])  # Publish empty path

    def goal_callback(self, msg):
        # Process new goal, plan route, and publish waypoints
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        self.goal_reached = False
        rospy.loginfo("%s - New goal received: (%f, %f)", rospy.get_name(), self.goal_point.x, self.goal_point.y)
        
        route = self.plan_route()
        if route is None:
            rospy.logwarn("%s - Unable to find a valid route to the goal.", rospy.get_name())
        else:
            waypoints = self.convert_lanelet_sequence_to_waypoints(route)
            adjusted_waypoints, adjusted_goal = self.adjust_path_and_goal(waypoints, self.goal_point)
            self.goal_point = adjusted_goal  # Update the goal point
            self.publish_waypoints(adjusted_waypoints)
            self.path_published = True

    def plan_route(self):
        # Plan a route from current position to goal
        if self.current_pose is None:
            rospy.logwarn("%s - Current pose not available. Cannot plan route.", rospy.get_name())
            return None

        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]

        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route is None:
            rospy.logwarn("%s - No route found between start and goal.", rospy.get_name())
            return None

        path = route.shortestPath()
        path_no_lane_change = path.getRemainingLane(start_lanelet)

        return path_no_lane_change

    def convert_lanelet_sequence_to_waypoints(self, lanelet_sequence):
        # Convert Lanelet sequence to a list of Waypoints
        waypoints = []
        last_point = None
        for lanelet in lanelet_sequence:
            if 'speed_ref' in lanelet.attributes:
                speed = min(float(lanelet.attributes['speed_ref']) / 3.6, self.speed_limit)  # Convert km/h to m/s
            else:
                speed = self.speed_limit

            for point in lanelet.centerline:
                if last_point and (point.x, point.y, point.z) == (last_point.x, last_point.y, last_point.z):
                    continue  # Skip overlapping points
                
                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.twist.twist.linear.x = speed
                waypoints.append(waypoint)
                last_point = point

        return waypoints

    def adjust_path_and_goal(self, waypoints, goal):
        # Adjust the path to end at the projected goal point
        path_points = [(wp.pose.pose.position.x, wp.pose.pose.position.y) for wp in waypoints]
        path_line = LineString(path_points)
        goal_point = Point(goal.x, goal.y)
        
        # Project goal onto path
        projected_point = path_line.interpolate(path_line.project(goal_point))
        
        # Find the segment of the path where the projected point lies
        for i in range(len(path_points) - 1):
            segment = LineString([path_points[i], path_points[i+1]])
            if segment.distance(projected_point) < 1e-6:  # Small threshold for floating-point comparison
                break
        
        # Create new waypoint at the projected point
        new_waypoint = Waypoint()
        new_waypoint.pose.pose.position.x = projected_point.x
        new_waypoint.pose.pose.position.y = projected_point.y
        new_waypoint.pose.pose.position.z = waypoints[i].pose.pose.position.z  # Interpolate Z if needed
        new_waypoint.twist.twist.linear.x = waypoints[i].twist.twist.linear.x  # Use speed from previous waypoint
        
        # Adjust the path
        adjusted_waypoints = waypoints[:i+1] + [new_waypoint]
        
        adjusted_goal = BasicPoint2d(projected_point.x, projected_point.y)
        
        rospy.loginfo("%s - Adjusted goal point: (%f, %f)", rospy.get_name(), 
                      adjusted_goal.x, adjusted_goal.y)
        
        return adjusted_waypoints, adjusted_goal

    def publish_waypoints(self, waypoints):
        # Publish the waypoints as a Lane message (handles both normal and empty paths)
        lane = Lane()
        lane.header.frame_id = self.map_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.waypoints_pub.publish(lane)
        if waypoints:
            rospy.loginfo("%s - Published %d waypoints to global_path", rospy.get_name(), len(waypoints))
        else:
            rospy.loginfo("%s - Published empty path to global_path", rospy.get_name())
            self.path_published = False

    def distance(self, point1, point2):
        # Calculate Euclidean distance between two points
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

if __name__ == '__main__':
    try:
        Lanelet2GlobalPlanner()
    except rospy.ROSInterruptException:
        pass
