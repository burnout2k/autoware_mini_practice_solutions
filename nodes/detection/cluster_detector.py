#!/usr/bin/env python3

import rospy
import numpy as np
from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32, Pose

BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data, dtype=np.float32)
        
        xyz_points = points[:, :3]
        labels = points[:, 3]
        
        if msg.header.frame_id != self.output_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.output_frame,
                    msg.header.frame_id,
                    msg.header.stamp,
                    rospy.Duration(self.transform_timeout)
                )
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
                
            tf_matrix = numpify(transform.transform).astype(np.float32)
            homogeneous_points = np.column_stack((xyz_points, np.ones(len(xyz_points))))
            transformed_homogeneous = homogeneous_points.dot(tf_matrix.T)
            xyz_points = transformed_homogeneous[:, :3]

        points = np.column_stack((xyz_points, labels))

        # Prepare header
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.output_frame

        # Create DetectedObjectArray message
        detected_objects_msg = DetectedObjectArray()
        detected_objects_msg.header = header

        # Get unique labels
        unique_labels = np.unique(labels)

        # Iterate over objects (clusters)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            mask = (labels == label)
            cluster_points = points[mask, :3]

            if len(cluster_points) < self.min_cluster_size:
                continue

            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)

            # Calculate convex hull
            points_2d = MultiPoint(cluster_points[:, :2])
            hull = points_2d.convex_hull
            convex_hull_points = [Point32(x, y, centroid[2]) for x, y in hull.exterior.coords]

            # Create DetectedObject
            obj = DetectedObject()
            obj.header = header
            obj.id = int(label)
            obj.label = "unknown"
            obj.color = BLUE80P
            obj.valid = True
            obj.space_frame = self.output_frame
            obj.pose_reliable = True
            obj.velocity_reliable = False
            obj.acceleration_reliable = False

            obj.pose.position.x = centroid[0]
            obj.pose.position.y = centroid[1]
            obj.pose.position.z = centroid[2]

            obj.convex_hull.polygon.points = convex_hull_points

            detected_objects_msg.objects.append(obj)

        # Publish detected objects
        self.objects_pub.publish(detected_objects_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()
