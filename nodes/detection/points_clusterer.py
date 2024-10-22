#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify, msgify
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from sklearn.cluster import DBSCAN

class PointsClusterer:
    def __init__(self):
        
        # Get clustering parameters from the parameter server
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')
        
        # Initialize DBSCAN clusterer
        self.clusterer = DBSCAN(
            eps=self.cluster_epsilon,
            min_samples=self.cluster_min_size,
            n_jobs=-1
        )
        
        # Create publisher before subscriber
        self.clusters_pub = rospy.Publisher(
            'points_clustered',
            PointCloud2,
            queue_size=1,
            tcp_nodelay=True
        )
        
        # Subscribe to the filtered points topic
        self.points_sub = rospy.Subscriber(
            'points_filtered',
            PointCloud2,
            self.points_callback,
            queue_size=1,
            buff_size=2**24,
            tcp_nodelay=True
        )

    def points_callback(self, msg):
        # Convert ROS message to numpy array
        data = numpify(msg)
        
        # Convert structured array to unstructured array
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        
        # Perform clustering
        labels = self.clusterer.fit_predict(points)
        
        # Filter out noise points (label == -1)
        valid_indices = labels != -1
        points_filtered = points[valid_indices]
        labels_filtered = labels[valid_indices]
        
        # Reshape labels to match points shape for concatenation
        labels_column = labels_filtered.reshape(-1, 1)
        
        # Concatenate points with labels
        points_labeled = np.hstack((points_filtered, labels_column))
        
        # Convert labelled points to PointCloud2 format
        data = unstructured_to_structured(
            points_labeled,
            dtype=np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('label', np.int32)
            ])
        )
        
        # Create and publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header = msg.header  # Copy timestamp and frame_id from input message
        
        self.clusters_pub.publish(cluster_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer', anonymous=True)
    node = PointsClusterer()
    node.run()
