#!/usr/bin/env python
import rospy
import random
import numpy as np
import joblib
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

laser_range = np.array([])

def scan_callback(msg):
    global laser_range
    laser_range = np.expand_dims(np.array(msg.ranges[:60] + msg.ranges[-60:]), axis=0)
    laser_range[laser_range == np.inf] = 3.5


if __name__ == "__main__":
    scan_sub = rospy.Subscriber('scan', LaserScan, scan_callback)
    cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    rospy.init_node('obstacle_avoidance')

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        # display menu options
        print("Choose a model to load:")
        print("1. Random Forest")
        print("2. Neural Network")
        print("3. Decision Tree")
        print("4. K-Nearest Neighbors")
        choice = input()

        if choice == "1":
            model_path = "model_random_forest.joblib"
        elif choice == "2":
            model_path = "model_neural_network.joblib"
        elif choice == "3":
            model_path = "model_decision_tree.joblib"
        elif choice == "4":
            model_path = "model_knn.joblib"
        else:
            print("Invalid choice, please try again.")
            continue

        # load selected model
        model = joblib.load(model_path)

        # run obstacle avoidance
        state_change_time = rospy.Time.now()

        while not rospy.is_shutdown():
            predictions = model.predict(laser_range)
            action = np.argmax(predictions)

            twist = Twist()
            if action == 0:
                twist.linear.x = 0.3
            else:
                twist.angular.z = 0.3
            cmd_vel_pub.publish(twist)

            rate.sleep()
