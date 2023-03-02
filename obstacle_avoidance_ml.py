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
        print("1. Decision Tree")
        print("2. Random Forest")
        print("3. K-Nearest Neighbors")
        print("4. MLP (Neural Network)")
        print("5. Logistic Regression")
        print("6. Gradient Boosting")
        print("7. AdaBoost")
        print("8. Extra Trees")
        choice = input()

        if choice == "1":
            model_path = "models/dt_model.joblib"
        elif choice == "2":
            model_path = "models/rf_model.joblib"
        elif choice == "3":
            model_path = "models/knn_model.joblib"
        elif choice == "4":
            model_path = "models/nn_model.joblib"
        elif choice == "5":
            model_path = "models/lr_model.joblib"
        elif choice == "6":
            model_path = "models/gb_model.joblib"
        elif choice == "7":
            model_path = "models/ada_model.joblib"
        elif choice == "8":
            model_path = "models/et_model.joblib"           
        else:
            print("Invalid choice, please try again.")
            continue

        # load selected model
        model = joblib.load(model_path)

        # run obstacle avoidance
        state_change_time = rospy.Time.now()

        while not rospy.is_shutdown():
            predictions = model.predict(laser_range)
            predictions = np.array([(1, 0) if x == 0 else (0, 1) for x in predictions]).reshape(-1, 2)
            action = np.argmax(predictions)

            twist = Twist()
            if action == 0:
                twist.linear.x = 0.3
            else:
                twist.angular.z = 0.3
            cmd_vel_pub.publish(twist)

            rate.sleep()
