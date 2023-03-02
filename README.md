# Machine Learning & Data Analysis

## Project - Autonomous Driving and Obstacle avoidance using Machine Learning algorithm in ROS.

## Team Members:
- Jerin Joy (5061530)
- Koushikmani Maskalmatti Lakshman (5053566)

## Description of the Project:
The Turtlebot robot is a popular platform for robotics research and education. With its sensors, cameras, and actuators, it provides a great platform for testing and developing algorithms for robot navigation and control. One of the key challenges in robotics is autonomous driving and obstacle avoidance, which involves navigating in an environment with unknown obstacles and avoiding collisions. This project aims to address this challenge by using machine learning algorithms to train the Turtlebot robot to avoid obstacles.

To implement obstacle avoidance, we start by collecting data using the Turtlebot's laser range finder. This sensor measures the distance to obstacles in the environment and provides a 2D scan of the surroundings. We use this data to train various machine learning classifiers such as Random Forest, KNN, MLP, AdaBoost, Logistic Regression, Gradient Boosting, Extra Tree and Decision Trees. These classifiers learn to classify the environment into obstacle and non-obstacle regions based on the laser scan data. Once the classifiers are trained, we use them to implement obstacle avoidance in the Turtlebot robot. This allows the robot to navigate autonomously in an unknown environment, avoiding obstacles in its path. By using machine learning, we can create intelligent robots that can adapt to different environments and tasks, making them more useful and versatile.


## Installation Setup:
- OS Version:[Ubuntu 20.04] (https://releases.ubuntu.com/focal/)
- This package is developed with the ROS Noetic. To install ROS Noetic, follow the official documentation [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu)


## Classifiers used:
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- MLP Classifier
- Logistic Regression
- Gradient Boosting
- AdaBoost 
- Extra Tree

## Code Execution Process:
1. Clone the package into src folder of the ROS Workspace:
```
git clone https://github.com/jerin-joy/ml_obstacle_avoidance
```

2. Compile the workspace by navigating to the workspace and entering in terminal:
```
catkin_make
```
3. Source the workspace by entering:
```
source devel/setup.bash
```
4. Launch the Gazebo simulator with the Turtlebot robot and simulation environment by entering:
```
roslaunch turtlebot3_neural_network turtlebot3_square.launch
```
5. To perform the obstacle avoidance of the robot using Laser Range Finder, enter:
```
python obstacle_avoidance_laser.py
```
6. Open a new terminal to record the data from LRF and enter:
```
python data_recorder.py
```
7. Open 'ml_training_evaluation.ipynb' jupyter notebook file and run all blocks for the data training and model evaluation.
8. Test the Machine Learning model on TurtleBot robot, launch the gazebo world again and enter:
```
python obstacle_avoidance_ml.py
```
9. Select the classifier you need to use from the list provided in the terminal. 
   
