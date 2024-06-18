#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, PoseStamped, Pose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from nav2_simple_commander.robot_navigator import BasicNavigator
from rclpy.duration import Duration
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from transforms3d.euler import euler2quat
from time import sleep
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock
from sensor_msgs.msg import Image
import numpy as np
from bboxes_ex_msgs.msg import BoundingBox, BoundingBoxes
import subprocess
import os
from std_msgs.msg import Bool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class GolfBallPipeline(Node):

    def __init__(self):
        super().__init__('golf_ball_pipeline')

        callback_group_1 = MutuallyExclusiveCallbackGroup()
        callback_group_2 = MutuallyExclusiveCallbackGroup()
        callback_group_3 = MutuallyExclusiveCallbackGroup()


        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10, callback_group=callback_group_1)


        self.golf_bal_pose_sub = self.create_subscription(
            BoundingBoxes, '/yolov5/bounding_boxes', self.bbox_callback, 10, callback_group=callback_group_1)

        self.pick_state_pub = self.create_publisher(Bool, '/pick_state', 10, callback_group=callback_group_2)
        self.pick_state_sub = self.create_subscription(Bool, '/pick_state', 
                        self.pick_state_callback, 10, callback_group=callback_group_2)


        self.mug_state_publisher = self.create_publisher(Bool, '/go_to_mug', 10, callback_group=callback_group_1)
        self.mug_state_subscriber = self.create_subscription(Bool, "/go_to_mug", self.mug_state_callback, 10, callback_group=callback_group_1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.count = 0
        self.rotate_in_place = True
        self.in_place_rotation_speed = 0.1
        self.got_goal = False
        self.map_frame = 'map'
        self.robot_base_frame = 'base_link'
        self.navigator = BasicNavigator()
        self.spin_angle = 20 * np.pi/180.0
        self.current_golf_ball_poses = None
        self.megapose = False
        self.tmux = True
        self.picking_ball = False
        # self.mug_state_publisher = False

        self.received_poses = None
        self.last_goal = None

        self.at_home = False
        self.going_home = False
        self.cancel_goal = False

        self.lock = Lock()

        self.stopping_condition = 0
        self.going_to_mug = False

    def mug_state_callback(self, msg):
        self.going_to_mug = msg.data

    def pick_state_callback(self, msg):
        self.picking_ball = msg.data
        print("############### Pick state: #############", self.picking_ball)
    
    def bbox_callback(self, msg):
        if self.going_to_mug == True:

            i = 0
        
            xmax = msg.bounding_boxes[i].xmax
            xmin = msg.bounding_boxes[i].xmin
            ymax = msg.bounding_boxes[i].ymax
            ymin = msg.bounding_boxes[i].ymin

            width = (xmax-xmin) 
            height = (ymax-ymin) 

            centroid_x = xmin + (width/2)
            centroid_y = ymin + (height/2)
            # centroid.append([centroid_x, centroid_y])

            # if (height and width) < 55 and (height and width) > 10:
                
            bbox_values = [118, 22, 439, 359]
            x_max = bbox_values[2]
            width_fixed = (bbox_values[2] - bbox_values[0])
            height_fixed = (bbox_values[3] - bbox_values[1])

            # centroid_x_fixed = (bbox_values[0] + (width_fixed/2))
            centroid_y_fixed = (bbox_values[1] + (height_fixed/2))
            centroid_x_fixed = x_max

            if height < height_fixed - 60: #60
                print(f"height less than + 20. height:{height}, height_fixed:{height_fixed}")
                if  centroid_x <= (centroid_x_fixed + 50) and centroid_x >= (centroid_x_fixed - 50) and centroid_y <= (centroid_y_fixed + 20) and centroid_y >= (centroid_y_fixed - 20) and ((height) < (height_fixed - 10)):
                    print("Going straight in front with high speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.1 #0.05
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)
                elif centroid_x < (centroid_x_fixed - 50): #20
                    print("going left with high speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.1 #0.05
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x > (centroid_x_fixed + 50): #20
                    print("Going right")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = -0.1 #0.05
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y > (centroid_y_fixed + 20): #20
                    print("Going front")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.1 #0.07
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y < (centroid_y_fixed - 20): #20
                    print("Going back")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = -0.02
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

            elif height >= height_fixed - 60: #60
                print("Height more than + 20.")
                if  centroid_x <= (centroid_x_fixed + 7) and centroid_x >= (centroid_x_fixed - 7) and centroid_y <= (centroid_y_fixed + 10) and centroid_y >= (centroid_y_fixed - 10) and ((height) < (height_fixed - 5)):
                    print("Going straight in front with low speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.01
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x < (centroid_x_fixed - 7):
                    print("going left with low speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.03 #0.02
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x > (centroid_x_fixed + 7):
                    print("Going right with low speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = -0.03 #0.02
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y > (centroid_y_fixed + 10): 
                    print("Going front with low speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.01
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y < (centroid_y_fixed - 10):
                    print("Going back with low speed")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = -0.01
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                if  centroid_x <= (centroid_x_fixed + 7) and centroid_x >= (centroid_x_fixed - 7) and centroid_y <= (centroid_y_fixed + 10) and centroid_y >= (centroid_y_fixed - 10) and ((height) > (height_fixed - 5)):
                    print("Got stopping condition.")
                    print(f"centroid_x:{centroid_x}, centroid_x_fixed:{centroid_x_fixed}")
                    
                    if self.stopping_condition > 2:
                        self.stopping_condition = 0
                        twist_msg = Twist()
                        twist_msg.angular.x = 0.0
                        twist_msg.angular.y = 0.0
                        twist_msg.angular.z = 0.0
                        twist_msg.linear.x = 0.0
                        twist_msg.linear.y = 0.0
                        twist_msg.linear.z = 0.0
                        self.cmd_vel_pub.publish(twist_msg)
                        self.going_to_mug = False
                        # sleep(3)

                        print("Placing ball in the Mug.")
                        os.system('bash '+ './demo_place_golf_ball.sh')
                        # sleep(20)
                        mug_state = Bool()
                        mug_state.data = False
                        counter = 0
                        while True:
                            print("####################Publishing mug state is false.##########################")
                            self.mug_state_publisher.publish(mug_state)
                            print("Published mug state.******************************")
                            counter += 1
                            if counter > 30:
                                break

                        sleep(20)
                        print("20 second after out of while.")



                    self.stopping_condition += 1
            #else     
                

            else:
                print("No condition satisfied for mug.")

                ########################################## Code for golf ball #######################################################



        elif (self.going_to_mug == False) and (self.picking_ball == False):
            
            i = 0
        
            xmax = msg.bounding_boxes[i].xmax
            xmin = msg.bounding_boxes[i].xmin
            ymax = msg.bounding_boxes[i].ymax
            ymin = msg.bounding_boxes[i].ymin

            width = (xmax-xmin) 
            height = (ymax-ymin) 

            centroid_x = xmin + (width/2)
            centroid_y = ymin + (height/2)
            # centroid.append([centroid_x, centroid_y])

            # if (height and width) < 55 and (height and width) > 10:
                
            bbox_values = [360, 51, 497, 173] #[397, 54, 538, 176]
            width_fixed = (bbox_values[2] - bbox_values[0])
            height_fixed = (bbox_values[3] - bbox_values[1])

            centroid_x_fixed = (bbox_values[0] + (width_fixed/2))
            centroid_y_fixed = (bbox_values[1] + (height_fixed/2))

            if width < width_fixed - 20:
        
                if centroid_x <= (centroid_x_fixed + 50) and centroid_x >= (centroid_x_fixed-50) and centroid_y <= (centroid_y_fixed + 30) and centroid_y >= (centroid_y_fixed - 30) and (width and height) < (height_fixed - 10):
                    print("Going straight in front")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.1 #0.07
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x < (centroid_x_fixed - 50):
                    print("going left for ball")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.1 #0.05
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x > (centroid_x_fixed + 50):
                    print("Going right")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = -0.1 #0.05
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y > (centroid_y_fixed + 30): 
                    print("Going front")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.1 #0.07
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y < (centroid_y_fixed - 30):
                    print("Going back")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = -0.02
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                    self.stopping_condition += 1
                #else     
                    

                else:
                    print("No condition satisfied for ball.")

            if width >= width_fixed - 20:

                if centroid_x <= (centroid_x_fixed + 5) and centroid_x >= (centroid_x_fixed - 5) and centroid_y <= (centroid_y_fixed + 7) and centroid_y >= (centroid_y_fixed - 7) and (width ) < (width_fixed - 17):
                    print("Going straight in front")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.02
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x < (centroid_x_fixed - 5):
                    print("going left with slow speed for ball")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.03 #0.02
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x > (centroid_x_fixed + 5):
                    print("Going right")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = -0.03 #0.02
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y > (centroid_y_fixed + 7): 
                    print("Going front")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = 0.02
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_y < (centroid_y_fixed - 7):
                    print("Going back")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = -0.02
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)

                elif centroid_x < (centroid_x_fixed + 5) and centroid_x > (centroid_x_fixed - 5) and centroid_y < (centroid_y_fixed + 7) and centroid_y > (centroid_y_fixed - 7) and (width )  > (width_fixed - 17):
                    print("Got stopping condition.")
                    
                    if self.stopping_condition > 2:
                        self.stopping_condition = 0
                        twist_msg = Twist()
                        twist_msg.angular.x = 0.0
                        twist_msg.angular.y = 0.0
                        twist_msg.angular.z = 0.0
                        twist_msg.linear.x = 0.0
                        twist_msg.linear.y = 0.0
                        twist_msg.linear.z = 0.0
                        self.cmd_vel_pub.publish(twist_msg)
                        if self.tmux is True:
                            print("running bash script.")
                            self.tmux = False
                            self.picking_ball = True
                            current_pick_state = Bool()
                            current_pick_state.data = self.picking_ball
                            self.pick_state_pub.publish(current_pick_state)
                            # os.system('bash '+ './megapose.sh')
                            #sleep(35)
                            os.system('bash '+ './demo_pick_golf_ball.sh')
                            sleep(15)
                            i = 0
                            while(i<30):
                                print("in While after picking.")
                                self.picking_ball = False
                                current_pick_state = Bool()
                                current_pick_state.data = self.picking_ball
                                self.pick_state_pub.publish(current_pick_state)
                                self.going_to_mug = True
                                go_to_mug_msg = Bool()
                                go_to_mug_msg.data = self.going_to_mug
                                self.mug_state_publisher.publish(go_to_mug_msg)
                                i+=1
                            

                        else:
                            #sleep(35)
                            os.system('bash '+ './demo_pick_golf_ball.sh')
                            sleep(15)
                            i = 0
                            while(i<30):
                                print("in While after picking.")
                                self.picking_ball = False
                                current_pick_state = Bool()
                                current_pick_state.data = self.picking_ball
                                self.pick_state_pub.publish(current_pick_state)
                                self.going_to_mug = True
                                go_to_mug_msg = Bool()
                                go_to_mug_msg.data = self.going_to_mug
                                self.mug_state_publisher.publish(go_to_mug_msg)
                                i+=1

                    self.stopping_condition += 1
                #else     
                    

                else:
                    print("No condition satisfied for ball.")


    def get_robot_pose(self):
        while 1:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.robot_base_frame,
                    rclpy.time.Time())
                break
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform {self.map_frame} to {self.robot_base_frame}: {ex}')
                sleep(1)
        return t

    def go_to_home(self):

        t = self.get_robot_pose()

        distance = np.sqrt(t.transform.translation.x**2 +
                           t.transform.translation.y**2)
        if distance < 0.3:
            self.get_logger().info("Robot is already at home position.")
            return

        # if self.got_goal is True:
        #     self.get_logger().info("Robot picking golf ball.")
        #     return

        # if self.at_home is False:
        self.get_logger().info("Robot going to home position.")
        home_pose = Pose()
        home_pose.position.x = 0.0
        home_pose.position.y = 0.0
        home_pose.orientation.w = 1.0
        home_pose.orientation.z = 0.0
        with self.lock:
            self.going_home = True
            self.send_goal(home_pose, keep_orientation=True, add_offset=False)
            self.going_home = False
        # self.at_home = True

    def get_closest_goal(self, poses):
        t = self.get_robot_pose()
        distances = [np.sqrt((t.transform.translation.x - pose.position.x)**2 +
                             (t.transform.translation.y - pose.position.y)**2) for pose in poses]
        return distances

    def golf_ball_pose_callback(self, msg):

        # if self.going_home is True:
        #     self.get_logger().info('Cancelling Home Goal!')
        #     self.cancel_goal = True
        #     self.going_home = False
        #     # self.at_home = False
        #     return

        if self.got_goal is False:
            self.got_goal = True
            self.get_logger().info(f'Pose :: {msg.poses}')

            goal_distances = self.get_closest_goal(msg.poses)
            self.get_logger().info(f'Distances :: {goal_distances}')
            if np.min(goal_distances) > 0.01:
                with self.lock:
                    self.send_goal(msg.poses[np.argmin(goal_distances)])
            
            self.got_goal = False
            # self.at_home = False
            sleep (9.0)
        else:
            sleep(0.5)

    def send_goal(self, pose, keep_orientation=False, add_offset=False):
        twist_msg = Twist()
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)

        # This requires a localizer node
        # self.navigator.waitUntilNav2Active(localizer="None")
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.map_frame
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()

        t = self.get_robot_pose()
        theta = np.arctan2(pose.position.y - t.transform.translation.y,
                        pose.position.x - t.transform.translation.x)
        
        if add_offset is True:
            offset_x = 0.2 * np.cos(theta)
            offset_y = 0.2 * np.sin(theta)
        else:
            offset_x = 0.0
            offset_y = 0.0

        if keep_orientation is True:
            w = pose.orientation.w
            z = pose.orientation.z
        else:
            w, x, y, z = euler2quat(0, 0, theta)


        goal_pose.pose.position.x = pose.position.x + 0.0 - offset_x
        goal_pose.pose.position.y = pose.position.y - offset_y
        goal_pose.pose.orientation.w = w
        goal_pose.pose.orientation.z = z

        self.rotate_in_place = False
        self.navigator.goToPose(goal_pose)
        i = 0
        while not self.navigator.isTaskComplete():
            i = i + 1
            feedback = self.navigator.getFeedback()
            if feedback and i % 5 == 0:
                print('Estimated time of arrival: ' + '{0:.0f}'.format(
                    Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9)
                    + ' seconds.')

            if self.cancel_goal is True:
                self.navigator.cancelTask()

        # Do something depending on the return code
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print('Goal succeeded!')
            sleep(10) # Only delay if goal succeeded
        elif result == TaskResult.CANCELED:
            print('Goal was canceled!')
            self.cancel_goal = False
            sleep(3)
        elif result == TaskResult.FAILED:
            print('Goal failed!')
        else:
            print('Goal has an invalid return status!')


if __name__ == '__main__':
    rclpy.init()
    node = GolfBallPipeline()
    # rclpy.spin(node)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()
