import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()
import argparse
import numpy as np
import cv2
import sys
sys.path.append("home/adi-robotics-kit/anaconda3/envs/megapose_ros/lib/python3.8/site-packages/webdataset")
sys.path.append("/home/adi-robotics-kit/Downloads/megapose6d/src/megapose/scripts/OpenManipulator/src/poses_to_topic/poses_to_topic")
sys.path.append("/home/adi-robotics-kit/anaconda3/envs/megapose/lib/python3.8/site-packages/")
sys.path.append("/home/adi-robotics-kit/Downloads/megapose6d")

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from bboxes_ex_msgs.msg import BoundingBox, BoundingBoxes
from std_msgs.msg import Int16, Bool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock
from time import sleep
import time


class BboxPublisher(Node):

    def __init__(self):
        super().__init__('bbox_publisher')


        callback_group_1 = MutuallyExclusiveCallbackGroup()
        callback_group_2 = MutuallyExclusiveCallbackGroup()

        self.bbox_publisher= self.create_publisher(BoundingBoxes, '/yolov5/bounding_boxes', 10)        
        
        self.rgb_img_subscriber = self.create_subscription(Image, '/camera/camera/color/image_raw', 
                                                       self.bbox_callback, 10, callback_group=callback_group_1)

        self.depth_img_subscriber = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', 
                                                       self.bbox_callback, 10, callback_group=callback_group_1)

        self.id_subscriber = self.create_subscription(Int16, '/class_id', 
                                                       self.id_callback, 10, callback_group=callback_group_2)

        self.inference_publisher = self.create_publisher(Image, '/yolo/image_raw', 10)

        self.mug_state_subscriber = self.create_subscription(Bool, "/go_to_mug", self.mug_state_callback, 
                                                        10, callback_group=callback_group_2)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10, callback_group=callback_group_1)
        
        print('------Loading YoloV8 model for object detection-----')
        from ultralytics import YOLO

        self.model = YOLO('/home/adi-robotics-kit/Downloads/megapose6d/src/megapose/scripts/weights/best.pt')
        print('------Model Loaded-----')

        self.class_id = None
        self.going_to_mug = None
        self.going_back = True
        self.found_golf_ball = False
        self.found_golf_ball_count = 0

    def mug_state_callback(self, msg):
        self.going_to_mug = msg.data
        print("#############################################")
        print("#############################################")
        print("#############################################")
        print("#############################################")
        print("Mug state:",msg.data)
#

    def get_bbox(self, im):

        print("Class ID:", self.class_id)
        # print("Hola")
        if self.going_to_mug == True:
            if self.going_back == True:
                t1 = time.time()
                d = 0.0
                while True:
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.0
                    twist_msg.linear.x = -0.1
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0                
                    self.cmd_vel_pub.publish(twist_msg)
                    t2 = time.time()
                    d = float(t2 - t1)
                    if d > 4.0:
                        break
                self.going_back = False


            results = self.model(im, classes=[3], max_det=1, conf=0.7) #self.class_id
            #results = self.model(im, max_det=1)
            # print("RESULT OF DETECTION: ", results)

            if not results[0]:
                if self.found_golf_ball is False:
                    print("result is none.")
                    twist_msg = Twist()
                    twist_msg.angular.x = 0.0
                    twist_msg.angular.y = 0.0
                    twist_msg.angular.z = 0.35
                    twist_msg.linear.x = 0.0
                    twist_msg.linear.y = 0.0
                    twist_msg.linear.z = 0.0
                    self.cmd_vel_pub.publish(twist_msg)
                    print('No bbox')
        
            elif results[0]:

                #self.found_golf_ball = True
                # self.found_golf_ball_count += 1
                cls_id = int(results[0].boxes.cls.to('cpu').numpy()[0])
                # print(cls_id.dtype, cls_id.shape)
                
                prob = float(results[0].boxes.conf.to('cpu').numpy()[0])
                bbox = results[0].boxes.xyxy.to('cpu').numpy().squeeze()
                
                x_min = None
                y_min = None
                x_max = None
                y_max = None


                if len(bbox) > 0:
                    print('bbox', bbox)
                    # print('Calculate min max')
                    x_min = int(bbox[0])
                    y_min = int(bbox[1])
                    x_max = int(bbox[2])
                    y_max = int(bbox[3])
                    width = x_max - x_min
                    height = y_max - y_min
                    #self.timer = self.create_timer(0.5, self.bbox_callback(self.color_image))
                    x_centroid = (width/2) + x_min
                    y_centroid = (height/2) + y_min
                    print(f'x_cent: {x_centroid }, y_cent: {y_centroid}')
                    print(f'width of bbox: {width}')
                    print(f'height of bbox: {height}')


                    bbox_dict = {"Bounding_box":bbox, "Class_id": results[0].names[cls_id], "Probability": prob,"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
                    print(type(bbox))
                    return bbox_dict
                    

            # else:
            #     print('No bbox')

        else:
            if self.class_id is not None:
                results = self.model(im, classes=[self.class_id], max_det=1, conf=0.7) #self.class_id
                #results = self.model(im, max_det=1)
                # print("RESULT OF DETECTION: ", results)

            
                if results[0]:
                    self.going_back = True

                    cls_id = int(results[0].boxes.cls.to('cpu').numpy()[0])
                    # print(cls_id.dtype, cls_id.shape)
                    
                    prob = float(results[0].boxes.conf.to('cpu').numpy()[0])
                    bbox = results[0].boxes.xyxy.to('cpu').numpy().squeeze()
                    
                    x_min = None
                    y_min = None
                    x_max = None
                    y_max = None


                    if len(bbox) > 0:
                        print('bbox', bbox)
                        # print('Calculate min max')
                        x_min = int(bbox[0])
                        y_min = int(bbox[1])
                        x_max = int(bbox[2])
                        y_max = int(bbox[3])
                        width = x_max - x_min
                        height = y_max - y_min
                        #self.timer = self.create_timer(0.5, self.bbox_callback(self.color_image))
                        x_centroid = (width/2) + x_min
                        y_centroid = (height/2) + y_min
                        print(f'x_cent: {x_centroid }, y_cent: {y_centroid}')
                        print(f'width of bbox: {width}')
                        print(f'height of bbox: {height}')


                        bbox_dict = {"Bounding_box":bbox, "Class_id": results[0].names[cls_id], "Probability": prob,"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
                        print(type(bbox))
                        return bbox_dict
                    

            else:
                print('No mug detected.')

            return None

    def id_callback(self, msg):
        self.class_id = msg.data
        print("Class id in callback:", self.class_id)   

    def bbox_callback(self, msg):
        bbox_msg = BoundingBox()
        color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # print(color_image[0])

        # print(color_image)
        # color_image = np.reshape(color_image,(msg.height, msg.width))
        bbox_dict = self.get_bbox(color_image)

        if bbox_dict is not None:
            bbox_msg.xmin = bbox_dict['x_min']
            bbox_msg.ymin = bbox_dict['y_min']
            bbox_msg.xmax = bbox_dict['x_max']
            bbox_msg.ymax = bbox_dict['y_max']
            bbox_msg.probability = bbox_dict['Probability']
            bbox_msg.class_id = bbox_dict["Class_id"] 

            bboxes_msg = BoundingBoxes()
            # print("hi")
            bboxes_msg.header.stamp = self.get_clock().now().to_msg()
            bboxes_msg.header.frame_id = 'camera_link'
            bboxes_msg.bounding_boxes.append(bbox_msg)
            print(bboxes_msg)
            if len(bboxes_msg.bounding_boxes) > 0:
                self.bbox_publisher.publish(bboxes_msg)


            start_point = (bbox_msg.xmin, bbox_msg.ymin)
            end_point = (bbox_msg.xmax, bbox_msg.ymax)
            cv2.rectangle(color_image, start_point, end_point, color=(0,0,0), thickness=2)

            
            cv2.putText(
                color_image,
                bbox_msg.class_id,
                (bbox_msg.xmin ,bbox_msg.ymin  - 10),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (0, 0, 0),
                thickness=2
            )
            cv2.imwrite("example_with_bounding_boxes.jpg", color_image)
            color_image = bridge.cv2_to_imgmsg(color_image,"bgr8") 
            self.inference_publisher.publish(color_image)
            



def main(args=None):
    rclpy.init(args=args)
    bbox_publisher = BboxPublisher()
    executor = MultiThreadedExecutor()
    executor.add_node(bbox_publisher)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
