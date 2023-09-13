#!/usr/bin/env python3

# Import the necessary libraries
import os

import numpy as np
import rospy  # Python library for ROS
from sensor_msgs.msg import Image  # Image is the message type

# Package to convert between ROS and OpenCV Images
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from metrics_refbox_msgs.msg import ObjectDetectionResult, Command

from ultralytics import YOLO

class object_detection():
    def __init__(self) -> None:
        rospy.loginfo("Object Detection node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        self.clip_size = 2  # manual number
        self.stop_sub_flag = False
        self.cnt = 0

        # yolo model config
        self._rospack = rospkg.RosPack()
        self.pkg_path = self._rospack.get_path('object_detection')
        self.model = YOLO(os.path.join(self.pkg_path, 'yolov8n.pt'))
        
        self.confidence_threshold = 0.5


        # publisher
        self.output_bb_pub = rospy.Publisher(
            "/metrics_refbox_client/object_detection_result", ObjectDetectionResult, queue_size=10)

        # subscriber
        self.requested_object = None
        self.referee_command_sub = rospy.Subscriber(
            "/metrics_refbox_client/command", Command, self._referee_command_cb)
        
        
        
        self.object_detection_msg = ObjectDetectionResult()
        self.object_detection_msg.message_type = ObjectDetectionResult.RESULT
        self.object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
        self.object_detection_msg.object_found = False

        # waiting for referee box to be ready
        rospy.loginfo("Waiting for referee box ...")

    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None

        """
        try:
            if not self.stop_sub_flag:

                # convert ros image to opencv image
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                if self.image_queue is None:
                    self.image_queue = []

                self.image_queue.append(cv_image)
                # print("Counter: ", len(self.image_queue))

                if len(self.image_queue) > self.clip_size:
                    # Clip size reached
                    # print("Clip size reached...")
                    rospy.loginfo("Image received..")

                    self.stop_sub_flag = True

                    # pop the first element
                    self.image_queue.pop(0)

                    # deregister subscriber
                    self.image_sub.unregister()

                    # call object inference method
                    self.object_inference()

        except CvBridgeError as e:
            rospy.logerr(
                "Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            self._check_failure()
            return

    def object_inference(self):

        rospy.loginfo("Object Inferencing Started...")

        opencv_img = self.image_queue[0]

        #####################
        # YOLO inferencing
        #####################
        self.results = self.model.predict(opencv_img, classes=[41])

    
        # Only publish the target object requested by the referee
        label = None
        print("---------------------------")
   
        # detection
        b = self.results[0].boxes.xyxy.cpu().numpy().astype(int)[0]   # box with xyxy format, (N, 4)
        conf = self.results[0].boxes.conf.cpu().numpy()[0]   # confidence score, (N, 1)
        class_id = self.results[0].boxes.cls.cpu().numpy().astype(int)[0]  # cls, (N, 1)
        print(b, conf,class_id)
        label = self.model.names[int(class_id)]

        if label:
        # Referee output message publishing
        
            self.object_detection_msg.message_type = ObjectDetectionResult.RESULT
            self.object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
            self.object_detection_msg.object_found = True
            self.object_detection_msg.box2d.min_x = b[0]
            self.object_detection_msg.box2d.min_y = b[1]
            self.object_detection_msg.box2d.max_x = b[2]
            self.object_detection_msg.box2d.max_y = b[3]

        # convert OpenCV image to ROS image message
        ros_image = self.cv_bridge.cv2_to_imgmsg(
            self.image_queue[0], encoding="passthrough")
        self.object_detection_msg.image = ros_image

        # publish message
        rospy.loginfo("Publishing result to referee...")
        self.output_bb_pub.publish(self.object_detection_msg)


        # ready for next image
        self.stop_sub_flag = False
        self.image_queue = []


    def _referee_command_cb(self, msg):

        # Referee comaand message (example)
        '''
        task: 1
        command: 1
        task_config: "{\"Target object\": \"Cup\"}"
        uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
        '''

        # START command from referee
        if msg.task == 1 and msg.command == 1:

            print("\nStart command received")

            self.image_sub = rospy.Subscriber("/camera/color/image_raw",
                                              Image,
                                              self._input_image_cb)

            # extract target object from task_config
            self.requested_object = msg.task_config.split(":")[
                1].split("\"")[1]
            print("\n")
            print("Requested object: ", self.requested_object)
            print("\n")

        # STOP command from referee
        if msg.command == 2:
            self.stop_sub_flag = True
            self.image_sub.unregister()
            rospy.loginfo("Received stopped command from referee")
            rospy.loginfo("Subscriber stopped")


if __name__ == "__main__":
    rospy.init_node("object_detection_node")
    object_detection_obj = object_detection()

    rospy.spin()
