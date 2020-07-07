#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as np
from fnn.fnn import FNN
from cmac.cmac import CMAC
from rldt.rl_dt_agent import RLDTAgent

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  
        self.right_arm_mirror = False
        self.robot_state = 0 # state detail see execution part
        self.red_blob_x = 0
        self.red_blob_y = 0
        self.change_red_blob_pos = True
        self.orange_blob_x = 0
        self.orange_blob_y = 0
        self.change_orange_blob_pos = True
        self.joint_to_change = ""
        self.joint_value_to_change = 0

        # neural network with 1 ReLU hidden layer, and output layer without activation
        self.nn = FNN(discrete=False, learning_rate=0.1, batch_size=1)
        self.nn.add_layer(2, 32, self.nn.ReLU, self.nn.RandnInitializer, self.nn.ZeroInitializer, 2, 0.001, "hidden1")
        # self.nn.add_layer(64, 32, self.nn.Sigmoid, self.nn.RandnInitializer, self.nn.ZeroInitializer, 2, 0.001, "hidden2")
        self.nn.add_layer(32, 2, self.nn.NoActivation, self.nn.RandnInitializer, self.nn.ZeroInitializer, 2, 0.001, "output")
        print(self.nn)
        self.nn.load_parameter("track_network")

        # CMAC, output layer without activation
        self.receptive_field = 5
        self.cmac_predictor = CMAC(2, 2, self.receptive_field, 50, lr=0.3)
        print(self.cmac_predictor)
        self.cmac_predictor.load_parameter("track_cmac")

        # RLDT agent
        self.agent = RLDTAgent(num_feature=3, action_space=["l", "r", "kick"], gamma=0.99, epsilon=0.0, desired_reward=10, step_max=10, gain_threshold=0.0, exploration_rate=0.0)
        if os.path.exists("RLDT_penalty_tree_0.json"):
            self.agent.load_agent("RLDT_penalty")
            print(self.agent.exploration_mode)


    # detail about instructions see keyboard_node help
    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + " I heard %s from key channel", data.data)
        
        if data.data == "armhome":
            self.robot_state = 1
            rospy.loginfo(rospy.get_caller_id() + " set arm to home position")

        elif data.data == "wave":
            self.robot_state = 2
            rospy.loginfo(rospy.get_caller_id() + " set arm to wave")

        elif data.data == "headhome":
            self.robot_state = 3
            rospy.loginfo(rospy.get_caller_id() + " set head to home position")

        elif data.data == "nod":
            self.robot_state = 4
            rospy.loginfo(rospy.get_caller_id() + " set head to nod")

        elif data.data == "armmirror":
            self.right_arm_mirror = not self.right_arm_mirror
            if self.right_arm_mirror:
                rospy.loginfo(rospy.get_caller_id() + " set right arm mirror start")
            else:
                rospy.loginfo(rospy.get_caller_id() + " set right arm mirror stop")

        elif data.data == "cvtest":
            self.robot_state = 5
            rospy.loginfo(rospy.get_caller_id() + " start cv test")    

        elif data.data[0:3] == "set" and data.data[3] != "r":
            self.robot_state = 6
            command = data.data.split()[1:]
            self.joint_to_change = command[0]
            self.joint_value_to_change = float(command[1])
            rospy.loginfo(rospy.get_caller_id() + " %s to %.4f position" % (self.joint_to_change, self.joint_value_to_change))

        elif data.data == "trackfnn":
            self.robot_state = 7
            rospy.loginfo(rospy.get_caller_id() + " start tracking red ball with fnn")

        elif data.data == "trackcmac":
            self.robot_state = 8
            rospy.loginfo(rospy.get_caller_id() + " start tracking red ball with CMAC")

        elif data.data[0:9] == "trackcmac":
            self.robot_state = 9
            self.change_red_blob_pos = False
            coord = data.data.split()[1:]
            self.red_blob_x = int(coord[0])
            self.red_blob_y = int(coord[1])
            rospy.loginfo(rospy.get_caller_id() + " start tracking red ball with CMAC, stop using real red blob, test coordiates [%d, %d]" % (self.red_blob_x, self.red_blob_y))

        elif data.data == "kick":
            self.robot_state = 10
            rospy.loginfo(rospy.get_caller_id() + " execute kick action")

        elif data.data == "penaltyready":
            self.robot_state = 11
            rospy.loginfo(rospy.get_caller_id() + " prepare to shoot penalty kick")

        elif data.data == "penaltydt":
            self.robot_state = 12
            rospy.loginfo(rospy.get_caller_id() + " shoot penalty kick with RL-DT")

        elif data.data[0:4] == "setr":
            self.robot_state = 13
            command = data.data.split()[1:]
            self.joint_to_change = command[0]
            self.joint_value_to_change = float(command[1])
            rospy.loginfo(rospy.get_caller_id() + " set %s to %.4f relatively" % (self.joint_to_change, self.joint_value_to_change))
            
    def learning_cb(self, data):
        rospy.loginfo(rospy.get_caller_id() + " I heard %s from learning channel", data.data)
        if data.data == "record":
            rospy.loginfo(rospy.get_caller_id() + " record position")
            self.change_red_blob_pos = True
            with open("training_x.txt", "a") as f:
                f.write("%d %d\n" % (self.red_blob_x, self.red_blob_y))
            with open ("training_y.txt", "a") as f:
                f.write("%f %f\n" % (self.joint_angles[2], self.joint_angles[3]))

        elif data.data[0:3] == "set" and data.data[3] != "r":
            self.robot_state = 6
            command = data.data.split()[1:]
            self.joint_to_change = command[0]
            self.joint_value_to_change = float(command[1])
            rospy.loginfo(rospy.get_caller_id() + " set %s to %.4f position" % (self.joint_to_change, self.joint_value_to_change))

        elif data.data[0:4] == "setr":
            self.robot_state = 13
            command = data.data.split()[1:]
            self.joint_to_change = command[0]
            self.joint_value_to_change = float(command[1])
            rospy.loginfo(rospy.get_caller_id() + " set %s to %.4f relatively" % (self.joint_to_change, self.joint_value_to_change))

        elif data.data == "kick":
            self.robot_state = 10
            rospy.loginfo(rospy.get_caller_id() + " execute kick action")

        elif data.data == "penaltyready":
            self.robot_state = 11
            rospy.loginfo(rospy.get_caller_id() + " prepare to shoot penalty kick")

        elif data.data == "state":
            rospy.loginfo(rospy.get_caller_id() + " send state info")
            index = self.joint_names.index("RHipRoll")
            RHipRoll = self.joint_angles[index]
            state = "%.4f %d %d" % (RHipRoll, self.orange_blob_x, self.red_blob_x)
            self.rlPub.publish(state)
            


    def joints_cb(self, data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False

    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

    def image_top_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        # print(cv_image.shape)
        # convert to HSV color space
        image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # two slices for red 
        red_high_upper = np.array([180, 255, 255])
        red_high_lower = np.array([170, 120, 120])
        red_low_upper = np.array([10, 255, 255])
        red_low_lower = np.array([0, 120, 120])
        # bit-mask for red area
        mask_high = cv2.inRange(image_hsv, red_high_lower, red_high_upper)
        mask_low = cv2.inRange(image_hsv, red_low_lower, red_low_upper)
        # combine them to one mask
        mask = cv2.addWeighted(mask_high, 1.0, mask_low, 1.0, 0.0)
        # erode and dilate to filter out noise
        mask = cv2.erode(mask, None)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours 
        _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            # find the contour with largest area
            max_contour = max(contours, key=cv2.contourArea)
            max_contour_moment = cv2.moments(max_contour)
            # obtain center coordinates
            center_x = int(max_contour_moment["m10"] / max_contour_moment["m00"])
            center_y = int(max_contour_moment["m01"] / max_contour_moment["m00"])            
            # only output coordinates if changed
            if (center_x != self.red_blob_x or center_y != self.red_blob_y) and self.change_red_blob_pos: 
                self.red_blob_x = center_x
                self.red_blob_y = center_y
                rospy.loginfo("New Red Blob Coordinates: %d, %d" % (center_x, center_y))
            # draw a cross at the center
            image_cross = cv2.drawMarker(cv_image, (self.red_blob_x, self.red_blob_y), (255, 255, 255), markerType=cv2.MARKER_CROSS)
        else:
            image_cross = cv_image    

        # cv2.imshow("TopCam", image_cross)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(5) # a small wait time is needed for the image to be displayed correctly

    def image_bottom_cb(self, data) :
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        # convert to HSV color space
        image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # one slice for orange 
        orange_upper = np.array([50, 255, 255])
        orange_lower = np.array([10, 60, 60])
        # bit-mask for orange area
        mask = cv2.inRange(image_hsv, orange_lower, orange_upper)
        # erode and dilate to filter out noise
        mask = cv2.erode(mask, None)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours 
        _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            # find the contour with largest area
            max_contour = max(contours, key=cv2.contourArea)
            max_contour_moment = cv2.moments(max_contour)
            # obtain center coordinates
            center_x = int(max_contour_moment["m10"] / max_contour_moment["m00"])
            center_y = int(max_contour_moment["m01"] / max_contour_moment["m00"])            
            # only output coordinates if changed
            if (center_x != self.orange_blob_x or center_y != self.orange_blob_y) and self.change_orange_blob_pos: 
                self.orange_blob_x = center_x
                self.orange_blob_y = center_y
                rospy.loginfo("New Orange Blob Coordinates: %d, %d" % (center_x, center_y))
            # draw a cross at the center
            image_cross = cv2.drawMarker(cv_image, (self.orange_blob_x, self.orange_blob_y), (255, 255, 255), markerType=cv2.MARKER_CROSS)
        else:
            image_cross = cv_image    

        # cv2.imshow("BottomCam", image_cross)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(5) # a small wait time is needed for the image to be displayed correctly

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def set_joint_angles(self, joint_name, head_angle):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_relative(self, joint_name, head_angle):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = True # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_quick(self, joint_name, head_angle, speed=1.0):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = speed # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
        
    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("learning", String, self.learning_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_top_cb)
        rospy.Subscriber("/nao_robot/camera/bottom/camera/image_raw",Image,self.image_bottom_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)
        self.rlPub = rospy.Publisher("rl_state", String, queue_size=10)

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        # each loop, robot acts according to the internal state
        while not rospy.is_shutdown():
            # state 1, arm back to home position, then stop
            if self.robot_state == 1:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                self.set_joint_angles("LShoulderPitch", 0.1)
                self.set_joint_angles("LElbowRoll", -0.6)
                self.set_joint_angles("LShoulderRoll", 0.1)
                self.set_joint_angles("LHand", 1)
                self.set_joint_angles("LWristYaw", -1.0)
                # right arm 
                self.set_joint_angles("RShoulderPitch", 0.8)
                self.set_joint_angles("RElbowRoll", 0.3)
                self.set_joint_angles("RShoulderRoll", -0.3)
                self.set_joint_angles("RHand", 0)
                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(2.0)
                self.set_stiffness(False)     

            # state 2, wave
            elif self.robot_state == 2:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                # wave forward
                self.set_joint_angles("LShoulderPitch", -1.0)
                self.set_joint_angles("LElbowRoll", -0.7)
                self.set_joint_angles("LShoulderRoll", 0.5)
                self.set_joint_angles("LHand", 1)
                # right arm mirror
                if self.right_arm_mirror:
                    self.set_joint_angles("RShoulderPitch", -1.0)
                    self.set_joint_angles("RElbowRoll", 0.7)
                    self.set_joint_angles("RShoulderRoll", -0.5)
                    self.set_joint_angles("RHand", 1)
                rospy.sleep(3.0)
                # wave backward
                self.set_joint_angles("LElbowRoll", -0.1)
                # right arm mirror
                if self.right_arm_mirror:
                    self.set_joint_angles("RElbowRoll", 0.1)
                rospy.sleep(2.0)
                self.set_stiffness(False)

            # state 3, head back to home position, then stop
            elif self.robot_state == 3:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                self.set_joint_angles("HeadPitch", 0.15)
                self.set_joint_angles("HeadYaw", 0.2)
                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(2.0)
                self.set_stiffness(False) 

            # state 4, head nod
            elif self.robot_state == 4:
                self.set_stiffness(True)
                rospy.sleep(1.0)
                # pitch up
                self.set_joint_angles("HeadPitch", -0.3)
                rospy.sleep(2.0)
                # pitch down
                self.set_joint_angles("HeadPitch", 0.3)
                rospy.sleep(2.0)
                self.set_stiffness(False) 

            # state 5, CV test
            elif self.robot_state == 5:
                self.set_stiffness(True)
                rospy.sleep(1.0)
                self.set_joint_angles("LShoulderPitch", 0.1)
                self.set_joint_angles("LElbowRoll", -1.0)
                self.set_joint_angles("LShoulderRoll", 0.1)
                self.set_joint_angles("LHand", 0)
                # right arm mirror
                if self.right_arm_mirror:
                    self.set_joint_angles("RShoulderPitch", 0.1)
                    self.set_joint_angles("RElbowRoll", 1.0)
                    self.set_joint_angles("RShoulderRoll", -0.1)
                    self.set_joint_angles("RHand", 0)
                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(2.0)
                self.set_stiffness(False)

            # state 6, set joint to specific position
            elif self.robot_state == 6:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                self.set_joint_angles(self.joint_to_change, self.joint_value_to_change)

                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(3.0)
                self.set_stiffness(False) 

            # state 7, track red ball with fnn
            elif self.robot_state == 7:
                # prepare input data from pixel
                x = np.array([self.red_blob_x  / 320.0, self.red_blob_y / 240.0]).reshape((-1, 1))
                # predict using pre-trained neural network
                y = self.nn.predict(x)
                lsp = y[0] # left shoulder pitch
                lsr = y[1] # left shoulder roll
                # ensure predicted values are valid
                # pitch [-1.5, 1.5]
                lsp = max(-1.5, lsp) 
                lsp = min(1.5, lsp)
                # roll [0, 1]
                lsr = max(0.0, lsr)
                lsr = min(1.0, lsr)
                self.set_stiffness(True)
                rospy.sleep(0.5)
                self.set_joint_angles("LShoulderPitch", lsp)
                self.set_joint_angles("LShoulderRoll", lsr)
                self.set_joint_angles("LHand", 1)

                rospy.sleep(1.0)
                self.set_stiffness(False) 

            # state 8, track red ball with CMAC
            elif self.robot_state == 8:
                # prepare input data from pixel
                x = np.array([self.red_blob_x  / 320.0, self.red_blob_y / 240.0]).reshape((-1, 1))
                # predict using pre-trained CMAC
                y = self.cmac_predictor.predict(x)
                lsp = y[0] * 2.0 - 1.0 # left shoulder pitch
                lsr = y[1] # left shoulder roll
                # ensure predicted values are valid
                # pitch [-1.0, 1.0]
                lsp = max(-1.0, lsp) 
                lsp = min(1.0, lsp)
                # roll [0, 1]
                lsr = max(0.0, lsr)
                lsr = min(1.0, lsr)
                self.set_stiffness(True)
                rospy.sleep(1.0)
                self.set_joint_angles("LShoulderPitch", lsp)
                self.set_joint_angles("LShoulderRoll", lsr)
                self.set_joint_angles("LHand", 1)

                rospy.sleep(2.0)
                self.set_stiffness(False) 

            # state 9, track red ball with CMAC, manually set blob position
            elif self.robot_state == 9:
                # prepare input data from pixel
                x = np.array([self.red_blob_x  / 320.0, self.red_blob_y / 240.0]).reshape((-1, 1))
                # predict using pre-trained CMAC
                y = self.cmac_predictor.predict(x)
                lsp = y[0] * 2.0 - 1.0 # left shoulder pitch
                lsr = y[1] # left shoulder roll
                # ensure predicted values are valid
                # pitch [-1.0, 1.0]
                lsp = max(-1.0, lsp) 
                lsp = min(1.0, lsp)
                # roll [0, 1]
                lsr = max(0.0, lsr)
                lsr = min(1.0, lsr)
                self.set_stiffness(True)
                rospy.sleep(1.0)
                self.set_joint_angles("LShoulderPitch", lsp)
                self.set_joint_angles("LShoulderRoll", lsr)
                self.set_joint_angles("LHand", 1)

                # set robot to no action state
                self.robot_state = 0
                self.change_red_blob_pos = True
                rospy.sleep(2.0)
                self.set_stiffness(False) 
            
            # state 10, execute kick action
            elif self.robot_state == 10:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                # self.set_joint_angles_quick("RAnkleRoll", 0.0)
                self.set_joint_angles_quick("RHipPitch", -0.5, 0.3)
                rospy.sleep(1.50)

                index = self.joint_names.index("RHipRoll")
                RHipRoll = self.joint_angles[index]
                if RHipRoll < -0.43:
                    self.set_joint_angles_quick("RAnklePitch", 0.5, 0.5)
                    self.set_joint_angles_quick("RAnkleRoll", 0.25, 0.5)
                else:
                    self.set_joint_angles_quick("RAnklePitch", 0.35, 0.5)
                    self.set_joint_angles_quick("RAnkleRoll", 0.1, 0.5)
                # elif RHipRoll < -0.39:
                #     self.set_joint_angles_quick("RAnklePitch", 0.40, 0.5)
                #     self.set_joint_angles_quick("RAnkleRoll", 0.2, 0.5)
                # elif RHipRoll < -0.37:
                #     self.set_joint_angles_quick("RAnklePitch", 0.25, 0.5)
                #     self.set_joint_angles_quick("RAnkleRoll", -0.1, 0.5)
                # else:
                #     self.set_joint_angles_quick("RAnklePitch", 0.2, 0.5)
                #     # self.set_joint_angles_quick("RAnkleRoll", -0.25)
                
                rospy.sleep(1.50)
                self.set_joint_angles_quick("RAnklePitch", 0.8)
                self.set_joint_angles_quick("RHipPitch", -1.5)
                self.set_joint_angles_quick("RKneePitch", 0.05)
                rospy.sleep(0.0)
                # self.set_joint_angles_quick("RAnklePitch", 0.8)
                rospy.sleep(2.0)

                self.set_joint_angles("RAnklePitch", -0.2)
                self.set_joint_angles("RKneePitch", 1.0)
                rospy.sleep(0.5)
                self.set_joint_angles("RHipPitch", 0.8)
                rospy.sleep(0.5)
                
                self.robot_state = 0
                rospy.sleep(1.0)

            # state 11, prepare to shoot penalty
            elif self.robot_state == 11:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                self.set_joint_angles("HeadPitch", 0.45)
                self.set_joint_angles("LShoulderRoll", 0.3)
                self.set_joint_angles("RShoulderRoll", -0.6)
                rospy.sleep(1.0)
                self.set_joint_angles("LShoulderPitch", 1.5)
                self.set_joint_angles("RShoulderPitch", 1.5)
                rospy.sleep(0.5)
                self.set_joint_angles("RAnkleRoll", 0.2)
                self.set_joint_angles("LAnkleRoll", 0.2)
                rospy.sleep(1.5)
                self.set_joint_angles("RAnkleRoll", 0.35)
                rospy.sleep(1.5)
                self.set_joint_angles("RHipRoll", -0.50)
                rospy.sleep(1.5)
                self.set_joint_angles("LHipRoll", -0.1)
                rospy.sleep(1.5)
                self.set_joint_angles("LAnkleRoll", 0.4)
                rospy.sleep(1.5)
                self.set_joint_angles("LHipRoll", -0.2)
                rospy.sleep(1.5)
                self.set_joint_angles("RHipPitch", 0.8)
                self.set_joint_angles("RAnklePitch", -0.2)
                rospy.sleep(1.0)
                # self.set_joint_angles("RAnkleRoll", 0.35)
                self.set_joint_angles("RKneePitch", 1.0)
                rospy.sleep(1.0)
                # self.set_joint_angles("LHipRoll", -0.25)
                # rospy.sleep(0.5)

                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(1.0)

            # state 12, use RL-DT to shoot penalty kick
            elif self.robot_state == 12:
                kicked = False
                step = 0
                while not kicked:
                    index = self.joint_names.index("RHipRoll")
                    RHipRoll_val = self.joint_angles[index]
                    ball_val = self.orange_blob_x
                    gk_val = self.red_blob_x
                    # RHipRoll from -0.4 to -0.6 in 0.02 interval
                    RHipRoll = int(round((RHipRoll_val + 0.4) / -0.02))
                    RHipRoll = max(0, RHipRoll)
                    RHipRoll = min(10, RHipRoll)
                    # ball x coordinate from 225 to 295 in 5 intervals
                    ball = int(round((ball_val - 225) / 14))
                    ball = max(0, ball)
                    ball = min(4, ball)
                    # goalkeeper x coordinate from 135 to 235 in 5 intervals
                    gk = int(round((gk_val - 135) / 20))
                    gk = max(0, gk)
                    gk = min(4, gk) 
                    s = [RHipRoll, ball, gk]
                    a = self.agent.choose_action(s, True)
                    if a == "l":
                        # setr RHipRoll 0.02
                        print("left")
                        if RHipRoll_val <= -0.41:
                            self.set_stiffness(True)
                            rospy.sleep(0.5)
                            self.set_joint_angles_relative("RHipRoll", 0.02)
                            rospy.sleep(1.0)
                            self.set_stiffness(False) 
                    elif a == "r":
                        print("right")
                        # setr RHipRoll -0.02
                        if RHipRoll_val >= -0.59:
                            self.set_stiffness(True)
                            rospy.sleep(0.5)
                            self.set_joint_angles_relative("RHipRoll", -0.02)
                            rospy.sleep(1.0)
                            self.set_stiffness(False) 
                    else:
                        # kick
                        print("kick")
                        kicked = True
                        self.robot_state = 10
                    step += 1
                    if step >= 20:
                        rospy.loginfo(rospy.get_caller_id() + " takes too long to prepare kick, abort")
                        self.robot_state = 0
                        break
            
            # state 13, set joint relative
            elif self.robot_state == 13:
                self.set_stiffness(True)
                rospy.sleep(0.5)
                self.set_joint_angles_relative(self.joint_to_change, self.joint_value_to_change)

                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(1.0)
                self.set_stiffness(False) 

            # no action
            else:
                self.set_stiffness(self.stiffness)

            rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
