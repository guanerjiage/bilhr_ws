#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  
        self.right_arm_mirror = False
        self.robot_state = 0 # 0: no action, 1: arm home position, 2: wave, 3: head home position, 4: head nod, 5: arm to middle

    # b: arm to home position, w: wave, r: right arm mirror start/stop, a: head to home position, n: head nod, t: arm to middle (cv test)
    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        if data.data == "b":
            self.robot_state = 1
            rospy.loginfo(rospy.get_caller_id() + " set arm to home position")

        if data.data == "w":
            self.robot_state = 2
            rospy.loginfo(rospy.get_caller_id() + " set arm to wave")

        if data.data == "r":
            self.right_arm_mirror = not self.right_arm_mirror
            if self.right_arm_mirror:
                rospy.loginfo(rospy.get_caller_id() + " right arm mirror start")
            else:
                rospy.loginfo(rospy.get_caller_id() + " right arm mirror stop")

        if data.data == "a":
            self.robot_state = 3
            rospy.loginfo(rospy.get_caller_id() + " set head to home position")

        if data.data == "n":
            self.robot_state = 4
            rospy.loginfo(rospy.get_caller_id() + " set head to nod")

        if data.data == "t":
            self.robot_state = 5
            rospy.loginfo(rospy.get_caller_id() + " set cv test")

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False

    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        # convert to HSV color space
        image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # two slices for red 
        red_high_upper = np.array([180, 255, 255])
        red_high_lower = np.array([170, 192, 192])
        red_low_upper = np.array([10, 255, 255])
        red_low_lower = np.array([0, 192, 192])
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
            # draw a cross at the center
            image_cross = cv2.drawMarker(cv_image, (center_x, center_y), (255, 255, 255), markerType=cv2.MARKER_CROSS)
        else:
            image_cross = cv_image        
        
        cv2.imshow("image window", image_cross)
        cv2.waitKey(5) # a small wait time is needed for the image to be displayed correctly

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
        


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        # each loop, robot acts according to the internal state
        while not rospy.is_shutdown():
            # state 1, arm back to home position, then stop
            if self.robot_state == 1:
                self.set_stiffness(True)
                rospy.sleep(1.0)
                self.set_joint_angles("LShoulderPitch", 0.1)
                self.set_joint_angles("LElbowRoll", -0.3)
                self.set_joint_angles("LShoulderRoll", 0.1)
                self.set_joint_angles("LHand", 0)
                # right arm mirror
                if self.right_arm_mirror:
                    self.set_joint_angles("RShoulderPitch", 0.1)
                    self.set_joint_angles("RElbowRoll", 0.3)
                    self.set_joint_angles("RShoulderRoll", -0.1)
                    self.set_joint_angles("RHand", 0)
                # set robot to no action state
                self.robot_state = 0
                rospy.sleep(2.0)
                self.set_stiffness(False)     

            # state 2, wave
            elif self.robot_state == 2:
                self.set_stiffness(True)
                rospy.sleep(1.0)
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
                rospy.sleep(1.0)
                self.set_joint_angles("HeadPitch", 0.15)
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
