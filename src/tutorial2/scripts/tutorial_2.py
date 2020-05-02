#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False 
        self.instruction = 'h' 
        self.follow_flag = 'n'

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def instruction_cb(self,data):
        if(data.data == 'f' or data.data == 'n'):
            self.follow_flag = data.data
        elif(data.data == 'r' or data.data == 'h' or data.data == 'd'):
            self.instruction = data.data

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
        
        cv2.imshow("image window",cv_image)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)

    def set_joint_angles(self, names, head_angles):
        joint_angles_to_set = JointAnglesWithSpeed()
        for i in range(len(names)): 
            joint_angles_to_set.joint_names.append(names[i]) 
            # each joint has a specific name, look into the joint_state topic or google
            joint_angles_to_set.joint_angles.append(head_angles[i]) 
            # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
        
    def set_home_position(self):
        print "home position"
        self.set_stiffness(True) # don't let the robot stay enabled for too long, the motors will overheat!! (don't go for lunch or something)
        rospy.sleep(1.0)
        if(self.follow_flag=='f'):
            print "home with follow"
            names = ["LShoulderPitch", "RShoulderPitch"]
            angles = [0.5, 0.5]
            self.set_joint_angles(names, angles)
        else:
            self.set_joint_angles(["LShoulderPitch"],[0.5])
        self.set_stiffness(False)

    def repetitive_move(self):
        print "I am here repetitive move!!"
        self.set_stiffness(True) 
        count = 0
        while(self.instruction=='r' ):
            if(self.follow_flag=='n'):
                rospy.sleep(1.0)
                self.set_joint_angles(["LShoulderPitch"],[count*-0.5])
            elif(self.follow_flag=='f'):
                rospy.sleep(1.0)
                names = ["LShoulderPitch", "RShoulderPitch"]
                angles = [count*-0.5, count*-0.5]
                self.set_joint_angles(names, angles)
            count = (count+1)%2
        self.set_home_position()
        self.set_stiffness(False)

    def dance(self):
        self.set_stiffness(True) 
        count = 0
        
        names0 = ["HeadPitch","LShoulderPitch", "RShoulderPitch","LShoulderRoll", "RShoulderRoll","LElbowRoll","RElbowRoll"]
        angles0 = [-0.5, 0, 0, 0, 0, 0, 0]
        names1 = ["HeadPitch", "HeadYaw", "LShoulderPitch", "RShoulderPitch","LShoulderRoll", "RShoulderRoll","LElbowRoll","RElbowRoll"]
        angles1 = [0.5,        -0.5,             -0.5,            0.5,              1,           0.3,            0,            1.5]
        names2 = ["HeadPitch", "HeadYaw", "LShoulderPitch", "RShoulderPitch","LShoulderRoll", "RShoulderRoll","LElbowRoll","RElbowRoll"]
        angles2 = [0.5,        0.5,             0.5,            -0.5,             -0.3,            -1,           -1.5,         0]
        names3 = ["HeadYaw","LShoulderPitch", "RShoulderPitch","LShoulderRoll", "RShoulderRoll","LElbowRoll","RElbowRoll"]
        angles3 = [ -1, -1, 1, -1, 0, 0]
        names4 = ["LShoulderPitch", "RShoulderPitch","LShoulderRoll", "RShoulderRoll","LElbowRoll","RElbowRoll"]
        angles4 = [ 1, 0, -1, -0.5]
        names = [names1, names2]
        angles = [angles1, angles2]

        while(self.instruction=='d' ):
            self.set_joint_angles(names[count], angles[count])
            count = (count+1)%2
            rospy.sleep(4.0)

        self.set_home_position()
        self.set_stiffness(False)

    def symetric_move(self):
        print "symetirc move"
        self.set_stiffness(True) 
        count = 0

        while(self.instruction=='f'):
            rospy.sleep(1.0)
            name = ["LShoulderRoll", "RShoulderRoll"]
            angle = [count*-0.5, count*-0.5]
            self.set_joint_angles(name, angle)
            count = (count+1)%2
        self.set_home_position()

        self.set_stiffness(False)


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        rospy.Subscriber("instruction", String, self.instruction_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)
        rate = rospy.Rate(10) # sets the sleep time to 10ms
        
        while not rospy.is_shutdown():
            if self.instruction=='h':
                self.set_home_position()
            elif self.instruction=='r':
                self.repetitive_move()
            elif self.instruction=='d':    
                self.dance()
            else:
                self.set_home_position()

        self.set_stiffness(self.stiffness)
        rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
