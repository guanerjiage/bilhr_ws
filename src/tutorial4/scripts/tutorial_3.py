#!/usr/bin/env python
import rospy
import numpy as np
from random import randint
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pickle

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False 
        self.instruction = 'b' 
        self.follow_flag = 'n'
        self.red_bolb_x = 0
        self.red_bolb_y = 0
        self.count = 0
        
        with open('/home/gejg/ros/bioinspired_ws/src/tutorial4/scripts/model.pkl') as f: 
            self.model = pickle.load(f)
        self.pos2idx = self.model[0]
        self.weights = self.model[1]
        self.neurons = self.model[2]
        #print self.pos2idx
        #print self.weights
        #print self.neurons
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

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # detect red color
        
        red_high_upper = np.array([180, 255, 255])
        red_high_lower = np.array([170, 110, 30])
        red_low_upper = np.array([10, 255, 255])
        red_low_lower = np.array([0, 110, 30])
        mask_high = cv2.inRange(hsv_image, red_high_lower, red_high_upper)
        mask_low = cv2.inRange(hsv_image, red_low_lower, red_low_upper)
        mask = cv2.bitwise_or(mask_high, mask_low )
        

        lower_green = np.array([50,100,30])
        upper_green = np.array([80,255,255])
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        color_detect = cv2.bitwise_and(hsv_image, hsv_image, mask = mask_green)
        blur = cv2.GaussianBlur(color_detect,(5,5),0)
        
        # find the biggest red blob
        _, _, v = cv2.split(blur)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(v)
        if(len(keypoints)>0):
            max_blob = keypoints[0]
            for keypoint in keypoints:
                if keypoint.size>max_blob.size:
                    max_blob = keypoint
            # added on the image                
            self.red_bolb_x = int(max_blob.pt[0])
            self.red_bolb_y = int(max_blob.pt[1])
            cv_image = cv2.drawMarker(cv_image, (self.red_bolb_x, self.red_bolb_y), (255, 255, 255), markerType=cv2.MARKER_CROSS)
        #cv2.imshow("gray",v)
        cv2.imshow("image window",cv_image)
        
        cv2.waitKey(4)

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
        print "I am here repetitive move!!"+str(self.count)
        self.set_stiffness(True) 
        if(self.follow_flag=='n'):
            rospy.sleep(2.0)
            r1=randint(0,70)
            r2=randint(0,100)
            self.set_joint_angles(["LShoulderPitch", "HeadYaw","LShoulderRoll"],[r1*-0.01, 0.8,-0.3+r2*0.01])
        elif(self.follow_flag=='f'):
            rospy.sleep(2.0)
            names = ["LShoulderPitch", "RShoulderPitch", "HeadYaw"]
            angles = [self.count*-0.5, self.count*-0.5, 0.5]
            self.set_joint_angles(names, angles)
        self.count = self.count+1
        #self.set_home_position()
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
    
    
    def catch_ball(self):
        print "I am here catch the ball!!"
        self.set_stiffness(True) 
        x = (float(self.red_bolb_x)/320)*0.99+0.01
        y = (float(self.red_bolb_y)/240)*0.99+0.01
        print x,y
        index = self.get_index([x,y])
        print index
        pred = self.forward(index)
        pred = pred[0]
        pred[0] = pred[0]*0.7-0.7
        pred[1] = pred[1]-0.3
        self.set_joint_angles(["LShoulderPitch", "HeadYaw","LShoulderRoll"],[pred[0], 0.8, pred[1]])
        self.set_stiffness(False) 

    def forward(self,index):
        x0=0
        x1=0
        for i in index:
            self.neurons[i]=1
            x0+=self.neurons[i]*self.weights[i][0]
            x1+=self.neurons[i]*self.weights[i][1]
        pred = np.zeros((1,2))    
        pred[0][0]=x0
        pred[0][1]=x1
        #print pred
        return pred
    
    def get_index(self,inputs):
        pos = self.active_neuro_pos(inputs)
        index = []
        for p in pos:
            i = self.pos2idx[str(p[0])+","+str(p[1])]
            index.append(i)
        return index

   

    def active_neuro_pos(self,inputs):
        resolution = 50
        field_size = 3.0
        RFpos = [[0,0],[1,1],[2,2]]
        Position = []
        for r in range(len(RFpos)):
            Coord = []
            for c in range(len(inputs)):
                input_index = round(inputs[c]*resolution)
                shift_amount = field_size - input_index%field_size
                local_coord = (shift_amount+RFpos[r][c])%field_size
                coord = input_index+local_coord
                Coord.append(coord) 
            Position.append(Coord)
        return Position

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
        f = open("./data.txt","a")
        while not rospy.is_shutdown():
            if self.instruction=='h':
                self.set_home_position()
            elif self.instruction=='r':
                self.repetitive_move()
            elif self.instruction=='d':    
                self.dance()
            elif self.instruction=='b':
                self.catch_ball()
            else:
                self.set_home_position()
            f.write(str(self.count) + ',' + str(self.red_bolb_x) +","+ str(self.red_bolb_y)+",")
            for i in range(len(self.joint_angles)):
                if self.joint_names[i]=="LShoulderPitch":
                    f.write(str(self.joint_angles[i])+",")
                elif self.joint_names[i]=="LShoulderRoll":
                    f.write(str(self.joint_angles[i])+"\n")
        f.close()
        self.set_stiffness(self.stiffness)
        rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    #f = open("data.txt","a")
    #f.write("hello")
    #f.close()
    central_instance.central_execute()
