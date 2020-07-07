#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import rospy
from std_msgs.msg import String

# read keyboard input
def read_input():
    pub = rospy.Publisher("key", String, queue_size=10)
    rospy.init_node("KeyboardNode", anonymous=True)
    rate = rospy.Rate(10)
    help_msg = "Instructions\n" + \
        "headhome: head to home position tutorial 3\n" + \
        "armhome: arm to home position tutorial 3\n" + \
        "trackfnn: track red ball with fnn\n" + \
        "trackcmac: track red ball with CMAC\n" + \
        "trackcmac coordinate: track red ball with CMAC using the set coordinates, e.g. trackcmac 11 12 means set the virtual red blob coordinates to [11, 12]\n" + \
        "nod: nod\n" + \
        "armmirror: right arm mirror start/stop \n" + \
        "set joint value: set joint degree to value, e.g. set LShoulderRoll 0.3: set left shoulder roll to 0.3 \n" + \
        "setr joint value: set joint position relatively to value, e.g. setr LShoulderRoll 0.3: set left shoulder roll +0.3 \n" + \
        "cvtest: test for cv2 \n" + \
        "wave: wave\n" + \
        "penaltyready: prepare for penalty kick\n" + \
        "penaltydt: use RLDT to shoot the penalty\n" + \
        "kick: execute kick action\n" + \
        "h: display this help\n" + \
        "q: quit\n"
    print(help_msg)
    while not rospy.is_shutdown():        
        msg = raw_input("Type a control signal: ")
        if msg[0:3] != "set":
            msg = msg.lower()
        # h and q are not sent to the subscriber
        if msg == "h":
            print(help_msg)
        elif msg == "q":
            break 
        else:
            pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        read_input()
    except rospy.ROSInterruptException:
        pass
