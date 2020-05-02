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
        "a: head to home position\n" + \
        "b: arm to home position \n" + \
        "n: nod\n" + \
        "r: right arm mirror start/stop \n" + \
        "t: test for cv2 \n" + \
        "w: wave\n" + \
        "h: display this help\n" + \
        "q: quit\n"
    print(help_msg)
    while not rospy.is_shutdown():        
        msg = raw_input("Type a control signal: ")
        msg = msg.lower()
        # h and q are not be sent to the subscriber
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