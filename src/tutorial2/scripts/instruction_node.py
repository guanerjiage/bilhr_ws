#! /usr/bin/env python
import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('instruction', String, queue_size=10)
    rospy.init_node('instruction_node', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        msg = raw_input('Type a instruction:\n h for home position\n r for repetitive movement\n f for symetric movement\n') 
        pub.publish(msg)
        rate.sleep()
    
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass