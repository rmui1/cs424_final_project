#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from darknet_ros_msgs.msg import BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = ""
found_objects = set()
goal_object = ''
goal_coordinates = (-1, -1)
goal_score = 0
goal_explanation = ''
reasonable_goal = False
distance_to_goal = -1

first_third = -1
last_third = -1

def handle_image(data):
    global image_publisher
    image_publisher.publish(data)

def handle_detections(data):
    print(data)

    # replace previous set of objects with current set
    global found_objects
    found_objects = set()

    if len(goal_object) == 0:
        pick_goal(found_objects)
    
    global goal_coordinates
    # set goal coordinates to center of bounding box for goal_object

def pick_goal(user_response, found_objects):
    # ask gpt3 to pick item and score its usefulness from 1-10 (in format: item;score)
    global prompt
    prompt += user_response
    
    if len(prompt) == 0:
        prompt += '''There\'s a {}. Which item is best for me? Answer with one word, all lowercase,
                        and no punctuation. Then, rate the object in terms of its usefulness to me at the
                        moment, from 1 to 10. Explain. Phrase your response in the
                        form: item;number;explanation\n\n
                    '''.format(', '.join(found_objects))
    else:
        prompt += '''Pick again from these objects: {}. Don't use any previous answers unless
                        there aren't any new answers. If you reuse a previous answer, say the usefulness is 0.
                        Phrase your response in the form: item;number;explanation\n\n
                    '''.format(', '.join(found_objects))
        
    gpt_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )['choices']

    f = open("/home/rmui1/catkin_ws/chat_records.txt", "a")
    f.write("Prompt: {}".format(prompt))
    f.close()

    if len(gpt_response) > 0:
        gpt_response = gpt_response[0]['text'].strip()

        global goal_object, goal_score, goal_explanation
        goal_object, goal_score, goal_explanation = gpt_response.split(';')
            
        global reasonable_goal
        reasonable_goal = goal_score > 5

        f = open("/home/rmui1/catkin_ws/chat_records.txt", "a")
        f.write("Response: {}".format(gpt_response))
        f.close()

def handle_depth_image(data):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
        print(e)
    (rows, cols) = cv_image.shape

    global first_third, last_third
    first_third = (int) (cols / 3)
    last_third = first_third * 2

    # get current distance to goal if one exists
    if goal_coordinates[0] > 0 and goal_coordinates[1] > 0:
        global distance_to_goal
        distance_to_goal = cv_image[goal_coordinates]
    else:
        distance_to_goal = -1

if __name__ == '__main__':
    rospy.init_node('turtlebot', anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=2)
    
    image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, handle_image)
    image_publisher = rospy.Publisher('/camera_reading', Image, queue_size=2)
    
    darknet_subscriber = rospy.Subscriber('bounding_boxes', BoundingBoxes, handle_detections)
    depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, handle_depth_image)

    rospy.spin()
    
    rate = rospy.Rate(0.1) # ROS Rate at 5Hz
    
    while not rospy.is_shutdown():
        reached_goal = False
        if goal_coordinates[0] > 0 and goal_coordinates[1] > 0:
            vel_msg = Twist()
            if first_third > 0 and goal_coordinates[1] < first_third:
                vel_msg.angular.z = 0.75
            elif last_third > 0 and goal_coordinates[1] > last_third:
                vel_msg.angular.z = -0.75
            elif distance_to_goal > 700:
                vel_msg.linear.x = 0.25
            else:
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                reached_goal = True
            global velocity_publisher
            velocity_publisher.publish(vel_msg)

        if reached_goal:
            if reasonable_goal:
                check_prompt = '''A user said, "{}". Is this user satisfied with the service they were
                                    provided? Answer 0 for no, 1 for yes.\n\n
                                '''.format(user_response)

                gpt_response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=check_prompt,
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )['choices']

                f = open("/home/rmui1/catkin_ws/chat_records.txt", "a")
                f.write("Prompt: {}".format(check_prompt))
                f.close()

                if len(gpt_response) > 0:
                    gpt_response = gpt_response[0]['text'].strip()

                    f = open("/home/rmui1/catkin_ws/chat_records.txt", "a")
                    f.write("Response: {}".format(gpt_response))
                    f.close()
                
                # ask user if it meets their needs
                if gpt_response == '1':
                    break

            new_goal = ""
            see_new_goal = False
            while not see_new_goal:
                # turn in circle, looking for new goal
                all_found_objects = set()
                performed_at_least_once = False

                rospy.wait_for_message('/odom', Odometry)
                start_orientation_1 = start_orientation_2 = orientation
                while not rospy.is_shutdown() and (abs(orientation - start_orientation_1) > 0.01 or not performed_at_least_once):
                    vel_msg = Twist()
                    if (abs(orientation - start_orientation_2) < 0.1):
                        vel_msg.angular.z = 0.75
                        velocity_publisher.publish(vel_msg)
                    else:
                        vel_msg.angular.z = 0
                        velocity_publisher.publish(vel_msg)

                        rospy.wait_for_message('bounding_boxes', BoundingBoxes)

                        if len(new_goal) > 0:
                            if new_goal in found_objects:
                                see_new_goal = True
                                break

                        all_found_objects = all_found_objects.union(found_objects)

                        performed_at_least_once = True
                        start_orientation_2 = orientation

                if not see_new_goal:
                    pick_goal(all_found_objects)
                    new_goal = goal_object
            
    rospy.spin()
