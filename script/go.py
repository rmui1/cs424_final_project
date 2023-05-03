#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import openai
from dotenv import load_dotenv

from ctypes import *
import math
import random
import os
import cv2
import numpy as np

import time

darknet_location = os.getcwd()[:os.getcwd().find('catkin_ws')] + 'catkin_ws/src/cs424_final_project/darknet/'

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

training_text = '''Example: I'm hungry. I have the following items: shirt. Which item is best for me? Pick one of the items and 
                    answer with one word, all lowercase, and no punctuation. If none of the items are good, pick a 
                    random one, but tell me that its usefulness is 0. Then, rate the object in terms of its usefulness
                    to me at the moment, from 1 to 10. Explain in second-person point of view. Phrase your response 
                    in the form: item;number;explanation\n\n
                    shirt;0;You can't eat a shirt, but it's the only available item.\n\n
                    Now, consider this new case with the same logic:\n\n
                '''
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
orientation = -1

user_response = ""

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
lib = CDLL(darknet_location+"libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

net = load_net(bytes(darknet_location + "yolov3.cfg", 'utf-8'), bytes(darknet_location + "yolov3.weights", 'utf-8'), 0)
meta = load_meta(bytes(darknet_location + "coco.data", 'utf-8'))

def reset_goal_object():
    global goal_object
    goal_object = ""

def np_image_to_c_IMAGE(input_frame):
    h, w, c = input_frame.shape
    flattened_image = input_frame.transpose(2, 0, 1).flatten().astype(np.float32)/255.
    c_float_p = POINTER(c_float)
    c_float_p_frame = flattened_image.ctypes.data_as(c_float_p)
    C_IMAGE_frame = IMAGE(w,h,c,c_float_p_frame)
    C_IMAGE_frame.ref = c_float_p_frame     # extra reference to data stored
    return C_IMAGE_frame

def detect(data, thresh=.5, hier_thresh=.5, nms=.45):
    bridge = CvBridge()

    try:
        cv2_image = bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as e:
        print(e)

    im = np_image_to_c_IMAGE(cv2_image)

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

def handle_image(data):
    darknet_results = detect(data)
    obj_info = {r[0].decode('utf-8'): r[2] for r in darknet_results}

    # replace previous set of objects with current set
    global found_objects, goal_coordinates, goal_object
    found_objects = set(obj_info.keys())

    if len(goal_object) == 0 and len(user_response) > 0:
        pick_goal(found_objects)
    
    # set goal coordinates to center of bounding box for goal_object
    if goal_object in obj_info:
        goal_coordinates = (int(obj_info[goal_object][1]), int(obj_info[goal_object][0]))

    # if len(goal_object) > 0:
    #     print('goal info:', goal_object, goal_score, goal_explanation)
    #     print(goal_coordinates)
    #     print(found_objects)

def handle_odom(data):
    global orientation
    orientation = data.pose.pose.orientation.w

def pick_goal(found_objects):
    global goal_object, goal_score, goal_explanation, reasonable_goal
    
    if len(found_objects) == 0:
        goal_object = ""
        return

    # ask gpt3 to pick item and score its usefulness from 1-10 (in format: item;score)
    global prompt
    
    # print('asking gpt')

    if len(prompt) == 0:
        prompt += '''{} {}. I have the following items: {}. Which item is best for me? Pick one of the items 
                        and answer with one word, all lowercase, and no punctuation. If none of the items are 
                        good, pick a  random one, but tell me that its usefulness is 0. Then, rate the object 
                        in terms of its usefulness to me at the moment, from 1 to 10. Explain in second-person 
                        point of view. Phrase your response in the form: item;number;explanation \n\n
                    '''.format(training_text, user_response, ', '.join(found_objects))
    else:
        prompt += '''{}. Pick again from these objects: {}. Don't use any previous answers unless
                        there aren't any new answers. If you reuse a previous answer, say the usefulness is 0.
                        Phrase your response in the form: item;number;explanation\n\n
                    '''.format(user_response, ', '.join(found_objects))
    
    gpt_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )['choices']

    while not rospy.is_shutdown() and len(gpt_response) == 0:
        continue

    print('gpt responded with ', gpt_response)
    print('current goal object: ', goal_object if len(goal_object) > 0 else 'none')

    gpt_response = gpt_response[0]['text'].strip()

    goal_object, goal_score, goal_explanation = gpt_response.split(';')
    reasonable_goal = int(goal_score) > 5

    prompt += gpt_response + '\n\n'

    f = open("/home/rmui1/catkin_ws/chat_records.txt", "w")
    f.write(prompt)
    f.close()

    if goal_object not in found_objects:
        goal_object = ""

            # prompt += '''That's not one of the available items. The 
            #     available items are: {}. Pick again from the available items.
            # '''.format(', '.join(found_objects))

    print(goal_object, found_objects)

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

    np_image = np.array(cv_image)

    # get current distance to goal if one exists
    if goal_coordinates[0] > 0 and goal_coordinates[1] > 0:
        global distance_to_goal
        distance_to_goal = np_image[goal_coordinates]
        # print(distance_to_goal)
    else:
        distance_to_goal = -1

if __name__ == '__main__':
    f = open("/home/rmui1/catkin_ws/check_records.txt", "w")
    f.write("")
    f.close()

    rospy.init_node('turtlebot', anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=2)
    
    image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, handle_image)    
    depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, handle_depth_image)
    odom_subscriber = rospy.Subscriber('/odom', Odometry, handle_odom)

    rate = rospy.Rate(0.1) # ROS Rate at 5Hz
    
    user_response = input('What are you in the mood for? ')

    while len(user_response) == 0:
        continue

    time.sleep(5)

    while not rospy.is_shutdown():

        reached_goal = False

        print(goal_object, goal_coordinates)

        if len(goal_object) > 0 and goal_coordinates[0] > 0 and goal_coordinates[1] > 0:

            print('have goal coordinates to ', goal_object)
            print(distance_to_goal, goal_coordinates[1])

            vel_msg = Twist()
            if first_third > 0 and goal_coordinates[1] < first_third:
                # print('left')
                vel_msg.angular.z = 0.5
            elif last_third > 0 and goal_coordinates[1] > last_third:
                # print('right')
                vel_msg.angular.z = -0.5
            elif distance_to_goal > 500:
                vel_msg.linear.x = 0.25
            else:
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                reached_goal = True
            velocity_publisher.publish(vel_msg)

            if not reached_goal:
                time.sleep(0.5)
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0
                velocity_publisher.publish(vel_msg)
                time.sleep(0.5)
                continue

        if reached_goal:
            # print("\n\n\n\n\n\n\nreached goal!!!\n\n\n\n\n\n\n\n\n")
            # print('goal info:', goal_object, goal_score, goal_explanation)
            
            if reasonable_goal:
                # ask user if it meets their needs
                user_satisfaction = input("Here's a {}. {} Does it meet your request? ".format(goal_object, goal_explanation))

                while not rospy.is_shutdown() and len(user_satisfaction) == 0:
                    continue

                check_prompt = '''When asked if they were happy with the service they were provided, a user said, "{}". 
                                    Is this user satisfied with the service? Answer 0 for no, 1 for yes.\n\n
                                '''.format(user_satisfaction)

                gpt_response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=check_prompt,
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )['choices']

                f = open("/home/rmui1/catkin_ws/check_records.txt", "a")
                f.write("\nPrompt: {}".format(check_prompt))
                f.close()

                while not rospy.is_shutdown() and len(gpt_response) == 0:
                    continue

                gpt_response = gpt_response[0]['text'].strip()

                f = open("/home/rmui1/catkin_ws/check_records.txt", "a")
                f.write("\nResponse: {}".format(gpt_response))
                f.close()
                
                if gpt_response == '1':
                    print("I'm glad to hear that!")
                    break

                print("Alright, I'll keep looking.")
                reset_goal_object()

        if len(goal_object) == 0:
            rospy.wait_for_message('/odom', Odometry)
            start_orientation = orientation
            while not rospy.is_shutdown() and abs(orientation - start_orientation) < 0.5:
                vel_msg = Twist()
                vel_msg.angular.z = 0.5
                velocity_publisher.publish(vel_msg)
                        
            vel_msg.angular.z = 0
            velocity_publisher.publish(vel_msg)

            reset_goal_object()
            while len(goal_object) == 0:
                continue

            print("after sleeping: ", goal_object, found_objects)

            if len(goal_object) > 0:
                # print(goal_object)
                pass