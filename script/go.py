#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import openai
from dotenv import load_dotenv

from ctypes import *
import math
import random
import os
import cv2
import numpy as np

darknet_location = os.getcwd()[:os.getcwd().find('catkin_ws')] + 'catkin_ws/src/cs424_final_project/darknet/'
print(darknet_location)

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

user_response = "I'm hungry. "

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

    print(obj_info)

    # replace previous set of objects with current set
    global found_objects
    found_objects = set(obj_info.keys())

    if len(goal_object) == 0:
        pick_goal(user_response, found_objects)
    
    # set goal coordinates to center of bounding box for goal_object
    global goal_coordinates
    if goal_object in obj_info:
        goal_coordinates = obj_info[goal_object][0:2]

def pick_goal(user_response, found_objects):
    print(found_objects, len(found_objects))

    if len(found_objects) == 0:
        return

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
    f.write("\nPrompt: {}".format(prompt))
    f.close()

    if len(gpt_response) > 0:
        gpt_response = gpt_response[0]['text'].strip()

        global goal_object, goal_score, goal_explanation
        goal_object, goal_score, goal_explanation = gpt_response.split(';')
            
        global reasonable_goal
        reasonable_goal = int(goal_score) > 5

        f = open("/home/rmui1/catkin_ws/chat_records.txt", "a")
        f.write("\nResponse: {}".format(gpt_response))
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
    # set_gpu(1)

    rospy.init_node('turtlebot', anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=2)
    
    image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, handle_image)    
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
                f.write("\nPrompt: {}".format(check_prompt))
                f.close()

                if len(gpt_response) > 0:
                    gpt_response = gpt_response[0]['text'].strip()

                    f = open("/home/rmui1/catkin_ws/chat_records.txt", "a")
                    f.write("\nResponse: {}".format(gpt_response))
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

                        if len(new_goal) > 0:
                            if new_goal in found_objects:
                                see_new_goal = True
                                break

                        all_found_objects = all_found_objects.union(found_objects)

                        performed_at_least_once = True
                        start_orientation_2 = orientation

                if not see_new_goal:
                    pick_goal(user_response, all_found_objects)
                    new_goal = goal_object
            
    rospy.spin()
