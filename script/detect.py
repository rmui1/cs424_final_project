from ctypes import *
import math
import random
import os
import cv2
import numpy as np

print(os.getcwd())
darknet_location = os.getcwd()[:os.getcwd().rfind('cs424_final_project')] + 'cs424_final_project/darknet/'
print(darknet_location)

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