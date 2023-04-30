from ctypes import *
import math
import random
import os
import cv2
import numpy as np

print(os.getcwd())
darknet_location = os.getcwd()[:os.getcwd().rfind('cs424_final_project')] + 'cs424_final_project/darknet/'
print(darknet_location)

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

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

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL(darknet_location+"libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def np_image_to_c_IMAGE(input_frame):
    h, w, c = input_frame.shape
    flattened_image = input_frame.transpose(2, 0, 1).flatten().astype(np.float32)/255.
    c_float_p = POINTER(c_float)
    c_float_p_frame = flattened_image.ctypes.data_as(c_float_p)
    C_IMAGE_frame = IMAGE(w,h,c,c_float_p_frame)
    C_IMAGE_frame.ref = c_float_p_frame     # extra reference to data stored
    # print(C_IMAGE_frame)
    # print(C_IMAGE_frame.data)
    # print(cast(C_IMAGE_frame.data,c_void_p))  # the pointer value
    # print(C_IMAGE_frame.data.contents)  # data valid
    return C_IMAGE_frame

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)

    cv2_image = cv2.imread(image.decode('utf-8'))

    im = np_image_to_c_IMAGE(cv2_image)
    # print(im)
    # print(im.data)
    # print(cast(im.data,c_void_p)) # pointer is the same, but contents freed if no ref.
    # print(im.data.contents)  # crashes here if extra reference not kept.

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
    # free_image(im)
    free_detections(dets, num)
    return res
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    print('loading')
    net = load_net(bytes(darknet_location + "yolov3.cfg", 'utf-8'), bytes(darknet_location + "yolov3.weights", 'utf-8'), 0)
    meta = load_meta(bytes(darknet_location + "coco.data", 'utf-8'))
    print('detecting')
    r = detect(net, meta, bytes(darknet_location + "data/dog.jpg", 'utf-8'))
    for obj in r:

        obj_name = obj[0].decode('utf-8')
        obj_x, obj_y, obj_w, obj_h = obj[2]
        print(obj_name, obj_x, obj_y, obj_w, obj_h)

    # r = detect(net, meta, bytes(darknet_location + "data/person.jpg", 'utf-8'))
    # for obj in r:
    #     print(obj)

    # r = detect(net, meta, bytes(darknet_location + "data/horses.jpg", 'utf-8'))
    # for obj in r:
    #     print(obj)