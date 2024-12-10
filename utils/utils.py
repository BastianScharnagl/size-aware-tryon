import importlib
import torch
from PIL import Image, ImageOps
import numpy as np
import json 
import cv2
import math
import os

def instantiate_from_config(config):
        
    module, cls = config["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("init_args", dict()))

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def resize_with_padding(img, expected_size, color):
    if len(expected_size) == 1:
        expected_size = (expected_size, expected_size)
        
    img.thumbnail((expected_size[0], expected_size[1]))

    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=color)

def read_image(path, size, color=(0,0,0)):
  
    img = Image.open(path)

    if ".png" in path:
        img.convert('RGB')

    img = resize_with_padding(img, size, color)
    
    img = np.array(img).astype(np.float32) / 255.0
    img = (img * 2.0) - 1.0

    return img

# load keypoints
def read_keypoints(filename):
    with open(filename, "r") as f:
        data = json.load(f)

        keypoints = data["keypoints"]
        t_keypoints = torch.zeros([28])

        for i in range(len(keypoints)):
                x_k = keypoints[i][0]
                y_k = keypoints[i][1]
                t_keypoints[i*2] = x_k
                t_keypoints[i*2+1] = y_k
                    
    return t_keypoints
    
def read_keypoints(data_root, folder_name, file_id):
    ext = ".json"
    if folder_name == "human_keypoints_posed":
        ext = "_0_keypoints.json"
    elif folder_name == "fashion_keypoints_stock":
        ext = "_1.json"
    elif folder_name == "fashion_keypoints_posed":
        ext = "_0.json"
        
    with open(os.path.join(data_root, folder_name, file_id+ext), "r") as f:
        data = json.load(f)
        if folder_name == "human_keypoints_posed":
            keypoints = data["people"][0]["pose_keypoints_2d"]
            t_keypoints = torch.zeros([int(len(keypoints)/3*2)])

            for i in range(len(keypoints)-2):
                if i % 3 == 0:
                    x_k = keypoints[i]
                    y_k = keypoints[i+1]
                        
                    t_keypoints[i//3*2] = x_k
                    t_keypoints[(i//3*2)+1] = y_k
        elif folder_name in ["fashion_keypoints_posed", "fashion_keypoints_stock"]:
            keypoints = data["keypoints"]
            t_keypoints = torch.zeros([28])

            for i in range(len(keypoints)):
                    x_k = keypoints[i][0]
                    y_k = keypoints[i][1]
                    t_keypoints[i*2] = x_k
                    t_keypoints[i*2+1] = y_k
                    
    return t_keypoints

JOINT_PAIRS_MAP_ALL = {(1, 2): {'joint_names': ('Neck', 'RShoulder')},
                       (1, 5): {'joint_names': ('Neck', 'LShoulder')},
                       (1, 8): {'joint_names': ('Neck', 'MidHip')},
                       (2, 3): {'joint_names': ('RShoulder', 'RElbow')},
                       (3, 4): {'joint_names': ('RElbow', 'RWrist')},
                       (5, 6): {'joint_names': ('LShoulder', 'LElbow')},
                       (6, 7): {'joint_names': ('LElbow', 'LWrist')},
                       (8, 9): {'joint_names': ('MidHip', 'RHip')},
                       (8, 12): {'joint_names': ('MidHip', 'LHip')},
                       (9, 10): {'joint_names': ('RHip', 'RKnee')},
                       (10, 11): {'joint_names': ('RKnee', 'RAnkle')},
                       (11, 22): {'joint_names': ('RAnkle', 'RBigToe')},
                       (11, 24): {'joint_names': ('RAnkle', 'RHeel')},
                       (12, 13): {'joint_names': ('LHip', 'LKnee')},
                       (13, 14): {'joint_names': ('LKnee', 'LAnkle')},
                       (14, 19): {'joint_names': ('LAnkle', 'LBigToe')},
                       (14, 21): {'joint_names': ('LAnkle', 'LHeel')},
                       (15, 17): {'joint_names': ('REye', 'REar')},
                       (16, 18): {'joint_names': ('LEye', 'LEar')},
                       (19, 20): {'joint_names': ('LBigToe', 'LSmallToe')},
                       (22, 23): {'joint_names': ('RBigToe', 'RSmallToe')}}

LANDMARK_PAIRS = [(1,2),(2,3),(1,4),(4,5),(5,6),(7,8),(8,9),(9,10),(10,11),(12,13),(13,14),(14,3),(6,7),(11,12)]

def draw_pose(img: np.ndarray, pose_coords):
    for j1, j2 in JOINT_PAIRS_MAP_ALL.keys():
        x1 = int(pose_coords[j1*2])
        y1 = int(pose_coords[j1*2+1])
        x2 = int(pose_coords[j2*2])
        y2 = int(pose_coords[j2*2+1])
        # rangle 10->11
        # langle 13->14
        if j1 == 10 and j2 == 11:
            a_r = math.atan((x2-x1)/(y2-y1))
        elif j1 == 13 and j2 == 14:
            a_l = math.atan((x2-x1)/(y2-y1))

        if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
            cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return img, a_r, a_l

def draw_landmarks(img: np.ndarray, landmarks, offset=0.0, a_r=0.0, a_l=0.0):
    for l1, l2 in LANDMARK_PAIRS:
        x1 = int(landmarks[(l1-1)*2])
        y1 = int(landmarks[(l1-1)*2+1])
        x2 = int(landmarks[(l2-1)*2])
        y2 = int(landmarks[(l2-1)*2+1])

        if offset != 0.0:
            if (l1 == 5 and l2 == 6):
                x2 = x2 + math.sin(a_r) * offset
                y2 = y2 + math.cos(a_r) * offset
                landmarks[(l2-1)*2] = x2
                landmarks[(l2-1)*2+1] = y2
                
            elif (l1 == 7 and l2 == 8):
                x1 = x1 + math.sin(a_r) * offset
                y1 = y1 + math.cos(a_r) * offset
                landmarks[(l1-1)*2] = x1
                landmarks[(l1-1)*2+1] = y1
            elif (l1 == 10 and l2 == 11):
                x2 = x2 + math.sin(a_r) * offset
                y2 = y2 + math.cos(a_r) * offset
                landmarks[(l2-1)*2] = x2
                landmarks[(l2-1)*2+1] = y2
            elif (l1 == 12 and l2 == 13):
                x1 = x1 + math.sin(a_r) * offset
                y1 = y1 + math.cos(a_r) * offset
                landmarks[(l1-1)*2] = x1
                landmarks[(l1-1)*2+1] = y1
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
    return img