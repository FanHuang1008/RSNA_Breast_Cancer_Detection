import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt


def to_3_channels(img, is_monochrome1):
    img = (img - img.min()) / (img.max() - img.min())
    
    if is_monochrome1:
        img = 1 - img    
    img = (img * 255)
    
    # 1 channel -> 3 channels
    img_3c = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))
    img_3c[:,:,0] = img 
    img_3c[:,:,1] = img
    img_3c[:,:,2] = img

    return img_3c.astype(np.uint8)

def get_yolo():
    b = plt.get_backend()
    model = torch.hub.load('./yolov5/', 'custom', path='./yolov5/rsna-roi-003.pt', source='local')
    matplotlib.use(b) 
    return model

def roi_extraction_yolov5(model, img):
    #select only best prediction
    prediction = model(img).pandas().xyxy[0].to_dict(orient='records')
    if len(prediction)==0:
        return None
        
    prediction = prediction[0]
    result = {key:prediction[key] for key in ['xmin','xmax','ymin','ymax']}
    
    if 768 - result['xmax'] < 10:
        result['xmax'] = 768
        
    if result['xmin'] < 10:
        result['xmin'] = 0
    return result

def roi_extraction_cv2(img):
    # Otsu's thresholding after Gaussian filtering
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    #_, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, breast_mask = cv2.threshold(blur, 0, 255, 16)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    roi = {
        "xmin": x,
        "ymin": y,
        "xmax": x + w,
        "ymax": y + h
    }
    return roi

def show_img_and_roi(img, roi):
    fig = plt.figure(figsize=(5, 5))
    rect = cv2.rectangle(img, (int(roi['xmin']), int(roi['ymin'])), (int(roi['xmax']), int(roi['ymax'])), (255,0,0), 4)
    
    plt.imshow(img, cmap='bone')
    plt.show()

def crop_img(img, roi):
    x, y = int(roi["xmin"]), int(roi["ymin"])
    w, h = int(roi["xmax"] - roi["xmin"]), int(roi["ymax"] - roi["ymin"])
    return img[y:y+h, x:x+w]

def resize_and_pad(img, input_size=[2048, 1024]):
    input_h, input_w = input_size
    ori_h, ori_w = img.shape[:2]
    ratio = min(input_h / ori_h, input_w / ori_w)
    # resize
    img = torch.from_numpy(img[:,:,0])
    img = img.to(torch.float32)
    img = F.interpolate(img.view(1, 1, ori_h, ori_w),
                        mode="bilinear",
                        scale_factor=ratio,
                        recompute_scale_factor=True)[0, 0]
    # padding
    padded_img = torch.zeros((input_h, input_w),
                             dtype=img.dtype,
                             device='cuda')
    cur_h, cur_w = img.shape
    y_start = (input_h - cur_h) // 2
    x_start = (input_w - cur_w) // 2
    padded_img[y_start:y_start + cur_h, x_start:x_start + cur_w] = img
    padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
    padded_img = padded_img.to(torch.uint8)
    padded_img = padded_img.cpu().numpy()
    
    return padded_img

def check_mkdir(file_path, isdir:bool=False):
    path_sep_list = os.path.abspath(file_path).split(os.path.sep)

    if ':' in path_sep_list[0]:
        path_sep_list[0] = path_sep_list[0]+os.path.sep

    for depth_i in range(len(path_sep_list)):
        if depth_i >0:
            dir_path = os.path.join(*path_sep_list[:depth_i])
            if path_sep_list[0]=='':
                dir_path = '/'+dir_path
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
    if isdir:
        if not os.path.exists(file_path):
            os.mkdir(file_path)