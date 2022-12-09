from __future__ import absolute_import
import time
from math import ceil as ceil
from craft_text_detector import Craft
import re
import numpy as np
import cv2
from PIL import Image
import torch
import math
import logging
import os
logging.basicConfig(filename='craft_logging.log', level=logging.DEBUG)

 
       
def main():
    test_folder = '/home/sathish-beltech/beltech/LicensePlateRecognition_testing/test_imgs_anpr'
    file_list = os.listdir(test_folder)
    text_localizer = Craft(cuda = True, long_size=1280)
    failed_count=0
    bike_c =0
    bike_time = 0
    nbike_c = 0
    nbike_time = 0
    for item in file_list:
        filepath = test_folder + '/' + item
        image = cv2.imread(filepath)
        if 'bike' in item.split('_')[1]:
            st=time.time()
            prediction = text_localizer.detect_text(image, long_size=640)
            et =time.time()-st
            bike_time += et
            bike_c += 1
        else:
            st=time.time()
            prediction = text_localizer.detect_text(image, long_size=480)
            et = time.time()-st
            nbike_time += et
            nbike_c += 1
        if not prediction['text_crops']:
           failed_count +=1
           print(failed_count, item)
           continue
        count = 0
        for text_crop in prediction['text_crops']:
            filepath = '/home/sathish-beltech/beltech/LicensePlateRecognition_testing/test_result/'+item+'_%d.png'%count
            cv2.imwrite(filepath,text_crop)
            count +=1
        
    btime = bike_time/bike_c
    nbtime = nbike_time/nbike_c
    total_time = (btime+nbtime)/2
    print(total_time)
    
    
if __name__ == '__main__':
    main()
