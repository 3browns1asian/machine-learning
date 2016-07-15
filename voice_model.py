# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:34:49 2016

@author: akshaybudhkar

This file reads sensor data from the data folder, and gathers the data
in the right format. Then the data is run through 3 ML models to determine
the right constants. Data is divided in training and testing set - and cross
validation is tested in 2 ways.

Format of a file in the data_folder:
signText_userX.txt

Format of the data in each file:
Every instance of the sensor array looks like this:
(first half corresponds to data from left glove, second half corresponds to
data from the right glove)
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, flex_1, flex_2, flex_3, flex_4, flex5 |
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, flex_1, flex_2, flex_3, flex_4, flex5


A complete sign will have multiple data instances. Complete signs are separated
from each other with the keyword END
"""

import os
import random
import math
import pandas as pd
import numpy as np
from scipy.fftpack import dct

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)
    
def prepare_training_data():
    data = []

    for file in os.listdir("./data"):
        values = {"left": [], "right": []}
        names = file.split("_")
        sign = names[0]
        user = names[1].split(".")[0]
        
        # Damn it Mac
        if sign == ".DS":
            continue
        
        left_vals = []
        right_vals = []        
        for line in open("./data/" + file):
            splits = line.split("|")
            
            if splits[0] == "END" or splits[0] == "END\n":
                values["left"].append(left_vals)
                values["right"].append(right_vals)
                left_vals = []
                right_vals = []
            else:
                left_array = splits[0].split(",")
                right_array = splits[1].split(",")
                
                left_value = [float(left_array[0]), float(left_array[1]), float(left_array[2]),
                              float(left_array[3]), float(left_array[4]), float(left_array[5]),
                              float(left_array[6]), float(left_array[7]),
                              float(left_array[8]), float(left_array[9]), float(left_array[10])]
                              
                right_value = [float(right_array[0]), float(right_array[1]), float(right_array[2]),
                              float(right_array[3]), float(right_array[4]), float(right_array[5]),
                              float(right_array[6]), float(right_array[7]),
                              float(right_array[8]), float(right_array[9]), float(right_array[10])]
                              
                left_vals.append(left_value)
                right_vals.append(right_value)
                            
        data.append({"label": sign, "user": user, "values": values})
        random.shuffle(data)
    
    return data
    
def feature_extraction(data):
    new_data = []
    
    for f_data in data:
        
        left_vals = [val for val in f_data["values"]["left"]]

        a = np.array(left_vals)

        for y in a:
            features = []
            b = np.array(y)
            
            if len(b) != 0 and len(b[0]) != 0:
                # Feature 1: Mean of DCT of Acceleration of X
                transformed_values_x = np.array(dct(b[:, 0]))
                features.append(round(np.mean(transformed_values_x), 3))
                
                # Feature 2: Mean of DCT of Acceleration of Y
                transformed_values_y = np.array(dct(b[:, 1]))
                features.append(round(np.mean(transformed_values_y), 3))
                
                # Feature 3: Mean of DCT of Acceleration of Z
                transformed_values_z = np.array(dct(b[:, 2]))
                features.append(round(np.mean(transformed_values_z), 3))
                
                # Feature 4/5: Mean Absolute Deviation and Mean of gyro in X
                features.append(round(mad(b[:, 3])))
                features.append(round(np.mean(b[:, 3])))
                
                # Feature 6/7: Mean Absolute Deviation and Mean of gyro in Y
                features.append(round(mad(b[:, 4])))
                features.append(round(np.mean(b[:, 4])))
                
                # Feature 8/9: Mean Absolute Deviation and Mean of gyro in Z
                features.append(round(mad(b[:, 5])))
                features.append(round(np.mean(b[:, 5])))
                
                # Feature 10/11: Standard Absolute Deviation and Mean of flex 1
                features.append(round(np.std(b[:, 6])))
                features.append(round(np.mean(b[:, 6])))
                
                # Feature 12/13: Standard Absolute Deviation and Mean of flex 2
                features.append(round(np.std(b[:, 7])))
                features.append(round(np.mean(b[:, 7])))
                
                # Feature 14/15: Standard Absolute Deviation and Mean of flex 3
                features.append(round(np.std(b[:, 8])))
                features.append(round(np.mean(b[:, 8])))
                
                # Feature 16/17: Standard Absolute Deviation and Mean of flex 4
                features.append(round(np.std(b[:, 9])))
                features.append(round(np.mean(b[:, 9])))
                
                # Feature 18/19: Standard Absolute Deviation and Mean of flex 5
                features.append(round(np.std(b[:, 10])))
                features.append(round(np.mean(b[:, 10])))
                
            new_data.append({"label": f_data["label"], "user": f_data["user"], "features": features})
    
    return new_data
    

print(feature_extraction(prepare_training_data()))