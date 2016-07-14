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

    
def prepare_training_data():
    data = []
    values = {"left": [], "right": []}
    for file in os.listdir("./data"):
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
                print("Reached the end")
            else:
                left_array = splits[0].split(",")
                right_array = splits[1].split(",")
                
                left_value = {"accel": [float(left_array[0]), float(left_array[1]), float(left_array[2])],
                              "gyro": [float(left_array[3]), float(left_array[4]), float(left_array[5])],
                              "flex": [float(left_array[6]), float(left_array[7]),
                              float(left_array[8]), float(left_array[9]), float(left_array[10])]}
                              
                right_value = {"accel": [float(right_array[0]), float(right_array[1]), float(right_array[2])],
                              "gyro": [float(right_array[3]), float(right_array[4]), float(right_array[5])],
                              "flex": [float(right_array[6]), float(right_array[7]),
                              float(right_array[8]), float(right_array[9]), float(right_array[10])]}
                              
                left_vals.append(left_value)
                right_vals.append(right_value)
                
                            
        data.append({"label": sign, "user": user, "values": values})
    
    return data
    

print(prepare_training_data())