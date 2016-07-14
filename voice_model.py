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
(first array corresponds to data from right glove, second array corresponds to
data from the left glove)
[[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, flex_1, flex_2, flex_3, flex_4, flex5],
[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, flex_1, flex_2, flex_3, flex_4, flex5]]
"""

import os

    
def prepare_training_data():
    data = []
    for file in os.listdir("./data"):
        names = file.split("_")
        sign = names[0]
        user = names[1].split(".")[0]
        
        data.append({"label": sign, "user": user, "data": []})
    
    return data
    

print(prepare_training_data())