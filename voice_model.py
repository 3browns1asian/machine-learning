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
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.externals import joblib


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)
    

def prepare_data(train=True):
    if train:
        directory = './data'
    else:
        directory = './predict'

    data = []

    for file in os.listdir(directory):
        values = {"left": [], "right": []}
        names = file.split("_")
        sign = names[0]
        user = names[1].split(".")[0]
        
        # Damn it Mac
        if sign == ".DS":
            continue
        
        left_vals = []
        right_vals = []        
        for line in open(directory + "/" + file):
            splits = line.split("|")
            
            if splits[0] == "END" or splits[0] == "END\n":
                values["left"].append(left_vals)
                values["right"].append(right_vals)
                left_vals = []
                right_vals = []
            else:
                if "|" in line:
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

                else:
                    left_array = splits[0].split(",")
                    left_value = [float(left_array[0]), float(left_array[1]), float(left_array[2]),
                                  float(left_array[3]), float(left_array[4]), float(left_array[5]),
                                  float(left_array[6]), float(left_array[7]),
                                  float(left_array[8]), float(left_array[9]), float(left_array[10])]
                    right_value = [0 for x in range(11)]

                left_vals.append(left_value)
                right_vals.append(right_value)
                            
        data.append({"label": sign, "user": user, "values": values})
        random.shuffle(data)
    
    return data
    

def feature_extraction(data):
    new_data = []
    
    for f_data in data:
        
        left_vals = np.array([val for val in f_data["values"]["left"]])
        right_vals = np.array([val for val in f_data["values"]["left"]])

        for y in range(len(left_vals)):
            features = []
            a = np.array(left_vals[y])
            b = np.array(right_vals[y])
            
            # Left hand features
            if len(a) != 0 and len(a[0]) != 0:
                # Feature 1: Mean of DCT of Acceleration of X
                transformed_values_x = np.array(dct(a[:, 0]))
                features.append(round(np.mean(transformed_values_x), 3))
                
                # Feature 2: Mean of DCT of Acceleration of Y
                transformed_values_y = np.array(dct(a[:, 1]))
                features.append(round(np.mean(transformed_values_y), 3))
                
                # Feature 3: Mean of DCT of Acceleration of Z
                transformed_values_z = np.array(dct(a[:, 2]))
                features.append(round(np.mean(transformed_values_z), 3))
                
                # Feature 4/5: Mean Absolute Deviation and Mean of gyro in X
                features.append(round(mad(a[:, 3])))
                features.append(round(np.mean(a[:, 3])))
                
                # Feature 6/7: Mean Absolute Deviation and Mean of gyro in Y
                features.append(round(mad(a[:, 4])))
                features.append(round(np.mean(a[:, 4])))
                
                # Feature 8/9: Mean Absolute Deviation and Mean of gyro in Z
                features.append(round(mad(a[:, 5])))
                features.append(round(np.mean(a[:, 5])))
                
                # Feature 10/11: Standard Absolute Deviation and Mean of flex 1
                features.append(round(np.std(a[:, 6])))
                features.append(round(np.mean(a[:, 6])))
                
                # Feature 12/13: Standard Absolute Deviation and Mean of flex 2
                features.append(round(np.std(a[:, 7])))
                features.append(round(np.mean(a[:, 7])))
                
                # Feature 14/15: Standard Absolute Deviation and Mean of flex 3
                features.append(round(np.std(a[:, 8])))
                features.append(round(np.mean(a[:, 8])))
                
                # Feature 16/17: Standard Absolute Deviation and Mean of flex 4
                features.append(round(np.std(a[:, 9])))
                features.append(round(np.mean(a[:, 9])))
                
                # Feature 18/19: Standard Absolute Deviation and Mean of flex 5
                features.append(round(np.std(a[:, 10])))
                features.append(round(np.mean(a[:, 10])))            
            
            # Right hand features
            if len(b) != 0 and len(b[0]) != 0:
                # Feature 20: Mean of DCT of Acceleration of X
                transformed_values_x = np.array(dct(b[:, 0]))
                features.append(round(np.mean(transformed_values_x), 3))
                
                # Feature 21: Mean of DCT of Acceleration of Y
                transformed_values_y = np.array(dct(b[:, 1]))
                features.append(round(np.mean(transformed_values_y), 3))
                
                # Feature 22: Mean of DCT of Acceleration of Z
                transformed_values_z = np.array(dct(b[:, 2]))
                features.append(round(np.mean(transformed_values_z), 3))
                
                # Feature 23/24: Mean Absolute Deviation and Mean of gyro in X
                features.append(round(mad(b[:, 3])))
                features.append(round(np.mean(b[:, 3])))
                
                # Feature 25/26: Mean Absolute Deviation and Mean of gyro in Y
                features.append(round(mad(b[:, 4])))
                features.append(round(np.mean(b[:, 4])))
                
                # Feature 27/28: Mean Absolute Deviation and Mean of gyro in Z
                features.append(round(mad(b[:, 5])))
                features.append(round(np.mean(b[:, 5])))
                
                # Feature 29/30: Standard Absolute Deviation and Mean of flex 1
                features.append(round(np.std(b[:, 6])))
                features.append(round(np.mean(b[:, 6])))
                
                # Feature 31/32: Standard Absolute Deviation and Mean of flex 2
                features.append(round(np.std(b[:, 7])))
                features.append(round(np.mean(b[:, 7])))
                
                # Feature 33/34: Standard Absolute Deviation and Mean of flex 3
                features.append(round(np.std(b[:, 8])))
                features.append(round(np.mean(b[:, 8])))
                
                # Feature 35/36: Standard Absolute Deviation and Mean of flex 4
                features.append(round(np.std(b[:, 9])))
                features.append(round(np.mean(b[:, 9])))
                
                # Feature 37/38: Standard Absolute Deviation and Mean of flex 5
                features.append(round(np.std(b[:, 10])))
                features.append(round(np.mean(b[:, 10])))
                
            if len(features) > 0:
                new_data.append({"label": f_data["label"], "user": f_data["user"], "features": features[:19]})
    
    return new_data
    

# Gather the training and the testing data
train_data = feature_extraction(prepare_data(train=True))
predict_data = feature_extraction(prepare_data(train=False))

df = pd.DataFrame(train_data)

cols = df.label.unique().tolist()
features = df.features

print(cols)

joblib.dump(cols, 'col.pkl')

# Grab the features and the labels
X_train = np.array(features.tolist())
Y_train = []

for label in df.label:
    Y_train.append(cols.index(label))

Y_train = np.array(Y_train)

predict_df = pd.DataFrame(predict_data)
X_pred = np.array(predict_df.features.tolist())

Y_pred = []
for label in predict_df.label:
    Y_pred.append(cols.index(label))

Y_pred = np.array(Y_pred)

# The classifiers
# Naiive Bayes
clf_1 = GaussianNB()
clf_1.fit(X_train, Y_train)

# Decision Trees
clf_2 = tree.DecisionTreeClassifier()
clf_2.fit(X_train, Y_train)

# SVM
clf_3 = svm.SVC(kernel='linear', C=10000.0)
clf_3.fit(X_train, Y_train)

# joblib.dump(clf_1, 'bayes.pkl')
# joblib.dump(clf_2, 'decision_tree.pkl')
# joblib.dump(clf_3, 'svm.pkl')

# clf_4 = joblib.load('bayes.pkl')
preds_nb = clf_1.predict(X_pred)
preds_dt = clf_2.predict(X_pred)
preds_svm = clf_3.predict(X_pred)

print("Naiive Bayes, Decision Tree, SVM")

# for i in range(len(preds_nb)):
#     print(cols[preds_nb[i]] + "--" + cols[preds_dt[i]] +  "--" + cols[preds_svm[i]])
#     print("End")

print(np.shape(X_train), np.shape(Y_train))
print(np.shape(X_pred), np.shape(Y_pred))
# K-Fold Cross Validation
#NOTE: Might throw an error where #of classes in subsample is 1.
#kf = KFold(len(X_train), n_folds = 2)
#
#scores_nb = []
#scores_dt = []
#scores_svm = []
#
#for train_index, test_index in kf:
#    X_t, X_test = X_train[train_index], X_train[test_index]
#    Y_t, Y_test = Y_train[train_index], Y_train[test_index]
#    
#    clf_test_nb = GaussianNB()
#    clf_test_nb.fit(X_t, Y_t)
#    scores_nb.append(clf_test_nb.score(X_test, Y_test))
#    
#    clf_test_dt = tree.DecisionTreeClassifier()
#    clf_test_dt.fit(X_t, Y_t)
#    scores_dt.append(clf_test_dt.score(X_test, Y_test))
#    
#    clf_test_svm = svm.SVC(kernel = "rbf", C=10000.0)
#    clf_test_svm.fit(X_t, Y_t)
#    scores_svm.append(clf_test_svm.score(X_test, Y_test))
#    
#print(scores_nb)
#print(scores_dt)
#print(scores_svm)

# Leave one user out cross validation
# leave_out_user = random.choice(df.user.unique().tolist())
# df_test = df.loc[df['user'] == leave_out_user]
# df_train = df.loc[df['user'] != leave_out_user]
#
# features = df_train.features
#
# # Grab the features and the labels
# X_train = np.array(features.tolist())
# Y_train = []
#
# for label in df_train.label:
#     Y_train.append(cols.index(label))
#
# Y_train = np.array(Y_train)
#
# X_pred = np.array(df_test.features.tolist())
# Y_pred = []
#
# for label in df_test.label:
#     Y_pred.append(cols.index(label))
#
# Y_pred = np.array(Y_pred)
#
#
# print np.shape(Y_train)
# # Accuracy
# clf_1 = GaussianNB()
# clf_1.fit(X_train, Y_train)
#
# # Decision Trees
# clf_2 = tree.DecisionTreeClassifier()
# clf_2.fit(X_train, Y_train)
#
# #SVM
# clf_3 = svm.SVC(kernel = "rbf", C=10000.0)
# clf_3.fit(X_train, Y_train)
#
print(clf_1.score(X_pred, Y_pred))
print(clf_2.score(X_pred, Y_pred))
print(clf_3.score(X_pred, Y_pred))
