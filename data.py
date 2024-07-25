####################################
import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize
from keras.metrics import MeanIoU
#########################################


def preprocess_data_image(directory_path_image):
    train_images = []

    for directory_path in glob.glob(directory_path_image):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            img = cv2.imread(img_path, 1)       
            #img = cv2.resize(img, (SIZE_Y, SIZE_X))
            train_images.append(img)
       
            #Convert list to array for machine learning processing        
            train_images = np.array(train_images)
    return train_images
 
def preprocess_data_image(directory_path_train_mask):
    train_masks = [] 
    for directory_path in glob.glob(directory_path_train_mask):
        for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
            mask = cv2.imread(mask_path, 0)       
            #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            train_masks.append(mask)
        
            #Convert list to array for machine learning processing          
            train_masks = np.array(train_masks)
    return train_masks


def preprocess_data_LabelEncoder():

    labelencoder = LabelEncoder()
    n, h, w = preprocess_data_image.shape
    train_masks_reshaped = preprocess_data_image.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    np.unique(train_masks_encoded_original_shape)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    return train_masks_input