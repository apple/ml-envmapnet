
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.

# HDR Environment Map Estimation for Real-Time Augmented Reality, CVPR 2021.
# Reference implementation of the FID (Frechet Inception Distance metric used in the above paper.



import numpy as np
import sys,os,platform,glob,time,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter('ignore')
import logging
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)


import keras
import scipy
import py360convert

# You can load this inception model from Keras, or load your own.
incp_img_size=299
incp_model=keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                       input_shape=(incp_img_size,incp_img_size,3),
                                                       pooling='avg')

# Helper to convert a batch of equirectangular to cubefaces, and then calculate inception features on those images.
# This function assumes the images have been correctly preprocessed to match the model requirements.
# For (batch,H,W,3) input, this function returns a (batch*4, 2048) array.
# Each equirectangular image is converted to cubemap, and the four side faces are fed as images to the inception model.
def inception_feature_for_equirectangular(equirectangular_images,incp_img_size,incp_model):
    # standarize eq_im to batch,h,w,channels for ease
    if len(np.shape(equirectangular_images))<4:
        equirectangular_images=np.expand_dims(equirectangular_images,axis=0)
    batch,h,w,c=np.shape(equirectangular_images)
    all_faces=[]
    for b in range(0,batch):
        cube_faces=py360convert.e2c(equirectangular_images[b],cube_format='list',face_w=incp_img_size)
        # here we ignore top and bottom since they usually do not have much features
        cube_faces=cube_faces[0:4]
        if b==0:
            all_faces=np.squeeze(cube_faces)
        else:
            all_faces=np.append(all_faces,cube_faces,axis=0)

    inception_features=incp_model.predict(all_faces)
    return inception_features

# Given inception features from two sets of images (e.g. training images and images predicted by a model), calculate FID.
# Given the definition of FID, it requires features of at least a few thousand images to be meaningful.
def calculate_fid(feature_set1, feature_set2):
    features_1 = np.asarray(feature_set1).astype(np.float32)
    features_2 = np.asarray(feature_set2).astype(np.float32)
    mu_1 = np.mean(features_1, axis=0)
    mu_2 = np.mean(features_2, axis=0)

    cv_1 = np.cov(features_1, rowvar=False)
    cv_2 = np.cov(features_2, rowvar=False)

    score = calculate_frechet_distance(mu_1, cv_1, mu_2, cv_2)
    return score

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    m = np.square(mu1 - mu2).sum()
    temp=np.dot(sigma1, sigma2)
    s = scipy.linalg.sqrtm(temp)
    fid = m + np.trace(sigma1 + sigma2 - 2 * s)

    return np.real(fid)
