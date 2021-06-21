# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.

# HDR Environment Map Estimation for Real-Time Augmented Reality, CVPR 2021.
# Demo application using the reference implementations of the AngularError and FID metric used in the above paper.

import cv2
import numpy as np
import random
from ParametricLights import extract_lights_from_equirectangular_image, calculate_angular_error_metric
from FIDCalculations import calculate_fid, inception_feature_for_equirectangular, incp_img_size, incp_model

def draw_lights_on_image(I_,light_params):
    from matplotlib import cm
    cnt = 1
    N = len(light_params)
    I=I_.copy()
    for l_ in light_params:
        l = l_[0]
        clr = np.asarray(cm.hot(cnt)[0:3])*255
        cv2.ellipse(I, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), int(l[4]), 0, 360, clr)
        cv2.circle(I, (int(l[0]), int(l[1])), 2, clr, -1)
        cnt += 1

    return I*255

def create_equirectangular_image_with_blob(lat,long,radius,noise=0.0):
    v = int(lat/180.0*128)
    u = int(long/360.0*256)
    im = np.random.uniform(0.0, noise, (128, 256, 3))
    cv2.circle(im, (u,v), radius, (10, 10, 10), -1)
    im = cv2.GaussianBlur(im, (19, 19), -1)
    return im

if __name__ == '__main__':

    #  **************** Angular Error Metric ****************
    #  Create two equirectangular images with light sources 180 degrees apart
    im1 = create_equirectangular_image_with_blob(90,45,3,noise=0.1)
    im2 = create_equirectangular_image_with_blob(90,45+180,3,noise=0.1)

    # Extract lights per image. Each contains a list of individual lights described
    # using the following parameters in order:
    # [lx, ly, mja, mna,0]: Image X & Y, Major & minor axis of ellipse
    # [el, az]: Elevation & Azimuth converted from equirectangular co-ordinate
    # [xx, yy, zz, sz]: Vector for center of light (used to easily calculate angular error) and angular size
    # color: mean color

    l1 = extract_lights_from_equirectangular_image(im1)
    l2 = extract_lights_from_equirectangular_image(im2)

    # Visualize the extracted light parameters and save the images
    im1_vis = draw_lights_on_image(im1, l1)
    cv2.imwrite('im1_lights.jpg', im1_vis[:,:,::-1])
    im2_vis = draw_lights_on_image(im2, l2)
    cv2.imwrite('im2_lights.jpg', im2_vis[:,:,::-1])

    # Compute angular error between the two images
    err = calculate_angular_error_metric(l1, l2)
    if err<0:
        print('One or more images did not have detected lights')
    else:
        print('Angular error in degrees:', err*180/np.pi)

    #  **************** FID ****************
    # Demo of how to use the FIDCalculation functions.
    # NOTE: FID should be calculated with at least a few thousand equirectangular images to be meaningful,
    # here we simply demonstrate the function call and usage on a much smaller set of images.
    # Create two sets of equirectangular images and extract features from them to compute FID between.

    eq_image_set1=[]
    eq_image_set2=[]
    for i in range(0, 5):
        eq1 = create_equirectangular_image_with_blob(random.randint(0,180), random.randint(0,360), 3,
                                                     noise=np.random.uniform(0,1.0))
        eq2 = create_equirectangular_image_with_blob(random.randint(0,180), random.randint(0,360), 5,
                                                     noise=np.random.uniform(0,1.0))

        # The version of inception model used in our code assumed LDR images processed to be in [-1,1] range

        eq_image_set1.append(eq1*2.0-1.0)
        eq_image_set2.append(eq2*2.0-1.0)

    ft1 = inception_feature_for_equirectangular(eq_image_set1, incp_img_size, incp_model)
    ft2 = inception_feature_for_equirectangular(eq_image_set2, incp_img_size, incp_model)

    fid = calculate_fid(ft1, ft2)
    print('FID:', fid)
