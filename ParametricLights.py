
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.

# HDR Environment Map Estimation for Real-Time Augmented Reality, CVPR 2021.
# Reference implementation of the AngularError metric used in the above paper.

# The parametric light detection and representation is inspired from the following paper
# Marc-André Gardner, Yannick Hold-Geoffroy, Kalyan Sunkavalli, Christian Gagné, and Jean-François Lalonde
# Deep Parametric Indoor Lighting Estimation
# International Conference on Computer Vision (ICCV), 2019



import numpy as np
from skimage import measure
from skimage import morphology
from skimage.morphology import square
import cv2

# Helpers to get parametric lights from an HDR equirectangular model,
# and calculate angular error between them

def extract_lights_from_equirectangular_image(I):
    light_seg,W=findLightBlobs(I)
    param_lights=turnLightBlobsToParams(light_seg,I)
    return param_lights

def calculate_angular_error_metric(gt_lights, my_lights, top_k=5):
        # for each gt light find closest predicted light
        if len(gt_lights) < 1 or len(my_lights) < 1:
            print('No lights here')
            return -1
        # trying to compensate for  python sizing
        if len(np.shape(gt_lights)) < 2:
            # there is just 1 light
            gt_lights = np.stack((gt_lights, gt_lights))
        if len(np.shape(my_lights)) < 2:
            # there is just 1 light
            my_lights = np.stack((my_lights, my_lights))

        top_k = np.minimum(top_k, np.shape(gt_lights)[0])
        top_k = np.minimum(top_k, np.shape(my_lights)[0])

        err = 0.0
        if len(gt_lights) > 0 and len(my_lights) > 0:
            for i in range(0, top_k):
                mne = float("inf")
                for j in range(0, top_k):
                    e_ = angularErrorBetweenLights(gt_lights[i], my_lights[j])
                    if e_ < mne:
                        mne = e_
                err += mne

            for i in range(0, top_k):
                mne = float("inf")
                for j in range(0, top_k):
                    e_ = angularErrorBetweenLights(my_lights[i], gt_lights[j])
                    if e_ < mne:
                        mne = e_

                err += mne

            err /= (top_k + top_k)

        return err


# --------- Functions that perform different steps in the extraction of lights & calculation of errors ------
# Weigh the equirectangular image to compensate spherical projection
def panoweight(im):
    hh, ww = np.shape(im)[0:2]
    offset = np.pi / (2 * hh)
    ang = np.linspace(offset, np.pi - offset, hh)
    wt = np.sin(ang)
    wt2 = np.transpose(np.tile(wt, (ww, 1)))
    J = im * wt2
    return J

# For HDR image I, find segments of light sources
def findLightBlobs(I):
    # per light blob the ratio between max and lowest intensity
    energy_ratio = 0.3
    # (1-min_energy_wrt_max)*total_energy should be explained by the lights
    min_energy_wrt_max = 0.2

    min_pixels_left = np.shape(I)[0] * np.shape(I)[1] * 0.1
    if len(np.shape(I)) > 2 and np.shape(I)[2] > 1:
        rgb2gr = [0.2126, 0.7152, 0.0722]
        G=I[:,:,0]*0
        for c in range(3):
            G = I[:,:,c]*rgb2gr[c]
    else:
        G = I

    k = int(np.shape(I)[0] * 0.01 / 2) * 2 + 1
    G = cv2.GaussianBlur(G, (k, k), 0)
    W = panoweight(G)
    # 1 where pixels yet not yet accounted for
    mask = np.ones_like(W)
    all_lights = np.zeros_like(W)

    total_energy = np.sum(W)
    cnt = 1
    while 1:
        # find the next peak value
        seed = np.amax(W * mask)
        # find all values within energy_ratio of seed
        curr_light = np.zeros_like(W)
        curr_light[W * mask >= seed * energy_ratio] = 1

        # mask out the connected component to seed
        cc = measure.label(curr_light)
        # get the component where seed was
        seed_pos_1, seed_pos_2 = np.where(W * mask >= seed)
        # mark the light
        for c_ in cc[seed_pos_1, seed_pos_2]:
            if c_:
                all_lights[cc == c_] = cnt
                # mask it out for next round
                mask[cc == c_] = 0
        cnt += 1

        if np.sum(mask) < min_pixels_left or cnt > 15:
            break
        if np.sum(mask * W) < min_energy_wrt_max * total_energy:
            # there is less than min_energy_wrt_max left in the image
            break
    all_lights_merge = all_lights.copy()

    # Below are heuristics designed to tackle cases that occur in some images
    if 1:
        # Eliminate the cases where due to lack of sharp lights, almost the entire image/background is detected as one source.
        # This should be handled in the ambient component not light segments.
        props = measure.regionprops(all_lights_merge.astype(np.int))
        img_area = np.shape(I)[0] * np.shape(I)[1]
        for p in props:
            if p.area / img_area > 0.5 or p.major_axis_length / np.shape(I)[1] > 0.9:
                all_lights_merge[all_lights_merge == p.label] = 0
    if 1:
        # Merging of lights. Due to the thresholds, its possible to get nested lights in some cases.
        # Hence we merge such cases

        all_lights_merge[all_lights_merge > 0] = 1
        all_lights_merge = morphology.dilation(all_lights_merge, square(2)).astype(np.int)
        all_lights_merge = measure.label(all_lights_merge)

    return all_lights_merge, W


# Do the fitting on the segments.
def turnLightBlobsToParams(all_lights, I):
    height, width = np.shape(I)[0:2]
    mx = int(np.amax(all_lights))
    lights = []

    for i in range(1, mx + 1):
        my_lights = 0 * all_lights.copy()
        my_lights[all_lights == i] = 1
        contours, _ = cv2.findContours(my_lights.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # find the centroid and axis of this light blob
            if len(c) > 5:
                (lx, ly), (mja, mna), angle = cv2.fitEllipse(c)
                if lx < 0 or ly < 0 or lx > width or ly > height:
                    (lx, ly), crad = cv2.minEnclosingCircle(c)
                    mja = mna = crad

            else:
                lx, ly = np.squeeze(np.mean(c, axis=0))
                mja = 1.0
                mna = 1.0

            mask = np.tile(np.expand_dims(my_lights, -1), (1, 1, 3))
            colr = np.sum(I * mask, axis=(0, 1)) / np.sum(mask[:, :, 0])
            el = (ly + 1.0) * 1.0 / height * np.pi
            az = (1.0 - (lx + 1.0) * 1.0 / (2.0 * height)) * 2.0 * np.pi
            xx = np.sin(el) * np.cos(az)
            yy = np.sin(el) * np.sin(az)
            zz = np.cos(el)
            sz = np.arccos(mja / mna)
            if sz == 0:
                sz = 2 * np.pi / (width * width)
            lights.append(([lx, ly, mja, mna,0], [el, az], [xx, yy, zz, sz], colr))

    return lights


# Given light1 and light2 obtained from two HDR images (using extractLightsFromHDRImage)
# calculate the angular error

def angularErrorBetweenLights(light1,light2):

    first_light = np.asarray(light1[2]) * [1, 1, 1, 0]
    my_light = np.asarray(light2[2]) * [1, 1, 1, 0]
    mse = np.sqrt(np.sum((my_light - first_light) ** 2))
    return 2.0 * np.arcsin(0.5 * mse)
