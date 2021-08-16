#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:31:04 2021

@author: krasax
"""

import os, cv2, math
import numpy as np


def normalizeImage(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = (image-np.mean(image))/np.std(image)
    normImg = np.zeros(image.shape)
    normImg = cv2.normalize(image,normImg,0,255,cv2.NORM_MINMAX)

    normImg = np.floor(normImg).astype(np.uint8)

    normImg = cv2.cvtColor(normImg, cv2.COLOR_GRAY2RGB)
    return normImg

def downscaleImage(targetSize, image, mask=None):
    
    if targetSize:
        if type(targetSize) is int:
            targetSize=(targetSize, targetSize)
        if targetSize[0]<image.shape[0] or targetSize[1]<image.shape[1]:
            image = cv2.resize(image,(targetSize[0],targetSize[1]),interpolation=cv2.INTER_CUBIC)
            if mask:
                mask = cv2.resize(mask,(targetSize[0],targetSize[1]),interpolation=cv2.INTER_NEAREST)
        
    if mask:
        return image, mask
    else:
        return image
    
    
def splitImage(targetSize, image, overlap=0):
    
    if not targetSize:
        return [image]
    if type(targetSize) is int:
        targetSize = (targetSize, targetSize)
    if targetSize[0]>=image.shape[0] and targetSize[1] >= image.shape[1]:
        return [image]
    
    overlap = correctOverlap(overlap)
    
    ystarts, yends = getSplitIndices(image.shape[0], targetSize[0], overlap)
    xstarts, xends = getSplitIndices(image.shape[1], targetSize[1], overlap)
    
    images = []
    for ys, ye in zip(ystarts, yends):
        for xs, xe in zip(xstarts, xends):
            
            if len(image.shape) == 3: #RGB
                images.append(image[ys:ye,xs:xe,:])
            elif len(image.shape) ==1: #Gray
                images.append(image[ys:ye,xs:xe])
            else:
                raise ValueError(f'Image should have two or three dimensions. Got {len(image.shape)}.')
                
    return images

def fuseMasks(targetShape, images, overlap=0):
    if len(images) == 1:
        return images[0]
    
    overlap = correctOverlap(overlap)
    ystarts, yends = getSplitIndices(targetShape[0], images[0].shape[0], overlap)
    xstarts, xends = getSplitIndices(targetShape[1], images[0].shape[1], overlap)
    
    assert len(xstarts) * len(ystarts) == len(images)
    
    #If there is an overlap of 10%, first each image should fill 5% of that overlap
    #Eg. 1st image goes from 0 to 1000 and 2nd image goes from 900 to 1900
    #Then first image should fill 0->950 and 2nd image should fill from 950
    
    
    
    outputImage = np.zeros((targetShape[0],targetShape[1])) #Mask is Grayscale
    count=0
    for yc, (ys, ye) in enumerate(zip(ystarts, yends)):
        if yc>0:
            if ystarts[yc-1] > ys:
                ys = round((ystarts[yc-1] + ys)/2)
                
        for xc, (xs, xe) in enumerate(zip(xstarts, xends)):
            if xc>0:
                if xstarts[xc-1] > xs:
                    xs = round((xstarts[xc-1] + xs)/2)
            im=images[count]
            
            outputImage[ys:ye,xs:xe] = im
            count+=1
            
    return outputImage
 
def correctOverlap(overlap):
    #Makes overlap a ratio between 0 and 1 if a percentage was specified
    
    if overlap > 1:
    #Probably percent
        if overlap > 100:
            raise ValueError(f'Overlap should be either a ratio between 0 and 1 or a percentage between 1 and 100. Got {overlap}')
        overlap = overlap/100
    
    return overlap
def getSplitIndices(imdim,target,overlap=0):
    if target>=imdim:
        return [0], [imdim]
    
    #First split from start till target
    starts = [0]
    ends = [target]
    newStart = math.floor(target*(1-overlap))
    newEnd = newStart+target
    
    while newEnd < imdim:
        #Intermediate images with overlap
        starts.append(newStart)
        ends.append(newEnd)
        
        newStart = math.floor(starts[-1]+target*(1-overlap))
        newEnd = newStart+target
    
    #Last split until end
    ends.append(imdim)
    starts.append(imdim-target)
    return starts, ends    