#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:31:04 2021

@author: krasax
"""

import os, cv2, math
import numpy as np


def getNetworkList(modelfolder):
    '''Returns an array containing all available checkpoints for each network type.
    Each entry in the array has the follwing format
    [networkType:checkpointName, networkType, checkpointName]'''
    networks = getNetworkTypes(modelfolder)

    network_list=[]
    
    for n in networks:
        checkpoints = getCheckpoints(modelfolder, n)
        if len(checkpoints)<1:
            continue
        for c in checkpoints:
            network_list.append([n+':'+c,n,c])
    return network_list

def getNetworkTypes(modelfolder):
    '''Returns all the configured types of neural networks.
    Each network has its own folder in the modelfolder where the folder is the name of the network
    A configured network also has a pipeline_default.config file in its folder'''
    networks = [f for f in os.listdir(modelfolder)
                if not f.startswith('.') and
                os.path.isdir(os.path.join(modelfolder,f)) and
                os.path.isfile(os.path.join(modelfolder,f,'pipeline_default.config'))]
    networks.sort()
    return networks

def getCheckpoints(modelfolder,network_name):
    '''Gets all names of checkpoints in the network's folder'''
    folder = os.path.join(modelfolder, network_name)
    checkpoints = [f[:-len('.index')] for f in os.listdir(folder) 
                   if not f.startswith('.') and f.endswith('.index')]
    checkpoints.sort()
    return checkpoints

def normalizeImage(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = (image-np.mean(image))/np.std(image)
    normImg = np.zeros(image.shape)
    normImg = cv2.normalize(image,normImg,0,255,cv2.NORM_MINMAX)

    normImg = np.floor(normImg).astype(np.uint8)

    normImg = cv2.cvtColor(normImg, cv2.COLOR_GRAY2RGB)
    return normImg

def downscaleImage(targetSize, image, mask=None, coordinates=None):
    
    if targetSize:
        targetSize = correctTargetSize(targetSize)
        if targetSize[0]<image.shape[0] or targetSize[1]<image.shape[1]:
            image = cv2.resize(image,(targetSize[0],targetSize[1]),interpolation=cv2.INTER_CUBIC)
            if mask is not None:
                mask = cv2.resize(mask,(targetSize[0],targetSize[1]),interpolation=cv2.INTER_NEAREST)
            if coordinates:
                coordinates = np.asarray(coordinates)
                yratio = targetSize[0]/image.shape[0]
                xratio = targetSize[1]/image.shape[1]

                coordinates[:,[0,2]] = coordinates[:,[0,2]] * xratio
                coordinates[:,[1,3]] = coordinates[:,[1,3]] * yratio       
                
    if mask is not None and coordinates is not None:
        return image, mask, coordinates
    elif mask is not None:
        return image, mask
    elif coordinates is not None:
        return image, coordinates
    else:
        return image

def splitImage(targetSize, image, overlap=0, coordinates=None, classes=None):
    def _return(image,coordinates):
        if coordinates is None:
            return [image]
        elif classes is None:
            return [image], [coordinates]
        else:
            return [image], [coordinates], [classes]
    
    if not targetSize:
        return _return(image,coordinates,classes)
    targetSize = correctTargetSize(targetSize)
    if targetSize[0]>=image.shape[0] and targetSize[1] >= image.shape[1]:
        return _return(image,coordinates,classes)

    overlap = correctOverlap(overlap)
    
    ystarts, yends = getSplitIndices(image.shape[0], targetSize[0], overlap)
    xstarts, xends = getSplitIndices(image.shape[1], targetSize[1], overlap)
    images, coords, clss = [], [], []
    for ys, ye in zip(ystarts, yends):
        for xs, xe in zip(xstarts, xends):
            
            if len(image.shape) == 3: #RGB
                images.append(image[ys:ye,xs:xe,:])
            elif len(image.shape) == 2: #Gray
                images.append(image[ys:ye,xs:xe])
            else:
                raise ValueError(f'Image should have two or three dimensions. Got {len(image.shape)}.')
                
            if coordinates is not None:
                if not np.any(coordinates):
                    #No coordinates on this image (i.e. coordinates is [])
                    coords.append([])
                    clss.append([])
                else:
                    split_coords=np.asarray(
                         [p for p in coordinates if p[0]>xs and p[2]<xe and
                          p[1]>ys and p[3]<ye] 
                         )
                    if not np.any(split_coords):
                        coords.append([])
                        clss.append([])
                    else:
                        split_coords[:,[0,2]] = split_coords[:,[0,2]] - xs
                        split_coords[:,[1,3]] = split_coords[:,[1,3]] - ys
                        coords.append(split_coords)
                        if classes is not None:
                            split_cls = [c for (c,p) in zip(classes,coordinates) 
                                         if p[0]>xs and p[2]<xe and
                                         p[1]>ys and p[3]<ye]
                            clss.append(split_cls)
     
    if coordinates is None:
        return images
    elif classes is None:
        return images, coords
    else:
        return images, coords, clss
    

def correctTargetSize(targetSize):
    if type(targetSize) is int:
        targetSize=(targetSize, targetSize)
    elif len(targetSize)==1:
        targetSize=(targetSize[0], targetSize[0])
    elif len(targetSize)>2:
        targetSize=targetSize[0:2]
    return targetSize

def upscale_coordinates(coords,targetShape,currentShape):
    if not coords or not currentShape:
        return coords
    currentShape = correctTargetSize(currentShape)
    if currentShape == targetShape:
        return coords
        
    yratio = targetShape[0] / currentShape[0]
    xratio = targetShape[1] / currentShape[1]
    
    upscaled = [[xratio * x, yratio * y] for (x,y) in coords]
    
    return upscaled

def get_coords_from_split(targetShape, splitShape, coords, overlap=0):
    if len(coords) == 1:
        return coords[0]
    splitShape = correctTargetSize(splitShape)
    overlap = correctOverlap(overlap)
    ystarts, _ = getSplitIndices(targetShape[0], splitShape[0], overlap)
    xstarts, _ = getSplitIndices(targetShape[1], splitShape[1], overlap)
    
    assert len(coords) == len(xstarts) * len(ystarts)
    
    coordinates = []
    count=0
    for ys in ystarts:
        for xs in xstarts:
            coord = coords[count]
            
            if coord: #No need to do anything if it is empty list
                conv_coords = [[xs+x, ys+y] for (x,y) in coord]
                coordinates+=conv_coords
                
            count+=1
            
    return coordinates

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
    if overlap is None:
        return 0
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