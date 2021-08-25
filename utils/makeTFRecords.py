#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:38:37 2021

@author: DAvid Kleindienst
"""
import os, cv2, io, math, random
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
from PIL import Image

from object_detection.utils import dataset_util

from utils import utils

def maketfRecordsFromConfig(config,filename,image_output_folder,downscale_targetSize=None,
                            split_targetSize=None,eval_probability=0.15):
    if not os.path.isdir(image_output_folder):
        os.mkdir(image_output_folder)
   
    if not eval_probability:
        filename += '.tfrecord'
        writer = tf.io.TFRecTFRecordWriter(filename)
    else:
        trainName = filename+'_train.tfrecord'
        evalName = filename+'_eval.tfrecord'
            
        train_writer = tf.io.TFRecordWriter(trainName) #create a writer that'll store our data to disk
        eval_writer = tf.io.TFRecordWriter(evalName)
    count = 0
    
    images = file2dict(config,',\t')
    folder = os.path.split(config)[0]
    routes = images['ROUTE']
    
    
    #Find all duplicates of images  
    duplicates=[r for r in routes if r.endswith('_dupl')]
    #Get routes of the original images of these duplicates
    duplicated_images=list(set([r.replace('_dupl','') for r in duplicates]))

    for route in images['ROUTE']:
        
        if eval_probability:
            writer = eval_writer if random.random() < eval_probability else train_writer
        
        if route in duplicates:
            continue #duplicates will be included when duplicated_images are processed
         
        if route in duplicated_images:
            labels = [route+'_mod.tif']
            labels+=[r+'_mod.tif' for r in duplicates if r.startswith(route)]
            out=convertImageToInstance(folder, route+'.tif', labels, image_output_folder,
                                       downscale_targetSize=downscale_targetSize,
                                       split_targetSize=split_targetSize)
        else:
            out = convertImageToInstance(folder,route+'.tif', route+'_mod.tif', image_output_folder,
                                       downscale_targetSize=downscale_targetSize,
                                       split_targetSize=split_targetSize)
        if out and type(out) is list:
            for o in out:
                if o:
                    writer.write(o.SerializeToString())
                    count += 1
        elif out:
            writer.write(out.SerializeToString())
            count += 1
        else:
            print(f'Skipped {route}')
            
        # if count>30:
        #     break
    writer.close()
    print(f"Wrote {count} elements to TFRecord")

def convertImageToInstance(folder,imageFile,labelFile,image_output_folder,backgroundIsWhite=True,
                           downscale_targetSize=None,split_targetSize=None,overlap=None):
    if backgroundIsWhite:
        npFunc=lambda x: 255*np.ones(x)
        minmaxFunc = lambda x: np.min(x, axis=0)
    else:
        npFunc = lambda x: np.zeros(x)
        minmaxFunc = lambda x: np.max(x, axis=0)
    img=cv2.cvtColor(cv2.imread(os.path.join(folder,imageFile)),cv2.COLOR_RGB2GRAY)
    if type(labelFile) is str:
        if os.path.isfile(os.path.join(folder,labelFile)):
            demarc=cv2.cvtColor(cv2.imread(os.path.join(folder,labelFile)),cv2.COLOR_RGB2GRAY)
        else:
            demarc = npFunc(img.shape)
    elif type(labelFile) is list:
        labelFiles = [os.path.join(folder, r) for r in labelFile]
        
            
        masks = np.asarray([cv2.cvtColor(cv2.imread(r),cv2.COLOR_RGB2GRAY) if os.path.isfile(r) else npFunc(img.shape) for r in labelFiles])
        demarc = minmaxFunc(masks)  #Pools the duplicates
       
    else:
        raise ValueError(f'Argument label file needs to be a filepath or list of filepaths. {labelFile} given.')
                            
    
    assert img.shape[0]==demarc.shape[0] and img.shape[1]==demarc.shape[1] and img.shape[0]==img.shape[1]
    
    img,demarc = utils.downscaleImage(downscale_targetSize, img, demarc)
    
    
    if backgroundIsWhite:
        backgroundcolor = np.max(demarc)
    else:
        backgroundcolor = np.min(demarc)
    #Convert Mask to binary
    mask = np.ones(demarc.shape,dtype=demarc.dtype)
    mask[demarc==backgroundcolor]=0
    
    close_dim = 8 #Some smoothing
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((close_dim,close_dim),np.uint8))
    

    
    # if split_targetSize and split_targetSize<img.shape[0]:
    #     num_splits = math.ceil(img.shape[0]/split_targetSize)
    #     examples = []
    #     count=1
    #     for yn in range(num_splits):
    #         for xn in range(num_splits):
    #             y = img.shape[0]-1-split_targetSize if yn==num_splits-1 else yn * split_targetSize
    #             x = img.shape[1]-1-split_targetSize if xn==num_splits-1 else xn * split_targetSize

                
    #             img_split=img[y:y+split_targetSize,x:x+split_targetSize]
    #             mask_split=mask[y:y+split_targetSize,x:x+split_targetSize]
    #             suffix = f'_split_{count}_of_{num_splits*num_splits}'
    #             jpgPath = os.path.join(image_output_folder, imageFile.replace('.tif', suffix+'.jpg'))
                
    #             examples.append(getExampleFromImage(img_split, mask_split, jpgPath))
    #             count += 1
    # else:
    #     jpgPath = os.path.join(image_output_folder, imageFile.replace('.tif', '.jpg'))
    #     return getExampleFromImage(img,mask,jpgPath)
    
    split_images = utils.splitImage(split_targetSize,img,overlap)
    split_masks = utils.splitImage(split_targetSize,mask,overlap)
    examples = []
    for count, (split_img, split_mask) in enumerate(zip(split_images,split_masks)):
        suffix = f'_split_{count+1}_of_{len(split_images)}'
        jpgPath = os.path.join(image_output_folder, imageFile.replace('.tif', suffix+'.jpg'))
        examples.append(getExampleFromImage(split_img, split_mask, jpgPath))
        
    return examples
                                    
    
def getExampleFromImage(img,mask,jpgPath):

    
    #remove the 3 pixels near each border, so no object touches the border
    mask[[0,1,2,-1,-2,-3],:]=0
    mask[:,[0,1,2,-1,-2,-3]]=0
    
    xmins, xmaxs, ymins, ymaxs = [], [], [], [] 
    classes, classes_text = [], []
    height = img.shape[0]
    width = img.shape[1]
    if np.any(mask):
        num_objects, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for x in range(1,num_objects):
            xmins.append(stats[x,0]/width)
            xmaxs.append((stats[x,0]+stats[x,2])/width)
            ymins.append(stats[x,1]/height)
            ymaxs.append((stats[x,1]+stats[x,3])/height)
            classes.append(1)
            classes_text.append('PSD'.encode('utf8'))

    # print(xmins)
    # print(xmaxs)
    # print(ymins)
    # print(ymaxs)
    # print('\n')
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #Normalize image to use whole 8bit space
    normImg = np.zeros(img.shape)
    normImg = cv2.normalize(img,normImg,0,255,cv2.NORM_MINMAX)
    
    if not os.path.isdir(os.path.split(jpgPath)[0]):
        os.mkdir(os.path.split(jpgPath)[0])
        
    cv2.imwrite(jpgPath,normImg)
    with tfv1.gfile.GFile(jpgPath, 'rb') as fid:
        encoded_jpg = fid.read()
        
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    if image.format != 'JPEG':
        raise ValueError(f'Image format not JPEG but {image.format}')
    if xmins:
        data = {
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            #What to do with these?
            'image/filename': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/format': dataset_util.bytes_feature('.jpg'.encode('utf8')),
    
        }
    else: #No objects in this image
        return None
        data = {
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
            
            #What to do with these?
            'image/filename': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/format': dataset_util.bytes_feature('.jpg'.encode('utf8')),
    
        }
        
    return tf.train.Example(features=tf.train.Features(feature=data))

# def parse_tfrecord(record):
#     example = tf.train.Example()
#     example.ParseFromString(record)
#     feat = example.features.feature

#     filename = feat['image/filename'].bytes_list.value[0].decode("utf-8")
#     img =  feat['image/encoded'].bytes_list.value[0]
#     label = feat['image/object/bbox/xmin'].bytes_list.value[0].decode("utf-8")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def file2dict(filename,delimiter=','):
    with open(filename,'r') as f:
        lines=f.readlines()
    sp=[x.strip('\n').split(delimiter) for x in lines]
    header=sp[0]
    sp=sp[1:]
    dic=dict()
    for i,h in enumerate(header):
        elements=[x[i] for x in sp]
        dic[h]=elements
    return dic

#filename = 'example_cat.jpg'
#image_format = b'jpg'






if __name__=='__main__':
    if 1:
        print('xz')
        config='/media/krasax/SSD/SerialEM/TrainData/ForTraining.dat'
        filename='/home/krasax/Python_scripts/PSD_Finder/split1024_onlyObj'
        folder='/home/krasax/Python_scripts/PSD_Finder/split1024_onlyObj/'
        maketfRecordsFromConfig(config, filename, folder, downscale_targetSize=2048,split_targetSize=1080)
        #1080 so it makes images with about 10% overlap
        #Todo: figure out a better overlap way which also works if theres more than 2 images per dimension
    else:
        pass
    
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
