#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:38:37 2021

@author: David Kleindienst
"""
import os, cv2, io, math, random, time, asyncio
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
from PIL import Image

from object_detection.utils import dataset_util

from utils import utils


def maketfRecords(input_type,input_items, filename, image_output_folder, 
                  class_name=None,
                  downscale_targetSize=None, split_targetSize=None,
                  eval_probability=0.15, overlap=None,
                  progressHandle=None, app=None):
    '''
    input type can be Darea, folder or XML
    '''
    if class_name is None and input_type != 'XML':
        raise ValueError(f'class_name needs to be provided for input_type {input_type}!')
    
    start=time.time()
    
    if os.path.isfile('tfRecord_Classes_tmp.pickle'):
        os.remove('tfRecord_Classes_tmp.pickle')
    
    if type(input_items) is str:
        input_items = [input_items]
    if not os.path.isdir(image_output_folder):
        os.mkdir(image_output_folder)
   
    if filename.endswith('.tfrecord'):
        filename=filename[:-9]
    
    count = 0
    nrFiles = len(input_items)
    for i,item in enumerate(input_items):
        if nrFiles>1:
            suffix = '-' + str(i).zfill(5) + '-of-' + str(nrFiles).zfill(5)
        else:
            suffix = ''
        if not eval_probability:
            filename += '.tfrecord' + suffix
            writer = tf.io.TFRecTFRecordWriter(filename)
        else:
            trainName = filename+'_train.tfrecord' + suffix
            evalName = filename+'_eval.tfrecord' + suffix
                
            train_writer = tf.io.TFRecordWriter(trainName) 
            eval_writer = tf.io.TFRecordWriter(evalName)
            writer=(train_writer,eval_writer)
        
        if input_type=='Darea':
            images = file2dict(item,',\t')
            folder = os.path.split(item)[0]
            routes = images['ROUTE']
        
            
            #Find all duplicates of images  
            duplicates = [r for r in routes if r.endswith('_dupl')]
            #Get routes of the original images of these duplicates
            duplicated_images = list(set([r.replace('_dupl','') for r in duplicates]))
        elif input_type=='folder' or input_type=='XML':
            #This option assumes no duplicates because _mod images were not made by Darea
            duplicates = []
            duplicated_images = []
            
            folder = item
            routes = os.listdir(folder)
            routes = [r[:-4] for r in routes if not r.startswith('.')
                     and r.endswith('.tif') and not r.endswith('_mod.tif')]
            
        else:
            raise ValueError(f'Input type {input_type} not known. Allowed input_types are "Darea", "folder" and "XML"')
        
        random.shuffle(routes)
    
        for route in routes:
            if progressHandle is not None and app is not None:
                progressHandle.setText(f'Processing image {count}')
                app.processEvents()
                
            if not os.path.isfile(os.path.join(folder,route+'.tif')):
                print(f'Image {route} not found. Skipped image.')
                continue
            
            if eval_probability:
                writer = eval_writer if random.random() < eval_probability else train_writer
            
            if route in duplicates:
                continue #duplicates will be included when duplicated_images are processed
             
            if route in duplicated_images:
                labels = [route+'_mod.tif']
                labels += [r+'_mod.tif' for r in duplicates if r.startswith(route)]
                out = convertImageToInstance(folder, route+'.tif', labels, image_output_folder,
                                             class_name,
                                           downscale_targetSize=downscale_targetSize,
                                           split_targetSize=split_targetSize, overlap=overlap)
            else:
                if input_type=='XML':
                    out = convertImageAndXMLToInstance(folder, route+'.tif',
                                                 route+'.xml', 
                                                 image_output_folder,
                                                 downscale_targetSize=downscale_targetSize,
                                                 split_targetSize=split_targetSize,
                                                 overlap=overlap)
                else:
                    out = convertImageToInstance(folder,route+'.tif', route+'_mod.tif', 
                                                 image_output_folder, class_name,
                                                 downscale_targetSize=downscale_targetSize,
                                                 split_targetSize=split_targetSize, 
                                                 overlap=overlap)
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
            
        if not eval_probability:    
            writer.close()
        else:
            train_writer.close()
            eval_writer.close()
    
        
        
    print(f"Wrote {count} elements to TFRecord in {round(time.time()-start)} seconds")
    if progressHandle is not None and app is not None:
        progressHandle.setText(f"Wrote {count} elements to TFRecord in {round(time.time()-start)} seconds")
        app.processEvents()
    
    if os.path.isfile('tfRecord_Classes_tmp.pickle'):
        os.remove('tfRecord_Classes_tmp.pickle')
        


def maketfRecordsFromConfig(configs,filename,image_output_folder,
                            class_name,downscale_targetSize=None,
                            split_targetSize=None,eval_probability=0.15, overlap=None,
                            progressHandle=None, app=None):
    #This function may be obsolete
    start=time.time()
    if type(configs) is str:
        configs = [configs]
    if not os.path.isdir(image_output_folder):
        os.mkdir(image_output_folder)
   
    if filename.endswith('.tfrecord'):
        filename=filename[:-9]
    
    count = 0
    nrFiles = len(configs)
    for i,config in enumerate(configs):
        if nrFiles>1:
            suffix = '-' + str(i).zfill(5) + '-of-' + str(nrFiles).zfill(5)
        else:
            suffix = ''
        if not eval_probability:
            filename += '.tfrecord' + suffix
            writer = tf.io.TFRecTFRecordWriter(filename)
        else:
            trainName = filename+'_train.tfrecord' + suffix
            evalName = filename+'_eval.tfrecord' + suffix
                
            train_writer = tf.io.TFRecordWriter(trainName) 
            eval_writer = tf.io.TFRecordWriter(evalName)
    
        images = file2dict(config,',\t')
        folder = os.path.split(config)[0]
        routes = images['ROUTE']
        
        
        #Find all duplicates of images  
        duplicates=[r for r in routes if r.endswith('_dupl')]
        #Get routes of the original images of these duplicates
        duplicated_images=list(set([r.replace('_dupl','') for r in duplicates]))
    
        for route in images['ROUTE']:
            if progressHandle is not None and app is not None:
                progressHandle.setText(f'Processing image {count}')
                app.processEvents()
            if not os.path.isfile(os.path.join(folder,route+'.tif')):
                print(f'Image {route} not found. Skipped image.')
                continue
            
            if eval_probability:
                writer = eval_writer if random.random() < eval_probability else train_writer
            
            if route in duplicates:
                continue #duplicates will be included when duplicated_images are processed
             
            if route in duplicated_images:
                labels = [route+'_mod.tif']
                labels+=[r+'_mod.tif' for r in duplicates if r.startswith(route)]
                out = convertImageToInstance(folder, route+'.tif', labels, 
                                             image_output_folder, class_name,
                                           downscale_targetSize=downscale_targetSize,
                                           split_targetSize=split_targetSize, overlap=overlap)
            else:
                out = convertImageToInstance(folder,route+'.tif', route+'_mod.tif', 
                                             image_output_folder, class_name,
                                           downscale_targetSize=downscale_targetSize,
                                           split_targetSize=split_targetSize, overlap=overlap)
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
                

        if not eval_probability:    
            writer.close()
        else:
            train_writer.close()
            eval_writer.close()
    print(f"Wrote {count} elements to TFRecord in {round(time.time()-start)} seconds")
    if progressHandle is not None and app is not None:
        progressHandle.setText(f"Wrote {count} elements to TFRecord in {round(time.time()-start)} seconds")
        app.processEvents()

def convertImageAndXMLToInstance(folder,imageFile,xml_file,image_output_folder,
                                 downscale_targetSize=None,split_targetSize=None,overlap=None):
    import xml.etree.ElementTree as ET
    img=cv2.cvtColor(cv2.imread(os.path.join(folder,imageFile)),cv2.COLOR_RGB2GRAY)
    
    if os.path.isfile(os.path.join(folder,xml_file)):
        tree=ET.parse(os.path.join(folder,xml_file))
        root=tree.getroot()
        
        #Make sure image size is same as described in xml
        assert img.shape[0]==int(root.find('size').find('height').text)
        assert img.shape[1]==int(root.find('size').find('width').text)
        boxes=[]
        classes=[]
        for obj in root.findall('object'):
            # bndbox = [xmin, ymin, xmax, ymax]
            bndbox = [int(obj.find('bndbox').find('xmin').text),
                      int(obj.find('bndbox').find('ymin').text),
                      int(obj.find('bndbox').find('xmax').text),
                      int(obj.find('bndbox').find('ymax').text)]
            boxes.append(bndbox)
            classes.append(obj.find('name').text)
    else:
        boxes = []
        classes = []
        
    img, boxes = utils.downscaleImage(downscale_targetSize, img, coordinates=boxes)    
    
    split_images, split_boxes, split_classes = utils.splitImage(split_targetSize,img, 
                                                                coordinates=boxes, 
                                                                classes=classes,
                                                                overlap=overlap)
    examples=[]
    for count, (split_img, split_box,split_class) in enumerate(zip(split_images,split_boxes, split_classes)):
        suffix = f'_split_{count+1}_of_{len(split_images)}'
        jpgPath = os.path.join(image_output_folder, imageFile.replace('.tif', suffix+'.jpg'))
        examples.append(getExampleFromImageAndBox(split_img, split_box,split_class, jpgPath))
        
    return examples

    
def convertImageToInstance(folder,imageFile,labelFile,image_output_folder,
                           class_name,backgroundIsWhite=True,
                           downscale_targetSize=None,split_targetSize=None,overlap=None):
    if backgroundIsWhite:
        npFunc=lambda x: 255*np.ones(x,dtype=np.uint8)
        minmaxFunc = lambda x: np.min(x, axis=0)
    else:
        npFunc = lambda x: np.zeros(x,dtype=np.uint8)
        minmaxFunc = lambda x: np.max(x, axis=0)
    try:
        img=cv2.cvtColor(cv2.imread(os.path.join(folder,imageFile)),cv2.COLOR_RGB2GRAY)
    except:
        print(imageFile)
        raise
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
        
    split_images = utils.splitImage(split_targetSize,img,overlap)
    split_masks = utils.splitImage(split_targetSize,mask,overlap)
    examples = []
    for count, (split_img, split_mask) in enumerate(zip(split_images,split_masks)):
        suffix = f'_split_{count+1}_of_{len(split_images)}'
        jpgPath = os.path.join(image_output_folder, imageFile.replace('.tif', suffix+'.jpg'))
        examples.append(getExampleFromImage(split_img, split_mask, jpgPath, class_name))
        
    return examples
 
def getExampleFromImageAndBox(img,boxes,classes_text,jpgPath):
    import pickle
    
    #Classes need to remain consistent between calls to this function
    #So its necessary to save and load to disk
    if os.path.isfile('tfRecord_Classes_tmp.pickle'):
        class_dict=pickle.load(open('tfRecord_Classes_tmp.pickle','rb'))
    else:
        class_dict=dict()
    
    height = img.shape[0]
    width = img.shape[1]
    
    classes=[]
    for c in classes_text:
        if c in class_dict:
            classes.append(c)
        else:
            if not class_dict:
                class_dict[c] = 1
                classes.append(1)
            else:
                val = max(class_dict.values())+1
                class_dict[c] = val
                classes.append(val)
        
    classes_text = [c.encode('utf8') for c in classes_text]
    
    pickle.dump(class_dict, open('tfRecord_Classes_tmp.pickle','wb'))
    
    height = img.shape[0]
    width = img.shape[1]

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #Normalize image to use whole 8bit space
    normImg = np.zeros(img.shape)
    normImg = cv2.normalize(img,normImg,0,255,cv2.NORM_MINMAX)
    
    if not os.path.isdir(os.path.split(jpgPath)[0]):
        os.mkdir(os.path.split(jpgPath)[0])     
               
        
    #Saving and then reading jpg seems useless, but somehow is necessary
    cv2.imwrite(jpgPath,normImg)
    with tfv1.gfile.GFile(jpgPath, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    
    if image.format != 'JPEG':
        raise ValueError(f'Image format not JPEG but {image.format}')
        
    if len(boxes)>0:
        data = {
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
            'image/object/bbox/xmin': dataset_util.float_list_feature(boxes[:,0]),
            'image/object/bbox/xmax': dataset_util.float_list_feature(boxes[:,2]),
            'image/object/bbox/ymin': dataset_util.float_list_feature(boxes[:,1]),
            'image/object/bbox/ymax': dataset_util.float_list_feature(boxes[:,3]),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/filename': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/format': dataset_util.bytes_feature('.jpg'.encode('utf8')),
    
        }
    else:
        data = {
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
            
            'image/filename': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/format': dataset_util.bytes_feature('.jpg'.encode('utf8')),
    
        }
        
    return tf.train.Example(features=tf.train.Features(feature=data))
    
def getExampleFromImage(img,mask,jpgPath,class_name):

    
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
            classes_text.append(class_name.encode('utf8'))

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #Normalize image to use whole 8bit space
    normImg = np.zeros(img.shape)
    normImg = cv2.normalize(img,normImg,0,255,cv2.NORM_MINMAX)
    
    if not os.path.isdir(os.path.split(jpgPath)[0]):
        os.mkdir(os.path.split(jpgPath)[0])
    #Saving and then reading jpg seems useless, but somehow is necessary
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
            'image/filename': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(jpgPath.encode('utf8')),
            'image/format': dataset_util.bytes_feature('.jpg'.encode('utf8')),
    
        }
    else: #No objects in this image
        #return None
        data = {
            'image/height' : dataset_util.int64_feature(height),
            'image/width' : dataset_util.int64_feature(width),
            'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
            
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
