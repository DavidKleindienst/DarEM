#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:09:59 2021

@author: krasax
"""

import os, cv2, json, time
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

#from object_detection.utils import label_map_util
from object_detection.utils import config_util
from modifiedObjectDetection import model_builder

#import matplotlib
#matplotlib.use('module://ipykernel.pylab.backend_inline')
#plt=matplotlib.pyplot

from utils import utils

def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_path', type=str, required=True, 
                        help='Path to the pipeline.config file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the checkpoint file')
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Folder containing the images to be predicted on')
    parser.add_argument('--min_score', type=float, default=0.15, 
                        help='Minimum score for a detection to be taken')
    parser.add_argument('--output_folder', type=str, default=None, 
                        help='Folder for output images. Keep as None for disabling image output.')
    parser.add_argument('--coordinate_file', type=str, default=None, 
                        help='Path where coordinate.json file should be created. Keep as None to not create coordinate file')
    parser.add_argument('--downscale_targetsize', type=int, nargs='+', default=2048, 
                        help='Targetsize of image after downscaling')
    parser.add_argument('--split_targetsize', type=int, nargs='+', default=768,
                        help='Targetsize of image after splitting')
    parser.add_argument('--overlap', type=float, default=0.1, 
                        help='Overlap between split images')
    
    if args is None:
        args=parser.parse_args()
    else:
        args=parser.parse_args(args)
    
    if args.output_folder is None and args.coordinate_file is None:
        raise ValueError('Either --output_folder or --coordinate_file has to be set!')
        
        
    if args.output_folder and not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)
    
    if args.coordinate_file:
        coordinates=dict()
        if os.path.isdir(args.coordinate_file):
            args.coordinate_file = os.path.join(args.coordinate_file, 'centroids.json')
        elif not args.coordinate_file.endswith('.json'):
            args.coordinate_file += '.json'
    
    #PATH_TO_CFG = '/home/krasax/Python_scripts/PSD_Finder/my_models/efficientnet_d2/pipeline.config'
    #PATH_TO_CKPT = '/home/krasax/Python_scripts/PSD_Finder/clusterTrainedModels/efficientdet_D2/ckpt-55'
    #PATH_TO_LABELS = '/home/krasax/Python_scripts/PSD_Finder/label_map.pbtxt'
    #MIN_SCORE = 0.15
    
    #output_folder = '/media/krasax/SSD/SerialEM/15E-MS4R-A1-AMPAR_N1/2021-04-30/pred_D2_C55_15Percent'
    #image_folder= '/media/krasax/SSD/SerialEM/15E-MS4R-A1-AMPAR_N1/2021-04-30/SR_long'
    
    
    
    image_names = [x for x in os.listdir(args.input_folder) if x.endswith('.tif') and not x.startswith('.')
                   and not x.endswith('_mod.tif') and not x.endswith('_morph.tif')]
    image_names.sort()
    
    #image_names = ['replica0007.tif']
    
    image_paths = [os.path.join(args.input_folder,x) for x in image_names]
    output_paths = [os.path.join(args.output_folder,x) if args.output_folder else None for x in image_names]
    
    
    
    print('Loading model... ', end='')
    start_time = time.time()
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(args.pipeline_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(args.checkpoint_path).expect_partial()
    current_time = time.time()
    print('Done! Took {} seconds'.format(current_time - start_time))
    #downscale_targetSize = 2048
    #split_targetSize = 768
    #overlap = 10
    for image_path,output_path in zip(image_paths,output_paths):
    
        print('Running inference for {}... '.format(image_path), end='')
    
        image_np = cv2.imread(image_path)
        originalSize = image_np.shape
        image_np = utils.downscaleImage(args.downscale_targetsize, image_np)
    
        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()
    
        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
        
        
        normImg = np.zeros(image_np.shape)
        normImg = cv2.normalize(image_np,normImg,0,255,cv2.NORM_MINMAX)
        
        images = utils.splitImage(args.split_targetsize, normImg, args.overlap)
        outImgs = []
        imCoords = []
        for im in images:
            input_tensor = tf.convert_to_tensor(np.expand_dims(im, 0), dtype=tf.float32)
            
            
            detections = detect_fn(input_tensor,detection_model)
        
            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
        
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
            label_id_offset = 1
            
            outIm = np.zeros((im.shape[0],im.shape[1]))
            splitCoords=[]
            
            for box in detections['detection_boxes'][detections['detection_scores']>args.min_score]:
                            
                ymin, xmin, ymax, xmax = box
                ymin = round(ymin * im.shape[0])
                ymax = round(ymax * im.shape[0])     
                xmin = round(xmin * im.shape[1])
                xmax = round(xmax * im.shape[1]) 
                
                if args.output_folder:
                    outIm[ymin:ymax,xmin:xmax] = 1
                
                    
                
                splitCoords.append([xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2])
                
                # plt.figure()
                # plt.imshow(im)
                # plt.figure()
                # plt.imshow(outIm)
            if args.output_folder:    
                outImgs.append(outIm)
            imCoords.append(splitCoords)
            

        if args.output_folder:    
            assert len(images) == len(outImgs)
                
            outImage = utils.fuseMasks(normImg.shape, outImgs, args.overlap)
            cv2.imwrite(output_path, np.uint8(outImage)*255)
        
        if args.coordinate_file:
            assert len(images) == len(imCoords)
            
            imageCoords = utils.get_coords_from_split(normImg.shape, args.split_targetsize,
                                                         imCoords, args.overlap)
            k = filepath_to_name(image_path)
            
            coordinates[k] = utils.upscale_coordinates(imageCoords, originalSize[0:2], normImg.shape[0:2])
            
        print(f'Done. Took {time.time()-current_time} seconds')
        current_time = time.time()
        
    #plt.show()
    if args.coordinate_file:
        with open(args.coordinate_file, 'w') as f:
            json.dump(coordinates,f)

    print(f'Finished. Took {time.time()-start_time} seconds for {len(image_paths)} images.')
    
@tf.function
def detect_fn(image,detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def filepath_to_name(full_name,remove_Mod=False):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    if remove_Mod and file_name.endswith('_mod'):
        file_name=file_name[0:-4]
    return file_name

if __name__=='__main__':
    
    main(None)

