#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:09:59 2021

@author: krasax
"""

import os, cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from modifiedObjectDetection import model_builder

import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')
plt=matplotlib.pyplot

import utils

PATH_TO_CFG = '/home/krasax/Python_scripts/PSD_Finder/my_models/efficientnet_d2/pipeline.config'
PATH_TO_CKPT = '/home/krasax/Python_scripts/PSD_Finder/clusterTrainedModels/efficientdet_D2/ckpt-55'
PATH_TO_LABELS = '/home/krasax/Python_scripts/PSD_Finder/label_map.pbtxt'
MIN_SCORE = 0.15

output_folder = '/media/krasax/SSD/SerialEM/15E-MS4R-A1-AMPAR_N1/2021-04-30/pred_D2_C55_15Percent'
image_folder= '/media/krasax/SSD/SerialEM/15E-MS4R-A1-AMPAR_N1/2021-04-30/SR_long'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

image_names = [x for x in os.listdir(image_folder) if x.endswith('.tif') and not x.startswith('.')
               and not x.endswith('_mod.tif') and not x.endswith('_morph.tif')]
image_names.sort()

#image_names = ['replica0007.tif']

image_paths = [os.path.join(image_folder,x) for x in image_names]
output_paths = [os.path.join(output_folder,x) for x in image_names]



print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(PATH_TO_CKPT).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


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

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
downscale_targetSize = 2048
split_targetSize = 768
overlap = 10
for image_path,output_path in zip(image_paths,output_paths):

    print('Running inference for {}... '.format(image_path), end='')

    image_np = cv2.imread(image_path)
    image_np = utils.downscaleImage(downscale_targetSize, image_np)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    
    
    normImg = np.zeros(image_np.shape)
    normImg = cv2.normalize(image_np,normImg,0,255,cv2.NORM_MINMAX)
    
    images= utils.splitImage(split_targetSize, normImg, overlap)
    outImgs = []
    for im in images:
        input_tensor = tf.convert_to_tensor(np.expand_dims(im, 0), dtype=tf.float32)
        
        
        detections = detect_fn(input_tensor)
    
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
        
        for box in detections['detection_boxes'][detections['detection_scores']>MIN_SCORE]:
                        
            ymin, xmin, ymax, xmax = box
            ymin = round(ymin * im.shape[0])
            ymax = round(ymax * im.shape[0])     
            xmin = round(xmin * im.shape[1])
            xmax = round(xmax * im.shape[1]) 
            outIm[ymin:ymax,xmin:xmax] = 1
        
        # plt.figure()
        # plt.imshow(im)
        # plt.figure()
        # plt.imshow(outIm)
        
        outImgs.append(outIm)
        
        
        # image_np_with_detections = im.copy()
    
        # image_np_with_detections=viz_utils.visualize_boxes_and_labels_on_image_array(
        #         image_np_with_detections,
        #         detections['detection_boxes'],
        #         detections['detection_classes']+label_id_offset,
        #         detections['detection_scores'],
        #         category_index,
        #         use_normalized_coordinates=True,
        #         max_boxes_to_draw=50,
        #         min_score_thresh=.15,
        #         agnostic_mode=False)
    
        # plt.figure()
        # plt.imshow(image_np_with_detections)
    
    assert len(images) == len(outImgs)
        
    outImage = utils.fuseMasks(normImg.shape, outImgs, overlap)
    cv2.imwrite(output_path, np.uint8(outImage)*255)
    # plt.figure()
    # plt.imshow(normImg)
    # plt.figure()
    # plt.imshow(outImage)
    
    print('Done')
plt.show()
