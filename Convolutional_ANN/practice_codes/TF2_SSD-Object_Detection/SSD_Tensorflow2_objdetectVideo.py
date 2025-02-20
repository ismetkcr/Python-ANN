# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 06:07:29 2024

@author: ismt
"""

import os
import imageio
import subprocess
import tensorflow as tf
import zipfile
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib
matplotlib.use("Qt5Agg")

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Download videos and unzip
# url = "https://lazyprogrammer.me/course_files/cnn_class2_videos.zip"
# filename = url.split("/")[-1]
# extract_to = "./extracted_files"

# # Download the file if it doesn't already exist
# if not os.path.exists(filename):
#     subprocess.run(["curl", "-O", url])
# else:
#     print("File already exists. Skipping download.")

# # Unzip the file if it exists
# if os.path.exists(filename):
#     # Create a directory to extract to if it doesn't exist
#     if not os.path.exists(extract_to):
#         os.makedirs(extract_to)

#     # Unzip the file
#     with zipfile.ZipFile(filename, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#     print(f"Extracted files to {extract_to}")
# else:
#     print("Zip file does not exist.")

# Model and label map paths
url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz'
INPUT_VIDEOS = ['catdog', 'safari', 'traffic']

PATH_TO_MODEL_DIR = tf.keras.utils.get_file(
    fname='ssd_resnet101_v1_fpn_640x640_coco17_tpu-8',
    origin=url,
    untar=True)

url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'

PATH_TO_LABELS = tf.keras.utils.get_file(
    fname='mscoco_label_map.pbtxt',
    origin=url,
    untar=False)

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS,
    use_display_name=True)

def detect_objects_in_image(image_np):
    """Detect objects in a single image frame."""
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    return image_np_with_detections

def detect_objects_in_video(input_video):
    """Detect objects in each frame of a video."""
    print(f'Running inference for {input_video}.mp4... ', end='')

    video_reader = imageio.get_reader(f'{input_video}.mp4')
    video_writer = imageio.get_writer(f'{input_video}_annotated.mp4', fps=10)

    t0 = time.time()
    n_frames = 0
    for frame in video_reader:
        n_frames += 1
        new_frame = detect_objects_in_image(frame)
        video_writer.append_data(new_frame)

    fps = n_frames / (time.time() - t0)
    print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

    video_writer.close()
    
# Run object detection on the first input video
detect_objects_in_video(INPUT_VIDEOS[0])
# #detect_objects_in_video(INPUT_VIDEOS[1])
# detect_objects_in_video(INPUT_VIDEOS[2])



# def detect_objects_in_video(input_video):
#     """Detect objects in each frame of a video."""
#     print(f'Running inference for {input_video}.mp4... ', end='')

#     video_reader = imageio.get_reader(f'{input_video}.mp4')
#     video_writer = imageio.get_writer(f'{input_video}_annotated.mp4', fps=10)

#     batch_size = 16
#     frame_batch = []
#     t0 = time.time()
#     n_frames = 0
    
#     for frame in video_reader:
#         frame_batch.append(frame)
#         n_frames += 1
        
#         if len(frame_batch) == batch_size:
#             processed_frames = detect_objects_in_images(np.array(frame_batch))
#             for processed_frame in processed_frames:
#                 video_writer.append_data((processed_frame * 255).astype(np.uint8))
#             frame_batch = []
    
#     # Process remaining frames in the last batch
#     if frame_batch:
#         processed_frames = detect_objects_in_images(np.array(frame_batch))
#         for processed_frame in processed_frames:
#             video_writer.append_data((processed_frame * 255).astype(np.uint8))

#     fps = n_frames / (time.time() - t0)
#     print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

#     video_writer.close()
    


