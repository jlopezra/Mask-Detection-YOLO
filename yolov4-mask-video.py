

"""
Objects Detection on Video with YOLO v4 and OpenCV
File: yolov4-mask-video.py
Author: Jose C. Lopez
"""


# Detecting Objects on Video with OpenCV deep learning library
#
# Algorithm: Yolo v4
# Steps: 1) Read input video 2) Load YOLO v4 Network
# 3) loop through each frame 4) Get reshaped-input-image from the frame
# 5) Implement Forward Pass  6) Get Bounding Boxes
# 7) Non-maximum Suppression 8) Draw Bounding Boxes with Labels
# 9) Write processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
import numpy as np
import cv2
import time
import json


"""
Reading input video
"""
# need opencv version 4.4.0 to run yolov4
print(cv2.__version__)

# Define 'VideoCapture' object and read video from file
vid_name = 'london_part6'
vid_path = 'videos/London_parts/'
video = cv2.VideoCapture(vid_path+vid_name+'.mp4')

# Variable for writer that will be used to write processed frames
writer = None

# Variables for width and height of the frames
h, w = None, None

"""
Loading YOLO v4 network
"""

# Load class labels from file
with open('yolov4_custom_model/classes.names') as f:
    labels = [line.strip() for line in f]

print('List with labels names:')
print(labels)
num_classes = len(labels)

# # Load trained YOLO v4 Objects Detector model with dnn library from OpenCV
# network = cv2.dnn.readNetFromDarknet('yolov4_coco_model/yolov4.cfg',
#                                      'yolov4_coco_model/yolov4.weights')

# Load trained YOLO v4 Objects Detector model with dnn library from OpenCV
network = cv2.dnn.readNetFromDarknet('yolov4_custom_model/yolov4_mask_test.cfg',
                                     'yolov4_custom_model/yolov4_mask_train_best_1024.weights')
# This is defined in the .cfg file
input_image_width = 1024
input_image_height = 832

# list with names of all layers from YOLO v4 network
layers_names_all = network.getLayerNames()
print()
print(layers_names_all)

# Output layers' names that we need in forward pass
layers_names_output = network.getUnconnectedOutLayersNames()
print()
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Set minimum probability to filter out weak predictions
probability_minimum = 0.5

# Setting threshold for filtering out bounding boxes with >
# threshold IOU and not max value with non-maximum suppression
threshold = 0.35

# Generating colors for representing every detected object
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')
colors = [[200, 100, 0], [200, 200, 200], [50, 0, 150]]
print()
print(type(colors))  # <class 'numpy.ndarray'>
# print(colors.shape)  # (80, 3)
print(colors[0])  # [172  10 127]

"""
Read frames in the loop
"""

# Variable for counting frames. Total frames prints at end
f = 0
# Variable for counting total time. Total time prints at end
t = 0
# dictionary for keeping track of number of masks, nomasks, and improper masks
masknum = {}

# Loop through each frames
while True:
    # Capturing frame-by-frame
    ret, frame = video.read()

    # If the frame was not retrieved (at the end of the video)
    # then we break the loop
    if not ret:
        break

    # Getting width and height of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        h, w = frame.shape[:2]

    """
    Get blob from current frame
    """

    # Get blob from current frame
    # The 'cv2.dnn.blobFromImage' function returns 4D blob from current
    # frame after mean subtraction, normalizing, and RB channels swapping.
    # Shape has number of frames, number of channels, width and height
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_image_width, input_image_height),
                                 swapRB=True, crop=False)
    """
    Implement Forward pass
    """

    # Forward pass with our blob and only through output layers
    # Calculate time for forward pass
    network.setInput(blob)  # set blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increment counters for frames and total time
    f += 1
    t += end - start

    # Showing spent time for single current frame
    print(f'Frame {f} took {end - start:.4f} seconds')

    """
    Getting bounding boxes
    """

    # Lists for detected bounding box, confidence, and class number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 class probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Get top left corner coordinates: x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Add results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    Non-maximum suppression
    """

    # non-max suppression of given bounding boxes. This will exclude bounding boxes if their
    # confidences are low or there is another bounding box for this region with higher confidence
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    Drawing bounding boxes and labels
    """
    mask = 0
    nomask = 0
    improper = 0

    # Check if there is at least one detected object after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates: width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Color for current bounding box
            color_box_current = colors[class_numbers[i]] #.tolist()

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width,
                          y_min + box_height),color_box_current, 2)

            # text with label and confidence for current bounding box
            text_box_current = f'{labels[int(class_numbers[i])]}: {confidences[i]:.4f}'

            # Put text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box_current, 2)
            if labels[int(class_numbers[i])] == "Mask":
                mask += 1
            elif labels[int(class_numbers[i])] == "No Mask":
                nomask += 1
            else:
                improper += 1

    # dictionary that keeps track of total masks, nomasks and improper masks at each frame
    masknum[f'{f}'] = [len(results), mask, nomask, improper]
    """
    Writing processed frame into the file
    """

    # Initialize writer
    # we do it only once from the very beginning
    if writer is None:
        # Construct code of the codec.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Write current processed frame into the video file
        writer = cv2.VideoWriter(vid_path + vid_name + f'_mask8_probmin({probability_minimum})thr({threshold}).mp4',
                                 fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)

    # write json file
    with open(vid_path+vid_name+'_json.txt', 'w') as outfile:
        json.dump(masknum, outfile)
"""
End of:
Reading frames in the loop
"""
# Print final results
print()
print('Total number of frames', f)
print(f'Total amount of time {t:.4f} seconds')
print('FPS:', round((f / t), 1))
print('JSON:')
print(json.dumps(masknum))


# Releasing video reader and writer
video.release()
writer.release()

