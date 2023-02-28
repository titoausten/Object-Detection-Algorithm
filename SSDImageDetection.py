"""
COMPUTER VISION - SSD ALGORITHM FOR OBJECT DETECTION 
Name: Tito Osadebey
LinkedIn: https://www.linkedin.com/in/tito-osadebe
Email: osadebe.tito@gmail.com

"""
import cv2 as cv
import numpy as np

# Not considering the Probability of objects less than 30%
THRESHOLD = 0.3

# Reducing the number of bounding boxes,
# the lower suppression threshold = minimum number of bounding boxes
SUPPRESSION_THRESHOLD = 0.3

# Required image size to be passed into the model
IMAGE_INPUT_SIZE = 320

# Required files for detection:
# the configuration file and the pre-trained model weights (for SSD in this case)
CONFIG_FILE = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WEIGHTS = 'frozen_inference_graph.pb'


def get_class_names(file_name='class_names.txt'):
    with open(file_name, 'rt') as file:
        names = file.read().rstrip('\n').split('\n')
    return names

def display_detected_objects(image, boxes_to_keep, all_bounding_boxes, object_names, class_ids):
    for index in boxes_to_keep:
        bounding_box = all_bounding_boxes[index]
        x,y,w,h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        cv.rectangle(image, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)
        class_with_confidence = object_names[class_ids[index]-1].upper() + ': ' + str(int(confidence_values[index] * 100)) + '%'
        cv.putText(image, class_with_confidence, (x, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)

def rescale_image(image, scale):
    '''
    The rescale_image() function is used to resize a given image frame to a specific scale.
    The function takes in two arguments.

    Parameters:
    - frame: the image frame that needs to be resized. The frame should be a numpy array with dimensions
    [height, width, channels].
    - scale: the scale factor that the image should be resized to. This should be a float value,
    where the value of 1 keeps the original size of the image and values less than or greater than 1
    reduces or increases the size of the image respectively.

    Returns:
    - the resized image as a numpy array with the same dimensions as the input image frame,
    but with the new width and height values as specified by the scale factor.

    Example:

    import cv2 as cv
    import numpy as np

    #Load an image frame as a numpy array
    frame = cv.imread("image.jpg")

    #Resize the image to half of its original size
    resized_frame = rescale_image(frame, 0.5)

    #Display the resized image
    cv.imshow('Resized Image', resized_image)
    cv.waitKey(0)
    '''
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

class_names = get_class_names()
'''
# DETECTING OBJECTS IN IMAGES
filename = 'IMG_0409.jpg'
image = rescale_image(cv.imread(filename), 0.25)

# Instantiating a Neural Network
neural_network = cv.dnn_DetectionModel(WEIGHTS, CONFIG_FILE)
# USING A CPU
neural_network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
neural_network.setInputSize(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)
neural_network.setInputScale(1.0/127.5)
neural_network.setInputMean((127.5, 127.5, 127.5))
neural_network.setInputSwapRB(True)

class_label_ids, confidence_values, bbox = neural_network.detect(image)
bbox = list(bbox)
confidence_values = list(np.array(confidence_values).reshape(1, -1))[0]

box_to_keep = cv.dnn.NMSBoxes(bbox, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
display_detected_objects(image, box_to_keep, bbox, class_names, class_label_ids)

cv.imshow('SSD Algorithm', image)
cv.waitKey()
'''


# DETECTING OBJECTS IN VIDEOS (REALTIME, filename=0)
#filename = 'Glitch B Dash & Konkrete- Clowns.mp4'
filename = 0
video = cv.VideoCapture(filename)

# Instantiating a Neural Network
neural_network = cv.dnn_DetectionModel(WEIGHTS, CONFIG_FILE)
# USING A CPU
neural_network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
neural_network.setInputSize(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)
neural_network.setInputScale(1.0/127.5)
neural_network.setInputMean((127.5, 127.5, 127.5))
neural_network.setInputSwapRB(True)

while True:
    frame_grabbed, frame = video.read()

    if not frame_grabbed:
        break
    
    class_label_ids, confidence_values, bbox = neural_network.detect(frame)

    bbox = list(bbox)
    confidence_values = list(np.array(confidence_values).reshape(1, -1))[0]

    box_to_keep = cv.dnn.NMSBoxes(bbox, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
    display_detected_objects(frame, box_to_keep, bbox, class_names, class_label_ids)

    cv.imshow('SSD Algorithm', frame)
    cv.waitKey(500)

video.release()
cv.destroyAllWindows()