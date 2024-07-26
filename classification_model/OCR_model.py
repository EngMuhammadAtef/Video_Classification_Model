# import libraries
import cv2
from pytesseract import image_to_string
from PIL import Image
import numpy as np

# config settings
IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256
LANG = 'eng+ara'

# Define a function to preprocess the image
def preprocess_image(image_path, target_size):
    # Load the image with OpenCV
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to the target size
    resized_img = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img

# Define a function to extract the text
def image2text(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    image = preprocess_image(image_path, target_size)
    string = image_to_string(image, lang=LANG)
    return ''.join(c for c in string if c.isdigit() or c.isalpha() or c in [' ', '\n']).replace('\n', ' ')

# Define a function to preprocess a frame
def preprocess_frame(frame, target_size):
    # Convert the frame to uint8
    frame_uint8 = (frame * 255).astype(np.uint8)
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
    # Resize the frame to the target size
    resized_frame = cv2.resize(gray_frame, target_size, interpolation=cv2.INTER_AREA)
    return resized_frame

# Define a function to extract text from an image frame
def frame_to_text(frame):
    pil_image = Image.fromarray(frame)
    string = image_to_string(pil_image)
    return ''.join(c for c in string if c.isdigit() or c.isalpha() or c in [' ', '\n']).replace('\n', ' ')

# Define a function to extract the text from video
def video2text(frames, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    texts = set()
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame, target_size)
        text = frame_to_text(preprocessed_frame)
        texts.add(text)
    return ' '.join(texts)
