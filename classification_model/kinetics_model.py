# import libraries
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("..")

# Load the TensorFlow kinetics classification model
vid_class_model = tf.saved_model.load('classification_model/kinetics_classification_model', tags=[])

# Load the labels
with open('classification_model/kinetics_label_map.txt') as obj:
    labels = [line.strip() for line in obj.readlines()]

# Define function to crop center square from frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

# Define function to load video with frame skipping
def load_video(path, max_frames=0, shape=(224, 224), frame_skip=10):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG) # Assuming FFmpeg backend
    frames = []
    frame_count = 0
    
    # Handle error: couldn't open video
    try:
        if not cap.isOpened():
            return np.array(frames)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Skip frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, shape)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            frame_count += 1
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

# Define function to make predictions
def predict(sample_video):
    # Add a batch axis to the sample video.
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    # 'default' is the signature name
    model_sig = vid_class_model.signatures['default']
    logits = model_sig(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)
    video_categories = {}
    
    # print("Top actions:")
    for i in np.argsort(probabilities)[::-1]:
        if int(probabilities[i]*100):
            video_categories[labels[i]] = np.round(probabilities[i].numpy()*100, 2)
            # print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")
    
    return (video_categories)