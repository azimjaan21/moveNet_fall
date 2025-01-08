import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load MoveNet model
model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
movenet = model.signatures["serving_default"]

POSE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head and shoulders
    (0, 5), (5, 6), (6, 7),          # Left arm
    (0, 8), (8, 9), (9, 10),         # Right arm
    (11, 12), (12, 13),              # Left leg
    (11, 14), (14, 15)               # Right leg
]

def preprocess_frame(frame):
    """Preprocess input frame for MoveNet."""
    input_image = tf.image.resize_with_pad(frame, 256, 256)  # Resize and pad to 256x256
    input_image = tf.cast(input_image, dtype=tf.int32)  # Ensure integer type
    return tf.expand_dims(input_image, axis=0)

def visualize_pose(frame, keypoints):
    """Draw keypoints and skeleton on the frame."""
    for idx, kp in enumerate(keypoints):
        x, y, confidence = int(kp[1] * frame.shape[1]), int(kp[0] * frame.shape[0]), kp[2]
        if confidence > 0.3:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for pair in POSE_PAIRS:
        part_a, part_b = pair
        if keypoints[part_a][2] > 0.3 and keypoints[part_b][2] > 0.3:
            x1, y1 = int(keypoints[part_a][1] * frame.shape[1]), int(keypoints[part_a][0] * frame.shape[0])
            x2, y2 = int(keypoints[part_b][1] * frame.shape[1]), int(keypoints[part_b][0] * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
