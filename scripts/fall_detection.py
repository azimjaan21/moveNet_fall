import cv2
import numpy as np
from pose_estimation import movenet, preprocess_frame, visualize_pose
from helpers import calculate_angle

video_path = 'samples/fall.mp4'
output_video_path = 'output_fall_detection.mp4'  # Path to save the output video

cap = cv2.VideoCapture(video_path)  # Open webcam feed

# Check if the video is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for saving output (e.g., frame width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up the VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Thresholds for fall detection
HEAD_HEIGHT_THRESHOLD = 0.8  # Head height relative to the frame
BODY_ANGLE_THRESHOLD = 70    # Degrees (horizontal body)
prev_keypoints = None  # To track movement across frames

print("Press 'q' to exit...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = preprocess_frame(rgb_frame)
    outputs = movenet(input_image)
    keypoints = outputs["output_0"].numpy()[0, 0, :, :]

    # Fall detection logic
    if len(keypoints) >= 17:
        head = keypoints[0]  # Head keypoint
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        left_hip, right_hip = keypoints[11], keypoints[12]

        # Calculate torso angle and head height
        torso_angle = calculate_angle((left_shoulder[1], left_shoulder[0]), (left_hip[1], left_hip[0]))
        head_height = head[0]

        fall_detected = (
            head_height > HEAD_HEIGHT_THRESHOLD
            and torso_angle > BODY_ANGLE_THRESHOLD
        )

        # Display results
        visualize_pose(frame, keypoints)
        if fall_detected:
            cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Show the video
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
