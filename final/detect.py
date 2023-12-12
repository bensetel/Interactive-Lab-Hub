# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run pose landmarker."""

#USE HANDS TO DETECT SHOULDER ROTATION

import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import board
import ssl

import paho.mqtt.client as mqtt
import uuid
import queue

import os
import glob
import pandas as pd
import math


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

client = mqtt.Client(str(uuid.uuid1()))
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.username_pw_set('idd', 'device@theFarm')
client.connect(
    'farlab.infosci.cornell.edu',
    port=8883)

init_positions = []

iters = 0

#USE HANDS TO DETECT SHOULDER ROTATION
pose_dict = {
    # 'nose':0,
    # 'left_eye_(inner)':1,
    # 'left_eye':2,
    # 'left_eye_(outer)':3,
    # 'right_eye_(inner)':4,
    # 'right_eye':5,
    # 'right_eye_(outer)':6,
    # 'left_ear':7,
    # 'right_ear':8,
    # 'mouth_(left)':9,
    # 'mouth_(right)':10,
    11:'left_shoulder',
    12:'right_shoulder',
    13:'left_elbow',
    14:'right_elbow',
    15:'left_wrist',
    16:'right_wrist',
    # 'left_pinky':17,
    # 'right_pinky':18,
    19:'left_index',
    20:'right_index',
    # 'left_thumb':21,
    # 'right_thumb':22,
    # 'left_hip':23,
    # 'right_hip':24,
    # 'left_knee':25,
    # 'right_knee':26,
    # 'left_ankle':27,
    # 'right_ankle':28,
    # 'left_heel':29,
    # 'right_heel':30,
    # 'left_foot_index':31,
    # 'right_foot_index':32
}
reverse_pose_dict = {}
for elem in [{y:x} for x,y in zip(list(pose_dict.keys()), list(pose_dict.values()))]:
    reverse_pose_dict.update(elem)

topic = 'IDD/cool_table/robit'
pd_len = len(list(pose_dict.keys()))
voter = [0] * pd_len
threshold = [0.1] * pd_len
INTERNAL_DELIMITER = '#'
LINE_DELIMITER = '*'

def pl_landmark_to_angle(pl_landmark):
    return pl_landmark
    
def run(model: str, num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float, min_tracking_confidence: float,
        output_segmentation_masks: bool,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the pose landmarker model bundle.
      num_poses: Max number of poses that can be detected by the landmarker.
      min_pose_detection_confidence: The minimum confidence score for pose
        detection to be considered successful.
s      min_pose_presence_confidence: The minimum confidence score of pose
        presence score in the pose landmark detection.
      min_tracking_confidence: The minimum confidence score for the pose
        tracking to be considered successful.
      output_segmentation_masks: Choose whether to visualize the segmentation
        mask or not.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    text_color2 = (255, 255, 255)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    overlay_alpha = 0.5
    mask_color = (100, 100, 0)  # cyan

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT, iters, voter, client, topic, init_positions, threshold, iters, pd_len, INTERNAL_DELIMITER, LINE_DELIMITER

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1
        pl = result.pose_landmarks
        #print('pl:', pl)
        print('COUNTER IS:', COUNTER)

        
        msg = ''
        print("-"*100)
        if len(pl) == 0:
            print('no landmarks!')
            msg = 'no_land'
        else:            
            if init_positions == []:
                print('initing')
                init_positions = pl[0]
                msg = 'init'
            else:
                msg = generate_message(pl[0])
            #calc shoulder
            lands = pl[0]
            lm1 = [lands[0], lands[11]]
            lm2 = [lands[12], lands[14]]
            angle = calculate_angle(lm1, lm2, ['y', 'z'])
            print('#'*50)
            print('ANGLE IS:', angle)
            print('#'*50)
                        
        client.publish(topic, msg)
        

        

        
    # Initialize the pose landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=save_result)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run pose landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            # Draw landmarks.
            for pose_landmarks in DETECTION_RESULT.pose_landmarks:
                
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in pose_landmarks
                ])

                # Draw landmarks and coordinates
                for i, landmark in enumerate(pose_landmarks_proto.landmark):
                    x_coord = int(landmark.x * width)
                    y_coord = int(landmark.y * height)
                    z_coord = landmark.z

                    # Display coordinates next to the landmark
                    if i == 11: #left shoulder
                        y_shift = -110
                        coord_text = f'({x_coord}, {y_coord})'
                        coord_location = (x_coord, y_coord + y_shift)
                        cv2.putText(current_frame, coord_text, coord_location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_size, text_color2, font_thickness, cv2.LINE_AA)
                   
                    elif i == 12: #right shoulder
                        y_shift = -110
                        coord_text = f'({x_coord}, {y_coord})'
                        
                        coord_location = (x_coord, y_coord + y_shift)
                        cv2.putText(current_frame, coord_text, coord_location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_size, text_color2, font_thickness, cv2.LINE_AA)

                    elif i == 13: #left elbow
                        y_shift = -110
                        coord_text = f'({x_coord}, {y_coord})'
                        coord_location = (x_coord, y_coord + y_shift)
                        cv2.putText(current_frame, coord_text, coord_location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_size, text_color2, font_thickness, cv2.LINE_AA)
                    
                    elif i == 14: #right elbow
                        y_shift = -110
                        coord_text = f'({x_coord}, {y_coord})'
                        coord_location = (x_coord, y_coord + y_shift)
                        cv2.putText(current_frame, coord_text, coord_location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_size, text_color2, font_thickness, cv2.LINE_AA)
                    
                    elif i == 19: #left index finger
                        y_shift = -110
                        coord_text = f'({x_coord}, {y_coord})'
                        coord_location = (x_coord, y_coord + y_shift)
                        cv2.putText(current_frame, coord_text, coord_location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_size, text_color2, font_thickness, cv2.LINE_AA)
                        
                    elif i == 20: #right index finger
                        y_shift = -110
                        coord_text = f'({x_coord}, {y_coord})'
                        coord_location = (x_coord, y_coord + y_shift)
                        cv2.putText(current_frame, coord_text, coord_location,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_size, text_color2, font_thickness, cv2.LINE_AA)
 
                mp_drawing.draw_landmarks(
                    current_frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())

        if (output_segmentation_masks and DETECTION_RESULT):
            if DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_image = np.zeros(image.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                visualized_mask = np.where(condition, mask_image, current_frame)
                current_frame = cv2.addWeighted(current_frame, overlay_alpha,
                                                visualized_mask, overlay_alpha,
                                                0)

        cv2.imshow('pose_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        
        time.sleep(0.1)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    
    
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker.task')
    parser.add_argument(
        '--numPoses',
        help='Max number of poses that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minPoseDetectionConfidence',
        help='The minimum confidence score for pose detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minPosePresenceConfidence',
        help='The minimum confidence score of pose presence score in the pose '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the pose tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--outputSegmentationMasks',
        help='Set this if you would also like to visualize the segmentation '
             'mask.',
        required=False,
        action='store_true')
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    args = parser.parse_args()

    run(args.model, int(args.numPoses), args.minPoseDetectionConfidence,
        args.minPosePresenceConfidence, args.minTrackingConfidence,
        args.outputSegmentationMasks,
        int(args.cameraId), args.frameWidth, args.frameHeight)

#############
# MATH SECTION
###############
def generate_message(landmarks):
    
    return ''

    # left_shoulder_lm = landmarks[reverse_pose_dict['left_shoulder']]
    # right_shoulder_lm = landmarks[reverse_pose_dict['right_shoulder']]
    # left_elbow_lm = pl[0][reverse_pose_dict['left_elbow']]
    # right_elbow_lm = pl[0][reverse_pose_dict['right_elbow']]
    
    # #shoulder rotation should be based on y distance of elbow from shoulder
    # #next servo out should be x distance from elbow to shoulder
    
    # left_shoulder_to_rotate = np.abs(left_shoulder_lm.y - left_elbow_lm.y)
    
    # right_shoulder_to_rotate = np.abs(right_shoulder_lm.y - right_elbow_lm.y)
    # left_elbow_to_rotate = np.abs(left_shoulder_lm.x - left_elbow_lm.x)
    # right_elbow_to_rotate = np.abs(right_shoulder_lm.x - right_elbow_lm.x)
    
    # ls_angle = pos_to_angle(left_shoulder_to_rotate, 'y', height, 'left_shoulder')
    # rs_angle = pos_to_angle(right_shoulder_to_rotate, 'y', height, 'right_shoulder')
    # le_angle = pos_to_angle(left_elbow_to_rotate, 'x', width, 'left_shoulder')
    # re_angle = pos_to_angle(right_elbow_to_rotate, 'x', width, 'right_shoulder')
    
    # msg += 'left_shoulder' + INTERNAL_DELIMITER + str(ls_angle) + LINE_DELIMITER + 'right_shoulder' + INTERNAL_DELIMITER + str(rs_angle) + LINE_DELIMITER + 'left_elbow' + INTERNAL_DELIMITER + str(le_angle) + LINE_DELIMITER + 'right_elbow' + INTERNAL_DELIMITER + str(re_angle) + LINE_DELIMITER
 

def pos_to_angle(cur_pos, dimension, coeff, name):
    rewrite = False
    df = pd.read_csv(f'{name}.dat', index_col='names')
    oldmax = df.loc[dimension]['max'] * coeff
    oldmin = df.loc[dimension]['min'] * coeff
    
    if cur_pos > oldmax:
        angle = 180
        df.loc[dimension]['max'] = cur_pos
        rewrite = True
        
    elif cur_pos < oldmin:
        angle = 0
        df.loc[dimension]['min'] = cur_pos
        rewrite = True
        
    else:
        angle = (((cur_pos * coeff) - oldmin) / (oldmax - oldmin)) * (180 - 0) + 0
        
    if rewrite:
        df.to_csv(f'{name}.dat')
    
    return round(angle)


#def calculate_angle(landmark1, landmark2, landmark3): #
def calculate_angle(landmarks1, landmarks2, dims):
    if dims == ['x', 'y']:
        a1, b1 = landmarks1[0].x, landmarks1[0].y
        a2, b2 = landmarks1[1].x, landmarks1[1].y

        a3, b3 = landmarks2[0].x, landmarks2[0].y
        a4, b4 = landmarks2[1].x, landmarks2[1].y
        
    elif dims == ['y', 'z']:
        a1, b1 = landmarks1[0].y, landmarks1[0].z
        a2, b2 = landmarks1[1].y, landmarks1[1].z

        a3, b3 = landmarks2[0].y, landmarks2[0].z
        a4, b4 = landmarks2[1].y, landmarks2[1].z
        
    
    
    # Calculate vectors
    vector1 = (a1 - a2, b1 - b2)
    vector2 = (a3 - a4, b3 - b4)

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    # Calculate angle in radians
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))

    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

# Example usage:
# landmark1 = (x1, y1)  # Replace with actual landmark coordinates
# landmark2 = (x2, y2)  # Replace with actual landmark coordinates
# landmark3 = (x3, y3)  # Replace with actual landmark coordinates

# angle = calculate_angle(landmark1, landmark2, landmark3)
# print(f"Angle: {angle} degrees")

    
if __name__ == '__main__':
    main()

    
