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

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import qwiic_button 

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

SHOULDER_RANGE = [100, 170]
LFLEXOR_XZ_RANGE = [0.8, 4.0]
LFLEXOR_XY_RANGE = [-15, 15]
HEAD_RANGE = [0, 200]

def pl_landmark_to_angle(pl_landmark):
    return pl_landmark


q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))
    
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

            #calc shoulder
            lands = pl[0]
            nose_to_lshoulder = [lands[0], lands[11]]
            rshoulder_to_wrist = [lands[12], lands[16]]#lands[14]]

            lshoulder_to_wrist = [lands[11], lands[15]]#lands[13]]

            chest = [lands[12], lands[11]]
            ears = [lands[7], lands[8]]
            
            rshangle = calculate_angle(nose_to_lshoulder, lshoulder_to_wrist, ['y', 'z'])
            lshangle = calculate_angle(nose_to_lshoulder, rshoulder_to_wrist, ['y', 'z'])


            lflexor_xz_angle = (lands[16].x*width - lands[12].x*width)/(lands[8].x*width - lands[7].x*width)
            lflexor_xy_angle = (lands[16].y*height - lands[12].y*height)/(lands[0].y*height - lands[10].y*height)    

            rflexor_xz_angle = (lands[15].x*width - lands[11].x*width)/(lands[7].x*width - lands[8].x*width)
            rflexor_xy_angle = (lands[15].y*height - lands[11].y*height)/(lands[0].y*height - lands[9].y*height)    

            lflangle_xz = trig_angle_to_servo_angle(lflexor_xz_angle, 'lflexor_xz')
            lflangle_xy = trig_angle_to_servo_angle(lflexor_xy_angle, 'lflexor_xy')

            rflangle_xz = trig_angle_to_servo_angle(rflexor_xz_angle, 'lflexor_xz')
            rflangle_xy = trig_angle_to_servo_angle(rflexor_xy_angle, 'lflexor_xy')

            head = (lands[0].x*width - lands[12].x*width)
            hangle = round(trig_angle_to_servo_angle(head, 'head'))
            hangle = 180 - max(0, min(180, hangle))

            lwrist = lands[16]
            rwrist = lands[15]
            lindex = lands[20]
            rindex = lands[19]
            
            #wrist-to-index against wrist-to-elbow
            
            #Lhand = 
            #rhand = 
            
            
            """
            print('#'*50)
            #print('LFANGLE IS:', lflangle)
            print("lflexor_xy is:", lflexor_xy_angle)
            print("lflexor_xz is:", lflexor_xz_angle)
            print("lflangle_xz is:", lflangle_xz)
            print("lflangle_xy is:", lflangle_xy)
            
            print('-'*50)
            
            print("rflexor_xy is:", rflexor_xy_angle)
            print("rflexor_xz is:", rflexor_xz_angle)
            print("rflangle_xz is:", rflangle_xz)
            print("rflangle_xy is:", rflangle_xy)
            """
            
            
            if lflangle_xz < 0:
                lflangle_xz = 0
            elif lflangle_xz > 180:
                lflangle_xz = 180
            if lflangle_xy < 0:
                lflangle_xy = 0
            elif lflangle_xy > 180:
                lflangle_xy = 180

                
            if rflangle_xz < 0:
                rflangle_xz = 0
            elif rflangle_xz > 180:
                rflangle_xz = 180
            if rflangle_xy < 0:
                rflangle_xy = 0
            elif rflangle_xy > 180:
                rflangle_xy = 180
            
                
            lflangle = round(max(0, min(((90 - lshangle)/90 * lflangle_xy + lshangle/90 * lflangle_xz), 180))/2)
            rflangle = round(max(0, min(((90 - rshangle)/90 * rflangle_xy + rshangle/90 * rflangle_xz), 180))/2)
            
            #print('lflangle is:', lflangle)
            #print('rflangle is:', rflangle)
            
            lshangle = round(trig_angle_to_servo_angle(lshangle, 'left_shoulder'))
            rshangle = round(trig_angle_to_servo_angle(rshangle, 'right_shoulder'))
            
            if lshangle < 0:
                lshangle = 0
            elif lshangle > 180:
                lshangle = 180
            if rshangle < 0:
                rshangle = 0
            elif rshangle > 180:
                rshangle = 180
                
            #print('ANGLE IS:', rshangle)
            #print('#'*50)

            
            msg = ''
            msg += 'left_shoulder' + INTERNAL_DELIMITER + str(lshangle) + LINE_DELIMITER + 'right_shoulder' + INTERNAL_DELIMITER + str(rshangle) + LINE_DELIMITER + 'left_elbow' + INTERNAL_DELIMITER + str(lflangle) + LINE_DELIMITER + 'right_elbow' + INTERNAL_DELIMITER + str(rflangle) + LINE_DELIMITER + 'head' + INTERNAL_DELIMITER + str(hangle) + LINE_DELIMITER
            
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

    my_button = qwiic_button.QwiicButton()
    brightness = 100

    if my_button.begin() == False:
        print("\nThe Qwiic Button isn't connected to the system. Please check your connection", \
            file=sys.stderr)
        raise Exception("Button not connected")
        
    print("\nButton ready!")
    device_info = sd.query_devices(None, "input")
    # soundfile expects an int, sounddevice provides a float:
    samplerate = int(device_info["default_samplerate"])
    model = Model(lang="en-us")
    send_next = False
    rec = KaldiRecognizer(model, samplerate)
    ticks = 0
    
    
    while True:
        print('new input stream')
        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=None,
            dtype="int16", channels=1, callback=callback):
            
            # Continuously capture images from the camera and run inference
            while cap.isOpened():
                if (my_button.is_button_pressed()) and not(send_next):
                    send_next = True
                    my_button.LED_on(brightness)

                    print("#" * 80)
                    print("Press Ctrl+C to stop the recording")
                    print("#" * 80)
                    break

                if send_next:
                    sdata = q.get()
                    tmp = rec.AcceptWaveform(sdata)
                    if tmp:
                        res = rec.Result()
                        msg = '&' + res.split(':')[1].split("}")[0] 
                        client.publish(topic, msg)
                        if ticks > 5:
                            send_next = False
                            my_button.LED_off()

                    if not my_button.is_button_pressed():
                        ticks += 1
                    else:
                        ticks = 0

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
                            z_coord = int(landmark.z * 100)

                            # Display coordinates next to the landmark
                            if i == 11: #left shoulder
                                y_shift = -110
                                coord_text = f'({x_coord}, {y_coord}, {z_coord})'
                                coord_location = (x_coord, y_coord + y_shift)
                                cv2.putText(current_frame, coord_text, coord_location,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, text_color2, font_thickness, cv2.LINE_AA)

                            elif i == 12: #right shoulder
                                y_shift = -110
                                coord_text = f'({x_coord}, {y_coord}, {z_coord})'

                                coord_location = (x_coord, y_coord + y_shift)
                                cv2.putText(current_frame, coord_text, coord_location,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, text_color2, font_thickness, cv2.LINE_AA)

                            elif i == 13: #left elbow
                                y_shift = -110
                                coord_text = f'({x_coord}, {y_coord}, {z_coord})'
                                coord_location = (x_coord, y_coord + y_shift)
                                cv2.putText(current_frame, coord_text, coord_location,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, text_color2, font_thickness, cv2.LINE_AA)

                            elif i == 14: #right elbow
                                y_shift = -110
                                coord_text = f'({x_coord}, {y_coord}, {z_coord})'
                                coord_location = (x_coord, y_coord + y_shift)
                                cv2.putText(current_frame, coord_text, coord_location,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, text_color2, font_thickness, cv2.LINE_AA)

                            elif i == 19: #left index finger
                                y_shift = -110
                                coord_text = f'({x_coord}, {y_coord}, {z_coord})'
                                coord_location = (x_coord, y_coord + y_shift)
                                cv2.putText(current_frame, coord_text, coord_location,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, text_color2, font_thickness, cv2.LINE_AA)

                            elif i == 20: #right index finger
                                y_shift = -110
                                coord_text = f'({x_coord}, {y_coord}, {z_coord})'
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

    elif dims == ['x', 'z']:
        a1, b1 = landmarks1[0].x, landmarks1[0].z
        a2, b2 = landmarks1[1].x, landmarks1[1].z

        a3, b3 = landmarks2[0].x, landmarks2[0].z
        a4, b4 = landmarks2[1].x, landmarks2[1].z
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

def trig_angle_to_servo_angle(trig_angle, servo_name):
    oldmin = 1
    oldmax = 2
    if 'shoulder' in servo_name:
        oldmin = SHOULDER_RANGE[0]
        oldmax = SHOULDER_RANGE[1]
    elif 'lflexor_xz' in servo_name:
        oldmin = LFLEXOR_XZ_RANGE[0]
        oldmax = LFLEXOR_XZ_RANGE[1]
    elif 'lflexor_xy' in servo_name:
        oldmin = LFLEXOR_XY_RANGE[0]
        oldmax = LFLEXOR_XY_RANGE[1]
    elif 'head' in servo_name:
        oldmin = HEAD_RANGE[0]
        oldmax = HEAD_RANGE[1]
        
    servo_angle = ((trig_angle - oldmin) / (oldmax - oldmin)) * (180 - 0) + 0
    return servo_angle

    
if __name__ == '__main__':
    main()

    
