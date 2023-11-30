import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time

import board
import ssl

import paho.mqtt.client as mqtt
import uuid
import queue



model_path = '/home/ben/Interactive-Lab-Hub/final/pose_landmarker_lite.task'

global voter
voter = 0

global iters
iters = 0

class HumanPoseDetection:
    def __init__(self):
        # TODO: change the path
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.result = mp.tasks.vision.PoseLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode       

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.callback
        )

        self.client = mqtt.Client(str(uuid.uuid1()))
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.username_pw_set('idd', 'device@theFarm')
        self.client.connect(
            'farlab.infosci.cornell.edu',
            port=8883)
        self.topic = 'IDD/cool_table/robit'
        self.init_positions = []
        self.threshold = 0.4
    def callback(self, result, output_image, timestamp_ms):
        global iters
        global voter

        pl = result.pose_landmarks
        print("-"*100)
        #print("pose landmarks:", pl)
        if len(pl) == 0:
            print('no landmarks!')
            self.client.publish(self.topic, 'no_land')
        else:
            print("len:", len(pl[0]))
            if self.init_positions == []:
                print('initing')
                self.init_positions = pl[0]
            else:
                left_change = np.abs(pl[0][13].y - self.init_positions[13].y)
                right_change = np.abs(pl[0][14].y - self.init_positions[14].y)
                print('left_change:', left_change)
                print('right_change:', right_change)

                """
                x_left_change = np.abs(pl[0][11].x - self.init_positions[11].x)
                x_right_change = np.abs(pl[0][12].x - self.init_positions[12].x)
                print('x_left_change:', x_left_change)
                print('x_right_change:', x_right_change)
                """
                if iters > 10:
                    voter = 0
                    iters = 0
                if (left_change < self.threshold) and (right_change < self.threshold):
                    print('threshold not met')

                elif left_change > right_change:
                    print('left!')
                    if voter <7:
                        voter += 1
                    print('voter is:', voter)
                else:
                    print('right!')
                    if voter > -7:
                        voter -= 1
                    print('voter is:', voter)
                if voter >= 5:
                    self.client.publish(self.topic, 'left')
                    print('*'*10)
                    print('sent left')
                elif voter <= -5:
                    print('*'*10)
                    print('sent right')
                    self.client.publish(self.topic, 'right')
                else:
                    print('voter is:', voter)
                    
                print('iters is:', iters)
                iters += 1
                
                #self.init_positions = pl[0]
                
        return

        
    def detect_pose(self):
        print("detecting pose")
        cap = cv2.VideoCapture('/dev/video0')
        print('camera opened')
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                _, image = cap.read()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, frame_timestamp_ms)
                time.sleep(0.05)
def main():
    HPD_ = HumanPoseDetection()
    HPD_.detect_pose()
    return


if __name__=="__main__":
    main()


