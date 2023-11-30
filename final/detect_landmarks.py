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
global DETECTING
DETECTING = False

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
        
    def callback(self, result, output_image, timestamp_ms):
        global DETECTING
        DETECTING = True
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
                left_change = np.abs(pl[0][11].y - self.init_positions[11].y)
                right_change = np.abs(pl[0][12].y - self.init_positions[12].y)
                if left_change > right_change:
                    print('left!')
                    self.client.publish(self.topic, 'left')
                else:
                    print('right!')
                    self.client.publish(self.topic, 'right')

        DETECTING = False
        return

        
    def detect_pose(self):
        global DETECTING
        print("detecting pose")
        cap = cv2.VideoCapture('/dev/video0')
        print('camera opened')
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                if DETECTING:
                    print('detecting...')
                    time.sleep(1)
                    continue
                _, image = cap.read()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, frame_timestamp_ms)


def main():
    HPD_ = HumanPoseDetection()
    HPD_.detect_pose()
    return


if __name__=="__main__":
    main()


