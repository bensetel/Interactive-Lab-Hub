import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time

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
        
    def callback(self, result, output_image, timestamp_ms):
        global DETECTING
        DETECTING = True
        print("-"*100)
        print("callback called!")
        print("type result:", type(result))
        print("type output_image:",type(output_image))
        print("pose landmarks:", result.pose_landmarks)

        #annotated_image = draw_landmarks_on_image(output_image, result)
        # cv2.imshow('blick', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

        # cv2.imshow('blick', output_image.numpy_view())
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
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


                
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    print('ano type:', type(annotated_image))
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            rgb_image.numpy_view(),
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def main():
    HPD_ = HumanPoseDetection()
    HPD_.detect_pose()
    return


if __name__=="__main__":
    main()



    """


    
def main():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()


wm = 'myimage'

cv2.imshow(wm, frame)
cv2.waitKey(0) 
cv2.destroyAllWindows()


        
img = frame

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
frame_timestamp_ms = int(time.time() * 1000)
result = landmarker.detect_async(mp_image, frame_timestamp_ms)
        #detector = vision.PoseLandmarker.create_from_options(options)
        

annotated_image = draw_landmarks_on_image(img, result)
cv2.imshow(window_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0) 
cv2.destroyAllWindows()


segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow(window_name, visualized_mask)
cv2.waitKey(0) 
cv2.destroyAllWindows()

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow(window_name, visualized_mask)
cv2.waitKey(0) 
cv2.destroyAllWindows()
        
return


"""






    """
def main():
    img = cv2.imread("image.jpg")
    window_name = 'myimage'

    cv2.imshow(window_name, img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("image.jpg")

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(img, detection_result)
    cv2.imshow(window_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imshow(window_name, visualized_mask)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imshow(window_name, visualized_mask)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return

"""

    """

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # ...
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
"""


    """
    
cv2.imshow(blick, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

time.sleep(4)
"""
