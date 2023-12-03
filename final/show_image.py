import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time


def main():
    img = cv2.imread("image.jpg")
    cv2.imshow('blick', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()

