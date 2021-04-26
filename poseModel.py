import cv2
import time
import mediapipe as mp
import numpy as np
from mss import mss
from PIL import Image


class poseDetector():
    def __init__(
        self,
        mode=False,
        upBody=False,
        smooth=True,
        detectionCo=0.5,
        trackCon=0.5
    ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCo = detectionCo
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            mode,
            upBody,
            smooth,
            detectionCo,
            trackCon
        )

    def findPOse(self, img, img2=[], draw=True):
        results = self.pose.process(img)
        if results.pose_landmarks:
            if draw:
                if img2.shape == img.shape:
                    self.mpDraw.draw_landmarks(
                        img2,
                        results.pose_landmarks,
                        self.mpPose.POSE_CONNECTIONS
                    )
                else:
                    self.mpDraw.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        self.mpPose.POSE_CONNECTIONS
                    )


cap = cv2.VideoCapture('01.mp4')
# cap=cv2.VideoCapture(0)
pTime = 0
blank = np.zeros((360, 640, 3))
detector = poseDetector()

# while True:
#     sucess, img = cap.read()
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     detector.findPOse(img)

#     # print(results.pose_landmarks)
monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
sct = mss()


def main():
    detector = poseDetector()

    # cap = cv2.VideoCapture('01.mp4')
    # cap=cv2.VideoCapture(0)
    pTime = 0
    blank = np.zeros((360, 640, 3))
    empty = np.zeros((10, 10, 3))
    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.resize(img, (640, 360))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        #  img = cap.read()
        detector.findPOse(img, blank)
        detector.findPOse(img, empty)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 55),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('window', img)
        cv2.imshow('blank', blank)
        blank = np.zeros((360, 640, 3))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
