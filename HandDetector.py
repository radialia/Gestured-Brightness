import cv2
import mediapipe as mp


class Detector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findIndex(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        self.landmarkList = {}

        if self.results.multi_hand_landmarks:
            for id, landmark in enumerate(self.results.multi_hand_landmarks[0].landmark):
                height, width, channel = imgRGB.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                self.landmarkList.update({id: [x, y]})

        return self.landmarkList
