import cv2
import mediapipe as mp


class Detector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.65, trackCon=0.65):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.tipIds = [4, 8, 12, 16, 20]

    def findIndex(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        self.landmarkList = []

        if self.results.multi_hand_landmarks:
            for id, landmark in enumerate(self.results.multi_hand_landmarks[0].landmark):
                height, width, _ = imgRGB.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                self.landmarkList.append([id, x, y])

        return self.landmarkList

    def fingerIsUp(self):
        self.fingersUp = []

        if(self.landmarkList):
            if(self.landmarkList[self.tipIds[0] - 1][1] > self.landmarkList[self.tipIds[0]][1]):
                self.fingersUp.append(1)
            else:
                self.fingersUp.append(0)

            for id in range(1, 5):
                if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:
                    self.fingersUp.append(1)
                else:
                    self.fingersUp.append(0)
        return self.fingersUp

    def findCenter(self):
        pass
    
