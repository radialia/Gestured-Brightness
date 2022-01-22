# Importing modules
import cv2
import math
import numpy as np
import screen_brightness_control as sbc
import HandDetector

# Initialise camera object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the width and height of the opencv window
cap.set(3, 800)
cap.set(4, 500)

# Initialise hand detection
detector = HandDetector.Detector()


while cap.isOpened():
    # Reads the camera
    success, video = cap.read()
    if not success:
        print("Ignoring empty camera frame....")
        continue

    # Flip the screen to give a selfie view
    video = cv2.flip(video, 1)

    # Show brightness text on the screen
    cv2.putText(video, f"Brightness: {int(sbc.get_brightness())}%",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (228, 0, 63), 2)

    # Show instruction to quit the window
    cv2.putText(video, f"Press 'q' to exit",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 0, 18), 2)

    # Finds index of each landmark on the recognised hand
    landmarkList = detector.findIndex(video)
    # Get which fingers are raised up
    fingers = detector.fingerIsUp()

    if(len(landmarkList) != 0):
        # Gets landmarks of thumb and index finger
        thumb = landmarkList[4]
        index = landmarkList[8]

        # If the thumb and index fingers are raised up
        if(fingers[0] == 1 and fingers[1] == 1):

            if(thumb and index):
                # Get the x and y coordinates from the landmarks
                thumb_x, thumb_y = thumb[1], thumb[2]
                index_x, index_y = index[1], index[2]

                # Get the center point between the thumb and index finger
                center_x, center_y = (thumb_x+index_x)//2, (thumb_y+index_y)//2

                # Draw circles on the indexes and a line in between them
                cv2.circle(video, (thumb_x, thumb_y),
                           10, (20, 63, 255), cv2.FILLED)
                cv2.circle(video, (index_x, index_y),
                           10, (20, 63, 255), cv2.FILLED)
                cv2.line(video, (thumb_x, thumb_y),
                         (index_x, index_y), (20, 241, 255), 3)
                # Draw a circle in the middle
                cv2.circle(video, (center_x, center_y),
                           5, (229, 45, 0), cv2.FILLED)

                # Get the distance between the index finger and thumb using distance formula
                distance = math.sqrt(
                    math.pow((thumb_x-index_x), 2) + math.pow((thumb_y-index_y), 2))
                # Convert the min andd amx limits of distance between the thumb and index finger into a range of 0 and 100
                brightness = np.interp(distance, [10, 290], [0, 100])

                # Set the brightness accordingly
                sbc.set_brightness(brightness, force=True)

    # Show the window
    cv2.imshow('Gestured Brightness', video)
    # Press 'q' to quit window
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
