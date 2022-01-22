# Importing required modules
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
import time
import math
import numpy as np
import screen_brightness_control as sbc
import HandDetector

# Initialise the camera object
vs = WebcamVideoStream(src=0).start()
# Starting the fps
fps = FPS().start()
pTime = 0

# Initialise hand detection
detector = HandDetector.Detector()

while True:
    # Read the frame
    frame = vs.read()
    # Flip the frame to give a selfie view
    frame = cv2.flip(frame, 1)
    # Resize the frame's width to 600px
    frame = imutils.resize(frame, width=600)

    # Get the height, width and channel from the frame
    height, width, channel = frame.shape

    # Calculating the fps
    cTime = time.time()
    _fps = 1 / (cTime - pTime)
    pTime = cTime

    # Putting the FPS text on the frame
    cv2.putText(frame, f"FPS: {int(_fps)}", (width-150, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Show brightness text on the screen
    cv2.putText(frame, f"Brightness: {int(sbc.get_brightness())}%",
                (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (228, 0, 63), 2)

    # Show instruction to quit the window
    cv2.putText(frame, f"Press 'q' to exit",
                (40, 80), cv2.FONT_HERSHEY_PLAIN, 2, (10, 0, 18), 2)

    # Finds index of each landmark on the recognised hand
    landmarkList = detector.findIndex(frame)
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
                cv2.circle(frame, (thumb_x, thumb_y),
                           10, (20, 63, 255), cv2.FILLED)
                cv2.circle(frame, (index_x, index_y),
                           10, (20, 63, 255), cv2.FILLED)
                cv2.line(frame, (thumb_x, thumb_y),
                         (index_x, index_y), (20, 241, 255), 3)
                # Draw a circle in the middle
                cv2.circle(frame, (center_x, center_y),
                           5, (229, 45, 0), cv2.FILLED)

                # Get the distance between the index finger and thumb using distance formula
                distance = math.sqrt(
                    math.pow((thumb_x-index_x), 2) + math.pow((thumb_y-index_y), 2))
                # Convert the min andd amx limits of distance between the thumb and index finger into a range of 0 and 100
                brightness = np.interp(distance, [10, 290], [0, 100])

                # Set the brightness accordingly
                sbc.set_brightness(brightness, force=True)

    # Show the frame in a window
    cv2.imshow("Gestured Brightness Control", frame)
    # Updates the frame simulataneously
    fps.update()

    # If the 'q' key is pressed then exit the window
    if(cv2.waitKey(5) & 0xFF == ord('q')):
        break

# Stop the fps
fps.stop()

# destroy all the windows
cv2.destroyAllWindows()
# Stop recording
vs.stop()
