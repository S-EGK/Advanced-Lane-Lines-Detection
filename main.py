# Import Libraries
import cv2
import glob
import numpy as np
import matplotlib.image as mpimg

# Import packages
from LaneLines import *
from Thresholding import *
from PerspectiveTransformation import *
from CameraCalibration import CameraCalibration

class FindLaneLines:
    """ This class is for parameter tunning.
    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, img):
        out_img = self.forward(img)
        return out_img

findLaneLines = FindLaneLines()

cap = cv2.VideoCapture("curved_lanes.mp4")
while(cap.isOpened()):
    _, frame = cap.read()

    # Convert the frame to the RGB color format used by mpimg.imread()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    out_img = findLaneLines.process_image(frame_rgb)

    cv2.imshow("result", out_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()