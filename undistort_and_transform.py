# STEP 3: Apply distortion correction and warping (perspective transformation)

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_cal/cali_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/straight_lines2.jpg')

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (undist.shape[1], undist.shape[0])
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        #Note: you could pick any four of the detected corners 
        # as long as those four corners define a rectangle
        #One especially smart way to do this would be to use four well-chosen
        # corners that were automatically detected during the undistortion steps
        #We recommend using the automatic detection of corners in your code
    src = np.float32(
        [
            [557, 476],
            [731, 476],
            [1042, 674],
            [281, 674]
        ]
    )
            # [588, 459],
            # [696, 459],
            # [975, 637],
            # [327, 637]
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32(
        [
            [280, 0],
            [1060, 0],
            [1060, 700],
            [280, 700]
        ]
    )
            # [300, 0],
            # [1000, 0],
            # [1000, 700],
            # [300, 700]
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src) # This is unnecessaary for this problem
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M

top_down, perspective_M = corners_unwarp(img, mtx, dist)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for displaying normal color
top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)    # for displaying normal color

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=10)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.show()