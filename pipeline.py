import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def calibrate_camera():
    # * dummy function only need to be run once, check calibrate_camera.py for details
    pass


def clr_grad_thres(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Sobel x
    # ! can also use gray instead of l_channel, but l_channel seems better
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary


def undist_warp(img, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (undist.shape[1], undist.shape[0])

    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        #Note: you could pick any four of the detected corners 
        # as long as those four corners define a rectangle
        #One especially smart way to do this would be to use four well-chosen
        # corners that were automatically detected during the undistortion steps
        #We recommend using the automatic detection of corners in your code
    src = np.float32(
        [
            [588, 459],
            [696, 459],
            [975, 637],
            [327, 637]
        ]
    )
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32(
        [
            [300, 0],
            [950, 0],
            [950, 700],
            [300, 700]
        ]
    )
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src) # This is unnecessaary for this problem
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M


def main():
    calibrate_camera()  # * dummy, done already previously

    # Read in an image
    img = cv2.imread('test_images/test6.jpg')
    # image = mpimg.imread('bridge_shadow.jpg')

    stacked, result = clr_grad_thres(img)

    '''
    # Plot the result
    f, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 3))  # ! can also display (ax1, ax2) or (ax2, ax3)
    f.tight_layout()

    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=10)

    ax2.imshow(stacked)
    ax2.set_title('Stacked Channels', fontsize=10)

    ax3.imshow(result, cmap="gray")
    ax3.set_title('clr_grad_thres Result', fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    '''

    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "camera_cal/cali_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    top_down, perspective_M = undist_warp(result, mtx, dist)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for displaying normal color
    # top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)    # for displaying normal color

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(top_down, cmap="gray")
    ax2.set_title('Undistorted and Warped Image', fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()





main()