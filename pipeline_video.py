import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ! GLOBALS
# Define conversions in x and y from pixels space to meters
ym_per_pix = 20/720 # meters per pixel in y dimension
xm_per_pix = 3.7/780 # meters per pixel in x dimension


def calibrate_camera():
    # * dummy function only need to be run once, check calibrate_camera.py for details
    pass


def clr_grad_thres(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

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


def undist_warp(img, mtx, dist, img_clr):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist_clr = cv2.undistort(img_clr, mtx, dist, None, mtx)
    img_size = (undist.shape[1], undist.shape[0])

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
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32(
        [
            [280, 0],
            [1060, 0],
            [1060, 720],
            [280, 720]
        ]
    )
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src) # This is unnecessaary for this problem
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv, undist_clr


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50 # ! 50 is recommended, but I think it should be way more

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        # ! low means lower boundary, high means higher boundary
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        # ! need to concatenate because pixel ids are grouped by windows
        # print(left_lane_inds)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # ! np.polyfit() outputs the parameters of the 二项式
    # print(left_fit)

    # ! ym and xm are added to get the left&right_fit_cr
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/780 # meters per pixel in x dimension

    ##### TO-DO: Fit new polynomials to x,y in world space #####
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    left_bottom_x = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
    right_bottom_x = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
    center_bottom_x = (left_bottom_x + right_bottom_x) / 2

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    '''
    # ! Comment together
    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    '''
    return out_img, left_fit_cr, right_fit_cr, center_bottom_x, left_fitx, right_fitx


def measure_curvature_real(left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20/720 # meters per pixel in y dimension
    # ! techniquelly line below is useless, it is useful at line 187
    xm_per_pix = 3.7/780 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 720
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    # ! curverad = (1+(2*A*y+B)**2)**(3/2)/abs(2*A)
    A = left_fit_cr[0]
    B = left_fit_cr[1]
    left_curverad = (1+(2*A*y_eval*ym_per_pix+B)**2)**(3/2)/np.absolute(2*A)  ## Implement the calculation of the left line here
    A = right_fit_cr[0]
    B = right_fit_cr[1]
    right_curverad = (1+(2*A*y_eval*ym_per_pix+B)**2)**(3/2)/np.absolute(2*A)  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad


def main():
    calibrate_camera()  # * dummy, done already previously

    # Read in an image
    img = mpimg.imread('test_images/test5.jpg')
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

    top_down, perspective_M, Minv, undist_clr = undist_warp(result, mtx, dist, img)

    '''
    # Plot the result
    # ! line below is unecessary as now I'm using mpimg instead of cv2 to read image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for displaying normal color
    # ! line below is unecessary as top_down is grayscaled
    # top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)    # for displaying normal color

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(top_down, cmap="gray")
    ax2.set_title('Undistorted and Warped Image', fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()
    '''

    # Load our image
    # binary_warped = mpimg.imread('warped_example.jpg')

    out_img, left_fit_cr, right_fit_cr, center_bottom_x, left_fitx, right_fitx = fit_polynomial(top_down)
    '''
    # ! Comment together
    # Plot the result
    plt.imshow(out_img)
    plt.show()
    '''

    # Calculate the radius of curvature in meters for both lane lines
    left_curverad, right_curverad = measure_curvature_real(left_fit_cr, right_fit_cr)
    vehicle_offset = (667 - center_bottom_x) * xm_per_pix

    print(left_curverad, 'm', right_curverad, 'm')
    if vehicle_offset >= 0:
        print("Vehicle at right:", vehicle_offset)
    else:
        print("Vehicle at left:", -vehicle_offset)

    # ! Warp the detected lane boundaries back onto the original image.
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(top_down).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist_clr, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.show()



main()
