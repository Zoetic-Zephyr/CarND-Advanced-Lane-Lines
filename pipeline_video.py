import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from find_lanes_utils import find_lane_pixels, search_around_poly
from global_vars import ym_per_pix, xm_per_pix, thumb_ratio, H, W, vid_offx, vid_offy, \
     prev_left_fit, prev_right_fit, sanity_ok

'''
FINE-TUNING GUIDE:
1. sx_thresh = (45, 135), not (20, 100) => better expected lane line angle
2. margin = 200, not 100 => too narrow, may not detect lane line if big curvature and disruption (shadow)
3. smoothing => big shadows
'''

def calibrate_camera():
    # * dummy function only need to be run once, check calibrate_camera.py for details
    pass


def clr_grad_thres(img, s_thresh=(170, 255), sx_thresh=(45, 135)):
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
    # * color_binary is just for demo
    # Stack each channel
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


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
            [1060, H],
            [280, H]
        ]
    )
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src) # This is unnecessaary for this problem
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv, undist_clr


def fit_polynomial(binary_warped):
    def _sanity_check(left_fit, right_fit, left_fit_cr, right_fit_cr):
        def __measure_curvature_real(left_fit_cr, right_fit_cr):
            # !Calculates the curvature of polynomial functions in meters.
            
            # Define y-value where we want radius of curvature
            # We'll choose the maximum y-value, corresponding to the bottom of the image
            y_eval = H
            
            ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
            # ! curverad = (1+(2*A*y+B)**2)**(3/2)/abs(2*A)
            A = left_fit_cr[0]
            B = left_fit_cr[1]
            left_curverad = (1+(2*A*y_eval*ym_per_pix+B)**2)**(3/2)/np.absolute(2*A)  ## Implement the calculation of the left line here
            A = right_fit_cr[0]
            B = right_fit_cr[1]
            right_curverad = (1+(2*A*y_eval*ym_per_pix+B)**2)**(3/2)/np.absolute(2*A)  ## Implement the calculation of the right line here
            
            return left_curverad, right_curverad


        curve_ok, lane_width_ok = False, False
        avg_curverad = -999
        # * Check for similar curvature (roughly parallel)
        left_curverad, right_curverad = __measure_curvature_real(left_fit_cr, right_fit_cr)
        if np.absolute(left_curverad - right_curverad)  < 50000:    # TODO: huge threshold
            curve_ok = True
            avg_curverad = (left_curverad + right_curverad) / 2
        # print(avg_curverad)

        # * Check for lane_width = 3.7
        # ! left&right_bottom_x here is in real world!
        left_bottom_x = left_fit[0]*H**2 + left_fit[1]*H + left_fit[2]
        right_bottom_x = right_fit[0]*H**2 + right_fit[1]*H + right_fit[2]
        bottom_width = (right_bottom_x - left_bottom_x) * xm_per_pix
        # print(bottom_width)

        left_middle_x = left_fit[0]*360**2 + left_fit[1]*360 + left_fit[2]
        right_middle_x = right_fit[0]*360**2 + right_fit[1]*360 + right_fit[2]
        middle_width = (right_middle_x - left_middle_x) * xm_per_pix
        # print(middle_width)

        left_top_x = left_fit[2]
        right_top_x = right_fit[2]
        top_width = (right_top_x - left_top_x) * xm_per_pix
        # print(top_width)

        if (bottom_width > 3.2 and bottom_width < 4.0) and (middle_width > 3.2 and middle_width < 4.2) and (top_width > 3.2 and top_width < 4.2):
            lane_width_ok = True

        # ! _sanity_check can never be satisfied somehow... 
        return (curve_ok and lane_width_ok), avg_curverad
        # return True, avg_curverad

    
    global sanity_ok
    if not sanity_ok:
        # Find our lane pixels first
        leftx, lefty, rightx, righty, top_down_poly = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty, top_down_poly = search_around_poly(binary_warped, prev_left_fit, prev_right_fit)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # ! np.polyfit() outputs the parameters of the 二项式
    # print(left_fit)

    ##### TO-DO: Fit new polynomials to x,y in world space #####
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # ! Does the detection make sense?
    sanity_ok, avg_curverad = _sanity_check(left_fit, right_fit, left_fit_cr, right_fit_cr)

    # * left&right_bottom_x is in pixel frame
    left_bottom_x = left_fit[0]*H**2 + left_fit[1]*H + left_fit[2]
    right_bottom_x = right_fit[0]*H**2 + right_fit[1]*H + right_fit[2]
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
    top_down_poly[lefty, leftx] = [255, 0, 0]
    top_down_poly[righty, rightx] = [0, 0, 255]

    # TODO: draw lane lines on output video
    '''
    # ! Comment together
    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    '''
    return top_down_poly, \
           center_bottom_x, left_fitx, right_fitx, avg_curverad, \
           left_fit, right_fit


def warp_back(top_down, left_fitx, right_fitx, Minv, \
    img, undist_clr, top_down_poly, clr_grad_out, \
    avg_curverad, center_bottom_x):
    # ! Warp the detected lane boundaries back onto the original image.

    def _gen_hud_text(avg_curverad, center_bottom_x):
        # Calculate the radius of curvature in meters for both lane lines
        hud_text = "Curvature Radius = " + str("{0:.2f}".format(avg_curverad)) + "(m)\t"

        vehicle_offset = (667 - center_bottom_x) * xm_per_pix

        if vehicle_offset >= 0:
            hud_text += "Vehicle at right: " + str("{0:.2f}".format(vehicle_offset)) + "(m)"
        else:
            hud_text += "Vehicle at left: " + str("{0:.2f}".format(-vehicle_offset)) + "(m)"

        return hud_text

    ploty = np.linspace(0, 719, num=H)# to cover same y-range as image

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

    # ! Text Overlay
    # * Block below is not encapsulated into a text_overlay() function because cv2.putText() returns void
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText1 = (900, 50)
    bottomLeftCornerOfText2 = (900, 130)
    fontScale              = 0.7
    fontColor              = (255,255,255)
    lineType               = 2
    hud_text               = _gen_hud_text(avg_curverad, center_bottom_x)

    curverad_text = hud_text.split("\t")[0]
    offset_text = hud_text.split("\t")[1]

    cv2.putText(newwarp,curverad_text, 
        bottomLeftCornerOfText1, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.putText(newwarp,offset_text, 
        bottomLeftCornerOfText2, 
        font, 
        fontScale,
        fontColor,
        lineType)


    # * Combine the result with the original image
    blend_vidout = cv2.addWeighted(src1=undist_clr, alpha=1, src2=newwarp, beta=0.3, gamma=0)

    # * Add rectangle to image for display detection results and curverad&offset data
    thumb_w, thumb_h = int(thumb_ratio * W), int(thumb_ratio * H)

    mask = blend_vidout.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(W, thumb_h+2*vid_offy), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_vidout = cv2.addWeighted(src1=blend_vidout, alpha=0.8, src2=mask, beta=0.2, gamma=0)

    # * Add overlay of detection image to video output
    clr_grad_out = cv2.resize(clr_grad_out, dsize=(thumb_w, thumb_h))
    clr_grad_out = np.dstack((clr_grad_out, clr_grad_out, clr_grad_out)) * 255

    top_down = cv2.resize(top_down, dsize=(thumb_w, thumb_h))
    top_down = np.dstack((top_down, top_down, top_down)) * 255

    top_down_poly = cv2.resize(top_down_poly, dsize=(thumb_w, thumb_h))

    blend_vidout[vid_offy:thumb_h+vid_offy, vid_offx:vid_offx+thumb_w, :] = clr_grad_out
    blend_vidout[vid_offy:thumb_h+vid_offy, 2*vid_offx+thumb_w:2*(vid_offx+thumb_w), :] = top_down
    blend_vidout[vid_offy:thumb_h+vid_offy, 3*vid_offx+2*thumb_w:3*(vid_offx+thumb_w), :] = top_down_poly
    # blend_vidout = cv2.addWeighted(blend_vidout, 1, top_down_poly, 0.5, 0)

    return blend_vidout


def process_image(img):
    calibrate_camera()  # * dummy, done already previously

    clr_grad_out = clr_grad_thres(img)

    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "camera_cal/cali_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    top_down, perspective_M, Minv, undist_clr = undist_warp(clr_grad_out, mtx, dist, img)

    top_down_poly, center_bottom_x, left_fitx, right_fitx, avg_curverad, \
    left_fit, right_fit = fit_polynomial(top_down)

    # ! update GLOBALS for left*right_fit
    global prev_left_fit
    global prev_right_fit
    prev_left_fit = left_fit
    prev_right_fit = right_fit

    # ! add checks & improvements here

    blend_vidout = warp_back(top_down, left_fitx, right_fitx, Minv, \
                             img, undist_clr, top_down_poly, clr_grad_out, \
                             avg_curverad, center_bottom_x)

    return blend_vidout


def main():
    out_name = 'out_project_video_test2.mp4'
    # clip1 = VideoFileClip("project_video.mp4").subclip(0,1)
    # clip1 = VideoFileClip("project_video.mp4").subclip(21, 25)
    clip1 = VideoFileClip("project_video.mp4").subclip(38, 43)
    # clip1 = VideoFileClip("project_video.mp4")
    out_video = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_video.write_videofile(out_name, audio=False)


main()
