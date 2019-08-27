## Advanced Lane Finding

### Zheng(Jack) Zhang jack.zhang@nyu.edu

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combo.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./out_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 1 through 73 of the file called `calibrate_camera.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 27 through 59 function `clr_grad_thres(img, s_thresh=(170, 255), sx_thresh=(45, 135))` in `pipeline_video.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `undist_warp(img, mtx, dist, img_clr)`, which appears in lines 62 through 99 in the file `pipeline_video.py`.  The `undist_warp(img, mtx, dist, img_clr)` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [
        [557, 476],
        [731, 476],
        [1042, 674],
        [281, 674]
    ]
)
dst = np.float32(
    [
        [280, 0],
        [1060, 0],
        [1060, 720],
        [280, 720]
    ]
)
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 557, 476  |   280, 0    |
| 731, 476  |   1060, 0   |
| 1042, 674 |  1060, 720  |
| 281, 674  |  280, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 104 through 120 in my function `__measure_curvature_real(left_fit_cr, right_fit_cr)` and 253 through 264 in function `_gen_hud_text(avg_curverad, center_bottom_x)` in `pipeline_video.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 248 through 333 in my code in `pipeline_video.py` in the function `warp_back(top_down, left_fitx, right_fitx, Minv, img, undist_clr, top_down_poly, clr_grad_out, avg_curverad, center_bottom_x)`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here are some modifications and features I implemented to make sure my pipeline is robust enough.

1. sx_thresh = (45, 135), not (20, 100) => better expected lane line orientation
2. margin = 200 (sliding_window) and 370 (search_from_prior), not 100 and 100 => detect lane line even if big curvature and disruption (shadow)
3. smoothing is applied using a 200 frames buffer to endure big shadows

Through completing the project I find that the very first step of my pipeline system `clr_grad_thres()` often plays the determining role in whether the lane lines in tricky conditions (low-res, shawdow, sunshine). In the cases of big lane line curvature, the margins used in `sliding_window` and `search_from_prior` need to be increased, however, this may bring in possible unnecessary disruptions such as shadows and sabotage the detection. It seems that a trade-off is unavoidable using the current set of methods.

