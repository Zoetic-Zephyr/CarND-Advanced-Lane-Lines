import numpy as np

# ! GLOBALS
# Define conversions in x and y from pixels space to meters
ym_per_pix = 20/720 # meters per pixel in y dimension
xm_per_pix = 3.7/780 # meters per pixel in x dimension

thumb_ratio = 0.2   # ratio to shrink the size of detection images for overlaying on video output

H = 720
W = 1280

vid_offx, vid_offy = 20, 15

# ! GLOBALS to be modified, use "global" keyword!!!
prev_left_fit = np.array([0, 0, 0])
prev_right_fit = np.array([0, 0, 0])

sanity_ok = False