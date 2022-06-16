import numpy as np
fx = 582.993
fy = 584.414
cx = 322.604
cy = 244.141
k1 = -0.474420
k2 = 0.286146
k3 = 0
p1 = -0.000999
p2 = 0.000991
skew_c = 0
cam_K=np.asarray([[fx,   0.,         cx ],
 [  0.,         fy, cy],
 [  0.,           0.,           1.        ]])