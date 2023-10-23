from dual_quaternion import Quaternion,DualQuaternion
import numpy as np
from math import pi

v = np.array([ 1,0,0 ])

r = Quaternion.from_axis_angle(np.array([0,1,0]),pi/2)
translation = np.array([50,40,30])

q = DualQuaternion.from_rot_trans(r,translation)

print(q.transform_vector(v)) # [50,40,1]