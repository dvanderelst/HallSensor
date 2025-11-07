from Library.Magnet import MagnetDisk, field_grid
from Library.Pendulum import sensor_fixed_magnet_trace
from Library.Utils import in2mm, mm2in
from matplotlib import pyplot as plt
import numpy as np


diameter_mm = in2mm(1/4)
thickness_mm = in2mm(1/32)
remanence = 'N52'

# distance along the rod from magnet center to the sensor
# (positive toward the pivot).
pivot_y_mm = 50
L_mm = 45
theta_array_deg = np.linspace(-10,10, 100)
sensitivity = 2.5 #mv/Gauss

m = MagnetDisk(diameter_mm, thickness_mm, remanence)
trace = sensor_fixed_magnet_trace(m, pivot_y_mm, L_mm, theta_array_deg)
field_grid(m, x_extent_mm=25, y_extent_mm=25, plot='magnitude')
sensor_pos = trace['sensor_pos_mm']
sensor_pos_x = sensor_pos[:, 0]
sensor_pos_y = sensor_pos[:, 1]
plt.plot(sensor_pos_x, sensor_pos_y, color='red')
plt.show()

B_read = trace['B_read']
plt.figure()
plt.subplot(211)
plt.plot(theta_array_deg, B_read, color='red')
plt.xlabel('Theta')
plt.ylabel('B_read, Gauss')
plt.subplot(212)
plt.plot(theta_array_deg, B_read * sensitivity, color='red')
plt.xlabel('Theta')
plt.ylabel('reading, mV')
plt.show()
