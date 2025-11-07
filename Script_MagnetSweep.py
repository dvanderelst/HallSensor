import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from Library.Magnet import MagnetDisk, field_grid
from Library.Pendulum import sensor_fixed_magnet_trace
from Library.Utils import in2mm, mm2in
from matplotlib import pyplot as plt
import matplotlib.tri as mtri

remanence = 'N52'
pivot_y_mm = 50
distance_between_sensor_magnet = 5
threshold_delta = 2500  # mV

theta_array_deg = np.linspace(-10,10, 100)
sensitivity = 2.5 #mv/Gauss

sizes = pd.read_csv('magnet_sizes.csv')
n_sizes = sizes.shape[0]

results = []
for i in range(n_sizes):
    print(i)
    current_magnet_shape = sizes.iloc[i, :]
    diameter_mm = float(current_magnet_shape.diam_mm)
    thickness_mm = float(current_magnet_shape.thick_mm)
    m = MagnetDisk(diameter_mm, thickness_mm, remanence)
    pendulum_length = (pivot_y_mm - thickness_mm / 2) - distance_between_sensor_magnet
    trace = sensor_fixed_magnet_trace(m, pivot_y_mm, pendulum_length, theta_array_deg)
    field_reading = trace['B_read']
    voltage_reading = field_reading * sensitivity

    # field_grid(m, x_extent_mm=25, y_extent_mm=25, plot='magnitude')
    # sensor_pos = trace['sensor_pos_mm']
    # sensor_pos_x = sensor_pos[:, 0]
    # sensor_pos_y = sensor_pos[:, 1]
    # plt.plot(sensor_pos_x, sensor_pos_y, color='red')
    # plt.show()


    min_field = float(np.min(field_reading))
    max_field = float(np.max(field_reading))
    min_voltage = float(np.min(voltage_reading))
    max_voltage = float(np.max(voltage_reading))
    result = [min_field, max_field, min_voltage, max_voltage]
    results.append(result)
    print(result)


# Attach results to sizes
results = pd.DataFrame(results)
results.columns = ['min_field', 'max_field', 'min_voltage', 'max_voltage']
final = pd.concat((sizes, results), axis=1)
# Add difference columns
final['delta_voltage'] = final['max_voltage'] - final['min_voltage']
final['in_range'] = final['delta_voltage'] < threshold_delta  # mV
# Sort by delta_voltage
final = final.sort_values(by='delta_voltage', ascending=False)
final.to_excel('sweep_magnet.xlsx')

# Create a scatter plot of diameter vs thickness, colored by delta_voltage
# Interpolate the results to get a surface plot

x = final['diam_mm'].to_numpy()
y = final['thick_mm'].to_numpy()
z = final['delta_voltage'].to_numpy()
low_mask = z < threshold_delta

tri = mtri.Triangulation(x, y)

plt.figure()
cs = plt.tricontourf(tri, z, levels=20, cmap='viridis')
plt.triplot(tri, color='k', alpha=0.2, linewidth=0.5)  # optional: show triangles
plt.scatter(x[low_mask], y[low_mask], facecolors='none', linewidths=1, edgecolors='k')
plt.colorbar(cs, label='Delta Voltage (mV)')
plt.xlabel('Diameter (mm)')
plt.ylabel('Thickness (mm)')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title('Magnet Size Sweep – Triangulated Surface')
plt.show()




