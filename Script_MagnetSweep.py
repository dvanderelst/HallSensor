import numpy as np
import pandas as pd
from Library.Magnet import MagnetDisk, field_grid
from Library.Pendulum import sensor_fixed_magnet_trace
from Library.Utils import frac_to_unicode
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Normalize


remanence = 'N52'
pivot_y_mm = 50 # mm
distance_between_sensor_magnet = 5 #mm
threshold_delta = 2500  # mV
max_diameter_plot = 15  # mm
max_thickness_plot = 10  # mm

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


z_plot = np.minimum(z, threshold_delta)
fig, ax = plt.subplots()
tri = mtri.Triangulation(x, y)

vmin = z.min();vmax = threshold_delta   # <-- upper limit of the colormap (everything above is "over")
# Copy the colormap so we can modify it
cmap = plt.get_cmap('hot').copy()
cmap.set_over('gray')   # <-- set the color for values above threshold
cs = ax.tricontourf(tri, z,cmap=cmap,norm=Normalize(vmin=vmin, vmax=vmax), extend='max')

ax.triplot(tri, color='k', alpha=0.2, linewidth=0.5)
low_mask = z < threshold_delta
ax.scatter(x[low_mask], y[low_mask], facecolors='none', linewidths=1, edgecolors='k')

ax.set_xlabel('Diameter (mm)')
ax.set_ylabel('Thickness (mm)')
# keep limits tight to the available data range instead of a hard-coded window
ax.set_xlim(0, x.max())
ax.set_ylim(0, y.max())
ax.set_title('Magnet Size Sweep – Triangulated Surface')

# secondary axes with inch fractions
diam_ticks_mm = np.sort(final['diam_mm'].unique())
diam_labels_in = [final.loc[final['diam_mm'] == mm, 'diam_in'].iloc[0] for mm in diam_ticks_mm]
thick_ticks_mm = np.sort(final['thick_mm'].unique())
thick_labels_in = [final.loc[final['thick_mm'] == mm, 'thick_in'].iloc[0] for mm in thick_ticks_mm]

ax_top = ax.secondary_xaxis('top')
ax_right = ax.secondary_yaxis('right')
ax_top.set_xticks(diam_ticks_mm);  ax_top.set_xticklabels([frac_to_unicode(s) for s in diam_labels_in])
ax_right.set_yticks(thick_ticks_mm); ax_right.set_yticklabels([frac_to_unicode(s) for s in thick_labels_in])
ax_top.set_xlabel('Diameter (inch fraction)')
ax_right.set_ylabel('Thickness (inch fraction)')

dia_min = sizes.diam_mm.min()
tck_min = sizes.thick_mm.min()

plt.xlim(dia_min-0.1, max_diameter_plot)
plt.ylim(tck_min-0.1, max_thickness_plot)

# ONE colorbar, placed right with spacing
cbar = fig.colorbar(cs, ax=ax, location='right', pad=0.08, shrink=0.9, aspect=25)
cbar.set_label('Delta Voltage (mV)')


fig.tight_layout()
plt.show()




