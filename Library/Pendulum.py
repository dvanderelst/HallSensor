import numpy as np
from scipy.spatial.transform import Rotation as R

def sensor_fixed_magnet_trace(magnet,
                              pivot_y_mm,          # pivot position in y (magnet is at origin)
                              L_mm,                # pendulum length (pivot -> sensor)
                              theta_array_deg,     # swing angles; 0° = rod along -y
                              axis0=(0, -1, 0),    # sensor sensitive axis at rest (along rod)
                              gauss=True):
    """
    Magnet fixed at origin. Pivot at (0, pivot_y_mm, 0).
    Sensor is at the pendulum tip. Pendulum swings in the x–y plane about +z.
    Returns dict with θ, sensor positions, Bx, By, Bz, |B|, and Hall reading Bread.
    """
    theta = np.deg2rad(np.asarray(theta_array_deg, float))

    # Sensor position in magnet frame:
    # pivot + Rz(θ) * (0, -L, 0)  (rod points down at θ=0)
    xs = 0.0 + L_mm * np.sin(theta)
    ys = pivot_y_mm - L_mm * np.cos(theta)
    zs = np.zeros_like(theta)

    # Field at those positions (vectorized)
    B = magnet.flux_density(xs, ys, zs, gauss=gauss)   # shape (n,3) for 1-D arrays
    if B.ndim == 1:  # single angle edge case
        B = B.reshape(1, 3)
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Sensor axis rotates with the rod
    axis0 = np.asarray(axis0, float) / np.linalg.norm(axis0)
    axes = R.from_euler('z', theta, degrees=False).apply(axis0)   # (n,3)
    B_read = Bx*axes[:,0] + By*axes[:,1] + Bz*axes[:,2]

    return dict(theta_deg=np.asarray(theta_array_deg, float),
                sensor_pos_mm=np.column_stack([xs, ys, zs]),
                Bx=Bx, By=By, Bz=Bz, Bmag=Bmag, B_read=B_read)
