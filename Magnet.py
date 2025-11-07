import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

tesla_to_gauss = 1e4
mm_to_m = 0.001

remanence_lookup = {
    'N30': 1.05, 'N33': 1.10, 'N35': 1.17,
    'N38': 1.22, 'N40': 1.27, 'N42': 1.30,
    'N45': 1.35, 'N48': 1.38, 'N50': 1.42,
    'N52': 1.45, 'N55': 1.50,
    # High-temp variants (lower Br)
    'N42SH': 1.25, 'N48H': 1.32, 'N35EH': 1.10
}



class MagnetDisk:
    """Axis along +y; dimensions in meters; Br in tesla or grade string like 'N52'."""

    # make the module-level lookup available as a class attribute
    remanence_lookup = remanence_lookup

    def __init__(self, diameter, thickness, remanence):
        # Convert grade strings to Br (tesla)
        if isinstance(remanence, str):
            key = remanence.upper().strip()
            if key not in self.remanence_lookup: raise ValueError(f"Unknown magnet grade {key}.")
            remanence = self.remanence_lookup[key]

        self.radius_mm = float(diameter / 2.0)
        self.thickness_mm = float(thickness)
        self.remanence = float(remanence)

        d = 2 * self.radius_mm * mm_to_m
        h = self.thickness_mm * mm_to_m
        self.m = magpy.magnet.Cylinder(dimension=(d, h), polarization=(0, 0, self.remanence))
        # Rotate cylinder so its z-axis aligns with +y in your coordinate frame
        self.m.orientation = R.from_euler('x', 90, degrees=True)

    def flux_density(self, x, y, z=0.0, gauss=True):
        # broadcast so scalars/grids mix cleanly
        xb, yb, zb = np.broadcast_arrays(x, y, z)
        pts = np.column_stack([xb.ravel(), yb.ravel(), zb.ravel()])
        flux_density = magpy.getB(self.m, pts * mm_to_m)
        if gauss: flux_density = flux_density * tesla_to_gauss
        return flux_density.reshape(xb.shape + (3,))

    def flux(self, x, y, z=0.0, gauss=True):
        vector = self.flux_density(x, y, z=z, gauss=gauss)
        magnitude =  np.linalg.norm(vector, axis=-1)
        return vector, magnitude

    def flux_density_components(self, x, y, z=0.0, gauss=True, components='y'):
        vector = self.flux_density(x, y, z, gauss=gauss)
        if components == 'x': return vector[..., 0]
        if components == 'y': return vector[..., 1]
        if components == 'z': return vector[..., 2]
        raise ValueError("Select a valid component.")

    @classmethod
    def available_grades(cls):
        """Return list of available grade strings."""
        return sorted(cls.remanence_lookup.keys())


def field_grid(magnet, x_extent_mm, y_extent_mm, steps=None, plot=None, z_mm=0.0, gauss=True):
    if steps is None: steps = 100
    if isinstance(steps, int):
        nx = ny = steps
    else:
        nx, ny = steps
    x = np.linspace(-x_extent_mm, x_extent_mm, nx)
    y = np.linspace(-y_extent_mm, y_extent_mm, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    Z = np.full_like(X, float(z_mm))

    # --- field
    B = magnet.flux_density(X, Y, Z, gauss=gauss)  # shape (ny, nx, 3)
    Bx = B[..., 0]
    By = B[..., 1]
    Bz = B[..., 2]
    Bmag = np.linalg.norm(B, axis=-1)

    # --- optional plot
    if plot is not None:
        if plot.lower() in ('x', 'bx'):
            data = Bx
            title = 'B_x'
        elif plot.lower() in ('y', 'by'):
            data = By
            title = 'B_y'
        elif plot.lower() in ('z', 'bz'):
            data = Bz
            title = 'B_z'
        elif plot.lower() in ('magnitude', 'mag', 'norm'):
            data = Bmag
            title = '|B|'
        else:
            raise ValueError("plot must be one of {'x','y','z','magnitude'}")
        # --- contour plot
        cmap = 'viridis'
        levels = np.arange(0, np.max(data[:]), 1000)
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, data, levels=levels, cmap=cmap)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label('Gauss' if gauss else 'Tesla')
        # draw magnet outline (centered at origin, axis = +y)
        w = 2 * magnet.radius_mm  # width along x (mm)
        h = magnet.thickness_mm  # height along y (mm)
        print(w, h)
        rect = Rectangle( (-magnet.radius_mm, -h / 2), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'{title} on z={z_mm} mm')
        plt.axis('equal')
        #plt.show()

    return X, Y, Bx, By, Bz, Bmag
