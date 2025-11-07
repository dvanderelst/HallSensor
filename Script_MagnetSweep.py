"""Batch-evaluate Hall sensor readings for a catalog of cylindrical magnets.

This script loads magnet dimensions from ``magnet_sizes.csv`` and, for each
entry, simulates the pendulum-mounted Hall sensor described in
``Script_TestMagnet``.  For the supplied swing configuration it records the
minimum and maximum magnetic field projection measured by the sensor as well as
its converted voltage.  The aggregated results are written to
``magnet_sweep_results.csv`` for easy comparison.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from Library.Magnet import MagnetDisk
from Library.Pendulum import sensor_fixed_magnet_trace


@dataclass(frozen=True)
class SweepConfig:
    """Parameters that define the pendulum sweep simulation."""

    remanence: float | str
    pivot_y_mm: float
    L_mm: float
    theta_array_deg: Iterable[float]
    sensitivity: float  # mV per gauss

    def theta_array(self) -> np.ndarray:
        return np.asarray(list(self.theta_array_deg), dtype=float)


@dataclass
class SweepResult:
    """Stores the extrema for a single magnet geometry."""

    diameter_mm: float
    thickness_mm: float
    B_min_G: float
    B_max_G: float
    V_min_mV: float
    V_max_mV: float

    @property
    def B_range_G(self) -> float:
        return self.B_max_G - self.B_min_G

    @property
    def V_range_mV(self) -> float:
        return self.V_max_mV - self.V_min_mV


def load_catalog(csv_path: Path) -> List[tuple[float, float]]:
    """Return a list of (diameter_mm, thickness_mm) pairs from the catalog CSV."""

    dimensions: List[tuple[float, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                diameter = float(row["diam_mm"])
                thickness = float(row["thick_mm"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid row in magnet catalog: {row!r}."
                ) from exc
            dimensions.append((diameter, thickness))
    return dimensions


def evaluate_magnet(diameter_mm: float, thickness_mm: float, config: SweepConfig) -> SweepResult:
    magnet = MagnetDisk(diameter_mm, thickness_mm, config.remanence)
    trace = sensor_fixed_magnet_trace(
        magnet,
        pivot_y_mm=config.pivot_y_mm,
        L_mm=config.L_mm,
        theta_array_deg=config.theta_array(),
    )

    B_read = trace["B_read"]
    voltage = B_read * config.sensitivity

    return SweepResult(
        diameter_mm=diameter_mm,
        thickness_mm=thickness_mm,
        B_min_G=float(np.min(B_read)),
        B_max_G=float(np.max(B_read)),
        V_min_mV=float(np.min(voltage)),
        V_max_mV=float(np.max(voltage)),
    )


def write_results(results: Iterable[SweepResult], output_path: Path) -> None:
    fieldnames = [
        "diameter_mm",
        "thickness_mm",
        "B_min_G",
        "B_max_G",
        "B_range_G",
        "V_min_mV",
        "V_max_mV",
        "V_range_mV",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "diameter_mm": f"{res.diameter_mm:.3f}",
                    "thickness_mm": f"{res.thickness_mm:.3f}",
                    "B_min_G": f"{res.B_min_G:.6f}",
                    "B_max_G": f"{res.B_max_G:.6f}",
                    "B_range_G": f"{res.B_range_G:.6f}",
                    "V_min_mV": f"{res.V_min_mV:.6f}",
                    "V_max_mV": f"{res.V_max_mV:.6f}",
                    "V_range_mV": f"{res.V_range_mV:.6f}",
                }
            )


def main() -> None:
    base_path = Path(__file__).resolve().parent
    catalog_path = base_path / "magnet_sizes.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {catalog_path}")

    # --- Simulation configuration (adjust to taste)
    config = SweepConfig(
        remanence="N52",
        pivot_y_mm=50.0,
        L_mm=45.0,
        theta_array_deg=np.linspace(-10, 10, 100),
        sensitivity=2.5,  # mV per gauss
    )

    magnets = load_catalog(catalog_path)
    if not magnets:
        raise ValueError("Magnet catalog is empty.")

    results = [evaluate_magnet(d, h, config) for d, h in magnets]

    output_path = base_path / "magnet_sweep_results.csv"
    write_results(results, output_path)

    print(f"Evaluated {len(results)} magnets. Results saved to {output_path}.")


if __name__ == "__main__":
    main()
