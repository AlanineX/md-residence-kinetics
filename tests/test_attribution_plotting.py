import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from scripts_kinetics.plotting import (
    attribution_order,
    harmonic_colors,
    plot_attribution,
)


class AttributionPlottingTests(unittest.TestCase):
    def test_palette_has_requested_size(self):
        for size in (1, 4, 7, 12, 20):
            self.assertEqual(len(harmonic_colors(size)), size)

    def test_residue_order_is_numeric(self):
        labels = ["residue_19", "residue_2", "residue_13"]
        components = np.ones((3, 2))
        order = attribution_order(labels, components, np.array([0.0, 1.0]))
        self.assertEqual([labels[i] for i in order],
                         ["residue_2", "residue_13", "residue_19"])

    def test_initial_percentage_order(self):
        labels = ["A", "B", "C"]
        components = np.array([[0.2, 0.1], [0.7, 0.3], [0.1, 0.05]])
        order = attribution_order(
            labels, components, np.array([0.0, 1.0]),
            method="initial_percentage")
        self.assertEqual([labels[i] for i in order], ["B", "A", "C"])

    def test_plot_writes_svg_for_arbitrary_component_count(self):
        time_ns = np.linspace(0.0, 1.0, 101)
        for count in (4, 12):
            components = np.asarray([
                np.exp(-time_ns * (i + 1)) / count for i in range(count)
            ])
            with tempfile.TemporaryDirectory() as outdir:
                path = plot_attribution(
                    time_ns, components.sum(axis=0), components,
                    [f"residue_{i + 1}" for i in range(count)],
                    f"n{count}", outdir, x_max_plot=1.0, n_bins=10)
                self.assertTrue(os.path.isfile(path))
                self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
