import unittest

import numpy as np

from scripts_kinetics.kinetics import (
    _attributed_sp_sums_counts,
    _sp_sums_counts,
)


class AttributionTests(unittest.TestCase):
    def test_single_component_matches_regional_survival(self):
        sets = [{1, 2}, {1, 2}, {2}, set()]
        _, regional, contributions, counts = _attributed_sp_sums_counts(
            {"site": sets}, tau_max=2)
        sums, expected_counts = _sp_sums_counts(sets, tau_max=2)
        expected = sums / expected_counts

        np.testing.assert_allclose(regional, expected)
        np.testing.assert_allclose(contributions["site"], expected)
        np.testing.assert_array_equal(counts, expected_counts)

    def test_overlap_is_split_without_double_counting(self):
        components = {
            "A": [{1, 2}, {1, 2}],
            "B": [{1}, {1}],
        }
        _, regional, contributions, _ = _attributed_sp_sums_counts(
            components, tau_max=1)

        np.testing.assert_allclose(regional, [1.0, 1.0])
        np.testing.assert_allclose(contributions["A"], [0.75, 0.75])
        np.testing.assert_allclose(contributions["B"], [0.25, 0.25])

    def test_initial_component_keeps_credit_after_switch(self):
        components = {
            "A": [{1}, set(), set()],
            "B": [set(), {1}, {1}],
        }
        _, regional, contributions, _ = _attributed_sp_sums_counts(
            components, tau_max=2, t0_stop=1)

        np.testing.assert_allclose(regional, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(contributions["A"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(contributions["B"], [0.0, 0.0, 0.0])

    def test_components_close_to_independent_union_curve(self):
        components = {
            "A": [{1, 2}, {1}, {1}, set()],
            "B": [{2, 3}, {2, 3}, {3}, {3}],
        }
        _, regional, contributions, _ = _attributed_sp_sums_counts(
            components, tau_max=2)
        additive = np.sum(np.stack(list(contributions.values())), axis=0)

        np.testing.assert_allclose(additive, regional, atol=1e-15)

    def test_rejects_mismatched_series_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            _attributed_sp_sums_counts({"A": [{1}], "B": [{1}, {1}]}, 0)


if __name__ == "__main__":
    unittest.main()
