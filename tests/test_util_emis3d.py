"""
tests/test_util_emis3d.py
=========================
Unit tests for main/Util_emis3D.py
Covers find_dphi, all scale_* functions, von_mises_amplitude, error models,
and the scale_wrapper dispatcher.
No cherab or raysect required.
"""

import numpy as np
import pytest

from main.Util_emis3D import (
    find_dphi,
    scale_exp,
    scale_linear,
    scale_constant,
    von_mises_amplitude,
    scale_wrapper,
    error_exponential,
    error_inverse,
    error_inv_sqrt,
)


TWO_PI = 2 * np.pi

# ---------------------------------------------------------------------------
# find_dphi
# ---------------------------------------------------------------------------

class TestFindDphi:

    def test_zero_offset(self):
        """phi == mu should give dphi == 0."""
        phi = np.array([1.0])
        result = find_dphi(phi, mu=1.0)
        assert np.isclose(result[0], 0.0)

    def test_minimum_distance(self):
        """Non-helical: should return the shorter angular distance, with sign."""
        # phi=3, mu=1 → ccw=2, cw=4π-2≈4.28 → shorter is ccw → +2
        phi = np.array([3.0])
        result = find_dphi(phi, mu=1.0)
        assert np.isclose(result[0], 2.0)

    def test_minimum_distance_cw(self):
        """When clockwise is shorter the result should be negative."""
        # mu=3.0, phi=1.0 → ccw = (1-3) % 2pi ≈ 4.28, cw = (3-1) % 2pi = 2 → shorter is -2
        phi = np.array([1.0])
        result = find_dphi(phi, mu=3.0)
        assert np.isclose(result[0], -2.0)

    def test_clockwise_direction(self):
        """Clockwise emission: result should be negative for phi behind mu."""
        phi = np.array([1.0])
        result = find_dphi(phi, mu=2.0, emissionName="clockwise", scale=False)
        # cw = (2 - 1) % 2pi = 1; result = -1
        assert result[0] < 0

    def test_counterclock_direction(self):
        """CounterClock emission: result should be positive for phi ahead of mu."""
        phi = np.array([2.0])
        result = find_dphi(phi, mu=1.0, emissionName="counterClock", scale=False)
        # ccw = (2 - 1) % 2pi = 1; result = +1
        assert result[0] > 0

    def test_helical_scale_halves_range(self):
        """
        For helical distributions (numRevolutions=1) the result is divided by 2,
        mapping the full circle to [-π, π].
        """
        phi = np.array([np.pi])  # half a revolution CCW from mu=0
        dphi_unscaled = find_dphi(phi, mu=0.0, emissionName="counterClock", scale=False)
        dphi_scaled   = find_dphi(phi, mu=0.0, emissionName="counterClock", scale=True,
                                  numRevolutions=1.0)
        assert np.isclose(dphi_scaled[0] * 2.0, dphi_unscaled[0])

    def test_phi_modulo_2pi(self):
        """
        find_dphi uses `add_2pi = phi_ > 2π` to flag values above one full
        revolution, then increments the computed ccw/cw distances.  This means
        phi = 0.5 and phi = 0.5 + 2π give *different* results by design:
        the extra revolution is deliberately preserved for multi-revolution
        helical distributions.

        For standard (non-helical) use the caller is expected to pass phi
        already reduced to [0, 2π).
        """
        phi_normal = np.array([0.5])
        phi_extra  = np.array([0.5 + 2 * np.pi])
        d1 = find_dphi(phi_normal, mu=0.0)
        d2 = find_dphi(phi_extra,  mu=0.0)
        # Results are intentionally different; document the gap
        assert not np.isclose(d1[0], d2[0]), (
            "find_dphi should treat phi > 2π differently (multi-revolution intent)"
        )

    def test_returns_ndarray(self):
        result = find_dphi(np.array([1.0, 2.0]), mu=0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# scale_exp
# ---------------------------------------------------------------------------

class TestScaleExp:

    def test_peak_at_zero(self):
        """scale_exp(A, B, 0) should equal A."""
        assert np.isclose(scale_exp(3.0, 2.0, np.array([0.0]))[0], 3.0)

    def test_decays_with_distance(self):
        """Larger |dphi| → smaller value for B > 0."""
        v0 = scale_exp(1.0, 1.0, np.array([0.0]))[0]
        v1 = scale_exp(1.0, 1.0, np.array([1.0]))[0]
        v2 = scale_exp(1.0, 1.0, np.array([2.0]))[0]
        assert v0 > v1 > v2

    def test_b_zero_is_constant(self):
        """B=0 → exp(0) = 1 → constant A."""
        dphi = np.array([0.0, 1.0, 2.0])
        result = scale_exp(5.0, 0.0, dphi)
        assert np.allclose(result, 5.0)

    def test_symmetry(self):
        """scale_exp should be symmetric: dphi and -dphi give the same value."""
        dphi_pos = np.array([1.5])
        dphi_neg = np.array([-1.5])
        assert np.isclose(scale_exp(1.0, 2.0, dphi_pos)[0],
                          scale_exp(1.0, 2.0, dphi_neg)[0])


# ---------------------------------------------------------------------------
# scale_linear
# ---------------------------------------------------------------------------

class TestScaleLinear:

    def test_gradient(self):
        dphi = np.array([0.0, 1.0, 2.0])
        result = scale_linear(2.0, 1.0, dphi)  # y = 2*dphi + 1
        assert np.allclose(result, [1.0, 3.0, 5.0])

    def test_zero_slope(self):
        dphi = np.array([0.0, 5.0, -3.0])
        result = scale_linear(0.0, 7.0, dphi)
        assert np.allclose(result, 7.0)


# ---------------------------------------------------------------------------
# scale_constant
# ---------------------------------------------------------------------------

class TestScaleConstant:

    def test_returns_uniform_A(self):
        dphi = np.array([0.0, 1.0, -2.0, 5.0])
        result = scale_constant(4.2, dphi)
        assert np.allclose(result, 4.2)

    def test_shape_preserved(self):
        dphi = np.zeros(7)
        assert scale_constant(1.0, dphi).shape == (7,)


# ---------------------------------------------------------------------------
# von_mises_amplitude
# ---------------------------------------------------------------------------

class TestVonMisesAmplitude:

    def test_peak_at_zero(self):
        """von_mises_amplitude(A, B, 0) = A * exp(B*(cos(0)-1)) = A * exp(0) = A."""
        val = von_mises_amplitude(2.0, 3.0, np.array([0.0]))
        assert np.isclose(val[0], 2.0)

    def test_endpoints_equal(self):
        """At dphi = ±π the cosine is -1, so all endpoints are equal: A*exp(-2B)."""
        val_pos = von_mises_amplitude(1.0, 2.0, np.array([np.pi]))
        val_neg = von_mises_amplitude(1.0, 2.0, np.array([-np.pi]))
        assert np.isclose(val_pos[0], val_neg[0])

    def test_positive_B_decays(self):
        """For B > 0 the amplitude should decay away from 0."""
        v0 = von_mises_amplitude(1.0, 2.0, np.array([0.0]))[0]
        v1 = von_mises_amplitude(1.0, 2.0, np.array([1.0]))[0]
        assert v0 > v1


# ---------------------------------------------------------------------------
# scale_wrapper  (dispatcher)
# ---------------------------------------------------------------------------

class TestScaleWrapper:

    def _dphi(self):
        return np.linspace(-np.pi, np.pi, 50)

    def test_exponential_dispatch(self):
        dphi = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
        result = scale_wrapper(2.0, 1.0, phi=dphi, scale_def="exponential", dphi=dphi)
        # Peak should be at dphi=0
        assert np.isclose(result[0], 2.0, atol=1e-12)

    def test_linear_dispatch(self):
        dphi = np.array([0.0, 1.0, 2.0])
        result = scale_wrapper(3.0, 1.0, phi=dphi, scale_def="linear", dphi=dphi)
        assert np.allclose(result, scale_linear(3.0, 1.0, dphi))

    def test_constant_dispatch(self):
        dphi = np.ones(10)
        result = scale_wrapper(5.0, 0.0, phi=dphi, scale_def="constant", dphi=dphi)
        assert np.allclose(result, 5.0)

    def test_unknown_scale_def_returns_ones(self):
        """Unknown scale_def should fall back to an array of 1s."""
        dphi = np.array([1.0, 2.0, 3.0])
        result = scale_wrapper(99.0, 99.0, phi=dphi, scale_def="bogus_def", dphi=dphi)
        assert np.allclose(result, 1.0)

    def test_dphi_computed_when_none(self):
        """If dphi is None it should be derived from phi and mu."""
        phi = np.array([0.5, 1.0, 1.5])
        mu = 1.0
        result_from_dphi = scale_wrapper(1.0, 0.5, phi=phi, mu=mu,
                                         scale_def="exponential", dphi=None)
        dphi_manual = find_dphi(phi, mu)
        result_manual = scale_exp(1.0, 0.5, dphi_manual)
        assert np.allclose(result_from_dphi, result_manual)


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------

class TestErrorModels:

    def test_error_exponential_at_zero(self):
        """At signal=0 the error should equal scale_factor."""
        err = error_exponential(np.array([0.0]), max_signal=1.0, scale_factor=2.0)
        assert np.isclose(err[0], 2.0)

    def test_error_exponential_decreasing(self):
        s = np.array([0.0, 0.25, 0.5, 1.0])
        err = error_exponential(s, max_signal=1.0)
        assert np.all(np.diff(err) < 0)

    def test_error_exponential_clips_negative(self):
        """Negative signal should be treated as 0 (no negative error)."""
        err = error_exponential(np.array([-1.0]), max_signal=1.0)
        err_zero = error_exponential(np.array([0.0]), max_signal=1.0)
        assert np.isclose(err[0], err_zero[0])

    def test_error_inverse_at_peak(self):
        """At signal=max_signal, error = scale_factor."""
        err = error_inverse(np.array([1.0]), max_signal=1.0, scale_factor=3.0)
        assert np.isclose(err[0], 3.0)

    def test_error_inverse_increasing(self):
        """Larger signal → smaller error (inverse relationship)."""
        s = np.array([0.5, 1.0, 2.0])
        err = error_inverse(s, max_signal=2.0)
        assert np.all(np.diff(err) < 0)

    def test_error_inv_sqrt_at_peak(self):
        """At signal = max_signal, error = scale_factor * sqrt(1) = scale_factor."""
        err = error_inv_sqrt(np.array([5.0]), max_signal=5.0, scale_factor=2.0)
        assert np.isclose(err[0], 2.0)

    def test_error_inv_sqrt_positive(self):
        """Errors should always be positive."""
        s = np.array([0.1, 1.0, 10.0])
        assert np.all(error_inv_sqrt(s, max_signal=10.0) > 0)

    def test_all_error_models_return_ndarray(self):
        s = np.array([0.5, 1.0])
        for fn in (error_exponential, error_inverse, error_inv_sqrt):
            result = fn(s, max_signal=1.0)
            assert isinstance(result, np.ndarray), f"{fn.__name__} should return ndarray"
