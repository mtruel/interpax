"""Tests for the JAX RBF interpolator."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.interpolate import RBFInterpolator as ScipyRBFInterpolator

from interpax._rbf_jax import RBFInterpolator

# Test data
def _get_test_data_1d():
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    return x[:, None], y

def _get_test_data_2d():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = Z.ravel()
    return points, values

def _get_test_data_complex():
    x = np.linspace(0, 1, 10)
    y = np.exp(2j * np.pi * x)
    return x[:, None], y

@pytest.mark.parametrize("kernel", [
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian"
])
def test_kernels_1d(kernel):
    """Test all kernels in 1D."""
    x, y = _get_test_data_1d()
    
    # Create interpolators
    rbf_jax = RBFInterpolator(x, y, kernel=kernel)
    rbf_scipy = ScipyRBFInterpolator(x, y, kernel=kernel)
    
    # Test points
    x_test = np.linspace(0, 1, 20)[:, None]
    
    # Evaluate
    y_jax = rbf_jax(x_test)
    y_scipy = rbf_scipy(x_test)
    
    # Compare
    np.testing.assert_allclose(y_jax, y_scipy, rtol=1e-10, atol=1e-10)

@pytest.mark.parametrize("kernel", [
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian"
])
def test_kernels_2d(kernel):
    """Test all kernels in 2D."""
    points, values = _get_test_data_2d()
    
    # Create interpolators
    rbf_jax = RBFInterpolator(points, values, kernel=kernel)
    rbf_scipy = ScipyRBFInterpolator(points, values, kernel=kernel)
    
    # Test points
    x_test = np.linspace(0, 1, 10)
    y_test = np.linspace(0, 1, 10)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    points_test = np.column_stack((X_test.ravel(), Y_test.ravel()))
    
    # Evaluate
    values_jax = rbf_jax(points_test)
    values_scipy = rbf_scipy(points_test)
    
    # Compare
    np.testing.assert_allclose(values_jax, values_scipy, rtol=1e-10, atol=1e-10)

def test_complex_values():
    """Test interpolation with complex values."""
    x, y = _get_test_data_complex()
    
    # Create interpolator
    rbf_jax = RBFInterpolator(x, y, kernel="thin_plate_spline")
    rbf_scipy = ScipyRBFInterpolator(x, y, kernel="thin_plate_spline")
    
    # Test points
    x_test = np.linspace(0, 1, 20)[:, None]
    
    # Evaluate
    y_jax = rbf_jax(x_test)
    y_scipy = rbf_scipy(x_test)
    
    # Compare
    np.testing.assert_allclose(y_jax, y_scipy, rtol=1e-10, atol=1e-10)

def test_smoothing():
    """Test interpolation with smoothing."""
    x, y = _get_test_data_1d()
    smoothing = 0.1
    
    # Create interpolators
    rbf_jax = RBFInterpolator(x, y, kernel="thin_plate_spline", smoothing=smoothing)
    rbf_scipy = ScipyRBFInterpolator(x, y, kernel="thin_plate_spline", smoothing=smoothing)
    
    # Test points
    x_test = np.linspace(0, 1, 20)[:, None]
    
    # Evaluate
    y_jax = rbf_jax(x_test)
    y_scipy = rbf_scipy(x_test)
    
    # Compare
    np.testing.assert_allclose(y_jax, y_scipy, rtol=1e-10, atol=1e-10)

def test_polynomial_degree():
    """Test interpolation with different polynomial degrees."""
    x, y = _get_test_data_1d()
    
    for degree in [-1, 0, 1, 2]:
        # Create interpolators
        rbf_jax = RBFInterpolator(x, y, kernel="thin_plate_spline", degree=degree)
        rbf_scipy = ScipyRBFInterpolator(x, y, kernel="thin_plate_spline", degree=degree)
        
        # Test points
        x_test = np.linspace(0, 1, 20)[:, None]
        
        # Evaluate
        y_jax = rbf_jax(x_test)
        y_scipy = rbf_scipy(x_test)
        
        # Compare
        np.testing.assert_allclose(y_jax, y_scipy, rtol=1e-10, atol=1e-10)

def test_jit():
    """Test that the interpolator works with JIT."""
    x, y = _get_test_data_1d()
    rbf = RBFInterpolator(x, y, kernel="thin_plate_spline")
    
    # JIT the evaluation
    jitted_eval = jax.jit(rbf)
    
    # Test points
    x_test = np.linspace(0, 1, 20)[:, None]
    
    # Evaluate both ways
    y_normal = rbf(x_test)
    y_jitted = jitted_eval(x_test)
    
    # Compare
    np.testing.assert_allclose(y_normal, y_jitted, rtol=1e-10, atol=1e-10)

def test_grad():
    """Test that the interpolator is differentiable."""
    x, y = _get_test_data_1d()
    rbf = RBFInterpolator(x, y, kernel="thin_plate_spline")
    
    def loss(params, x):
        return jnp.sum(rbf(x) ** 2)
    
    # Test points
    x_test = np.linspace(0, 1, 20)[:, None]
    
    # Compute gradient
    grad_fn = jax.grad(loss, argnums=1)
    grad = grad_fn(None, x_test)
    
    # Check that gradient is not None and has correct shape
    assert grad is not None
    assert grad.shape == x_test.shape

def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    x, y = _get_test_data_1d()
    
    # Test invalid kernel
    with pytest.raises(ValueError):
        RBFInterpolator(x, y, kernel="invalid_kernel")
    
    # Test invalid degree
    with pytest.raises(ValueError):
        RBFInterpolator(x, y, degree=-2)
    
    # Test invalid smoothing shape
    with pytest.raises(ValueError):
        RBFInterpolator(x, y, smoothing=np.ones(len(x) + 1))
    
    # Test invalid input dimensions
    with pytest.raises(ValueError):
        RBFInterpolator(x, y)(np.array([0, 1]))  # 1D instead of 2D
    
    # Test invalid evaluation dimensions
    with pytest.raises(ValueError):
        RBFInterpolator(x, y)(np.array([[0, 1]]))  # Wrong number of dimensions

def test_epsilon_required():
    """Test that epsilon is required for appropriate kernels."""
    x, y = _get_test_data_1d()
    
    # These kernels require epsilon
    for kernel in ["multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"]:
        with pytest.raises(ValueError):
            RBFInterpolator(x, y, kernel=kernel)
    
    # These kernels don't require epsilon
    for kernel in ["linear", "thin_plate_spline", "cubic", "quintic"]:
        RBFInterpolator(x, y, kernel=kernel)  # Should not raise

def test_memory_chunking():
    """Test that memory chunking works for large inputs."""
    # Create a large dataset
    x = np.linspace(0, 1, 1000)[:, None]
    y = np.sin(2 * np.pi * x)
    
    # Create interpolator with small memory budget
    rbf = RBFInterpolator(x, y, kernel="thin_plate_spline")
    
    # Test points
    x_test = np.linspace(0, 1, 2000)[:, None]
    
    # Evaluate with small memory budget
    y_test = rbf._chunk_evaluator(
        x_test, x, rbf._shift, rbf._scale, rbf._coeffs, memory_budget=1000
    )
    
    # Check that the result has the correct shape
    assert y_test.shape == (2000, 1)
    
    # Check that the result is reasonable
    assert np.all(np.isfinite(y_test))
    assert np.all(np.abs(y_test) <= 1.1)  # sin is bounded by 1 