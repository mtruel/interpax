"""Tests for the JAX RBF interpolator."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.interpolate import RBFInterpolator as ScipyRBFInterpolator

from interpax._rbf import RBFInterpolator

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

def test_neighbors_basic():
    """Test basic neighbors functionality."""
    points, values = _get_test_data_2d()
    
    # Create interpolators with and without neighbors
    rbf_global = RBFInterpolator(points, values, kernel="thin_plate_spline")
    rbf_local = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=10)
    
    # Test points
    x_test = np.linspace(0, 1, 10)
    y_test = np.linspace(0, 1, 10)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    points_test = np.column_stack((X_test.ravel(), Y_test.ravel()))
    
    # Evaluate both
    values_global = rbf_global(points_test)
    values_local = rbf_local(points_test)
    
    # Results should be finite and of correct shape
    assert values_global.shape == values_local.shape
    assert np.all(np.isfinite(values_global))
    assert np.all(np.isfinite(values_local))
    
    # Results should be similar but not necessarily identical
    # (local interpolation will be different from global)
    correlation = np.corrcoef(values_global, values_local)[0, 1]
    assert correlation > 0.7  # Should be reasonably correlated

@pytest.mark.parametrize("neighbors", [1, 5, 10, 15])
def test_neighbors_count(neighbors):
    """Test different neighbor counts."""
    points, values = _get_test_data_2d()
    
    # Create interpolator with neighbors
    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=neighbors)
    
    # Test points
    points_test = np.array([[0.5, 0.5], [0.2, 0.8]])
    
    # Evaluate
    values_test = rbf(points_test)
    
    # Results should be finite and of correct shape
    assert values_test.shape == (2,)
    assert np.all(np.isfinite(values_test))

@pytest.mark.parametrize("kernel", [
    "linear",
    "thin_plate_spline", 
    "cubic",
    "quintic"
])
def test_neighbors_kernels(kernel):
    """Test neighbors functionality with different kernels."""
    points, values = _get_test_data_2d()
    
    # Create interpolator with neighbors
    rbf = RBFInterpolator(points, values, kernel=kernel, neighbors=8)
    
    # Test points
    points_test = np.array([[0.3, 0.7], [0.8, 0.2]])
    
    # Evaluate
    values_test = rbf(points_test)
    
    # Results should be finite and of correct shape
    assert values_test.shape == (2,)
    assert np.all(np.isfinite(values_test))

def test_neighbors_vs_scipy():
    """Test neighbors against scipy implementation."""
    points, values = _get_test_data_2d()
    
    # Create interpolators
    rbf_jax = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=10)
    rbf_scipy = ScipyRBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=10)
    
    # Test points
    points_test = np.array([[0.25, 0.75], [0.6, 0.4]])
    
    # Evaluate
    values_jax = rbf_jax(points_test)
    values_scipy = rbf_scipy(points_test)
    
    # Compare - should be very close
    np.testing.assert_allclose(values_jax, values_scipy, rtol=1e-8, atol=1e-8)

def test_neighbors_edge_cases():
    """Test edge cases for neighbors."""
    points, values = _get_test_data_2d()
    n_points = len(points)
    
    # Test neighbors = 1
    rbf1 = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=1)
    points_test = np.array([[0.5, 0.5]])
    result1 = rbf1(points_test)
    assert result1.shape == (1,)
    assert np.isfinite(result1[0])
    
    # Test neighbors > number of points (should be clamped)
    rbf_large = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=n_points + 10)
    result_large = rbf_large(points_test)
    assert result_large.shape == (1,)
    assert np.isfinite(result_large[0])
    
    # Test neighbors = all points (should be similar to global)
    rbf_all = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=n_points)
    rbf_global = RBFInterpolator(points, values, kernel="thin_plate_spline")
    
    result_all = rbf_all(points_test)
    result_global = rbf_global(points_test)
    
    # Should be very close when using all neighbors
    np.testing.assert_allclose(result_all, result_global, rtol=1e-10, atol=1e-10)

def test_neighbors_with_smoothing():
    """Test neighbors with smoothing parameter."""
    points, values = _get_test_data_2d()
    
    # Create interpolator with neighbors and smoothing
    rbf = RBFInterpolator(
        points, values, 
        kernel="thin_plate_spline", 
        neighbors=8, 
        smoothing=0.1
    )
    
    # Test points
    points_test = np.array([[0.3, 0.7], [0.8, 0.2]])
    
    # Evaluate
    values_test = rbf(points_test)
    
    # Results should be finite and of correct shape
    assert values_test.shape == (2,)
    assert np.all(np.isfinite(values_test))

def test_neighbors_jit():
    """Test that neighbors work with JIT."""
    points, values = _get_test_data_2d()
    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=8)
    
    # JIT the evaluation
    jitted_eval = jax.jit(rbf)
    
    # Test points
    points_test = np.array([[0.3, 0.7], [0.8, 0.2]])
    
    # Evaluate both ways
    values_normal = rbf(points_test)
    values_jitted = jitted_eval(points_test)
    
    # Compare
    np.testing.assert_allclose(values_normal, values_jitted, rtol=1e-10, atol=1e-10)

def test_neighbors_grad():
    """Test that neighbors work with gradients."""
    points, values = _get_test_data_2d()
    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=8)
    
    def loss(x):
        return jnp.sum(rbf(x) ** 2)
    
    # Test points
    points_test = np.array([[0.3, 0.7], [0.8, 0.2]])
    
    # Compute gradient
    grad_fn = jax.grad(loss)
    grad = grad_fn(points_test)
    
    # Check that gradient is not None and has correct shape
    assert grad is not None
    assert grad.shape == points_test.shape
    assert np.all(np.isfinite(grad))

def test_neighbors_complex_values():
    """Test neighbors with complex values."""
    # Create complex test data
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = np.exp(2j * np.pi * (X + Y)).ravel()
    
    # Create interpolator with neighbors
    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", neighbors=10)
    
    # Test points
    points_test = np.array([[0.3, 0.7], [0.8, 0.2]])
    
    # Evaluate
    values_test = rbf(points_test)
    
    # Results should be finite and of correct shape
    assert values_test.shape == (2,)
    assert np.all(np.isfinite(values_test))
    assert np.iscomplexobj(values_test) 