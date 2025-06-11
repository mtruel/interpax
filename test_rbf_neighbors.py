"""Test script for RBF interpolator with neighbors."""

import jax
import jax.numpy as jnp
import numpy as np
from interpax._rbf import RBFInterpolator

def test_rbf_with_neighbors():
    """Test RBF interpolator with neighbors parameter."""
    # Set random seed for reproducibility
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)
    
    # Create some test data
    points = jax.random.normal(key1, shape=(100, 2))
    values = jnp.sum(points**2, axis=1)  # Simple quadratic function
    
    # Test points
    test_points = jax.random.normal(key2, shape=(20, 2))
    
    print("Testing RBF interpolator without neighbors...")
    # Test without neighbors (global interpolation)
    rbf_global = RBFInterpolator(points, values, kernel="thin_plate_spline")
    result_global = rbf_global(test_points)
    print(f"Global interpolation shape: {result_global.shape}")
    print(f"Global interpolation mean: {jnp.mean(result_global):.4f}")
    
    print("\nTesting RBF interpolator with neighbors...")
    # Test with neighbors (local interpolation)
    rbf_local = RBFInterpolator(points, values, neighbors=10, kernel="thin_plate_spline")
    result_local = rbf_local(test_points)
    print(f"Local interpolation shape: {result_local.shape}")
    print(f"Local interpolation mean: {jnp.mean(result_local):.4f}")
    
    # Compare results
    diff = jnp.mean(jnp.abs(result_global - result_local))
    print(f"\nMean absolute difference: {diff:.4f}")
    
    # Test with different number of neighbors
    print("\nTesting with different neighbor counts...")
    for n_neighbors in [5, 15, 25]:
        rbf_n = RBFInterpolator(points, values, neighbors=n_neighbors, kernel="thin_plate_spline")
        result_n = rbf_n(test_points)
        diff_n = jnp.mean(jnp.abs(result_global - result_n))
        print(f"  {n_neighbors} neighbors - mean difference: {diff_n:.4f}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_rbf_with_neighbors() 