from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from interpax._rbf import RBFInterpolator

CMAP = "viridis"


def main(
    seed: int = 0,
    n: int = 100,
    grid_size: int = 400,
    neighbors: int = 50,
) -> None:
    # Generate random scattered data points
    key = jax.random.PRNGKey(seed)
    cloud_points = jax.random.uniform(key, (n, 2))
    values = jnp.sin(2 * jnp.pi * cloud_points[:, 0]) * jnp.cos(
        2 * jnp.pi * cloud_points[:, 1]
    )

    rbf = RBFInterpolator(
        cloud_points,
        values,
        kernel="thin_plate_spline",
        neighbors=neighbors,
    )

    rbf = jax.jit(rbf)

    # For jacobian evaluation with vmap
    def rbf_single(point: jax.Array) -> jax.Array:
        return rbf(point[None, :])[0]

    # Evaluate on a grid for visualization
    grid_x, grid_y = jnp.mgrid[0 : 1 : grid_size * 1j, 0 : 1 : grid_size * 1j]
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    rbf_values = rbf(grid_points).reshape(grid_size, grid_size)
    rbf_gradients = jax.vmap(jax.grad(rbf_single))(grid_points)

    residuals = values - rbf(cloud_points)
    print(f"Max interpolation residual: {jnp.abs(residuals).max():.3e}")

    norm = Normalize(
        vmin=min(rbf_values.min(), values.min()),
        vmax=max(rbf_values.max(), values.max()),
    )

    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        grid_x,
        grid_y,
        rbf_values,
        cmap=CMAP,
        norm=norm,
    )
    ax.scatter(
        cloud_points[:, 0],
        cloud_points[:, 1],
        c=values,
        cmap=CMAP,
        norm=norm,
        edgecolors="k",
    )
    ax.set_title("RBF Interpolation with JAX and data points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, label="Values")

    jacobian_x = rbf_gradients[:, 0].reshape(grid_size, grid_size)
    jacobian_y = rbf_gradients[:, 1].reshape(grid_size, grid_size)

    fig_jac, axes = plt.subplots(1, 2, figsize=(12, 5))
    im_x = axes[0].pcolormesh(
        grid_x,
        grid_y,
        jacobian_x,
        cmap=CMAP,
    )
    axes[0].set_title("Jacobian d/dx")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig_jac.colorbar(im_x, ax=axes[0], label="d/dx")

    im_y = axes[1].pcolormesh(
        grid_x,
        grid_y,
        jacobian_y,
        cmap=CMAP,
        shading="auto",
    )
    axes[1].set_title("Jacobian d/dy")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    fig_jac.colorbar(im_y, ax=axes[1], label="d/dy")

    plt.show()


if __name__ == "__main__":
    main()
