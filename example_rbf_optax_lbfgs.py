from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from interpax import RBFInterpolator


@dataclass(frozen=True)
class Bounds:
    low: jnp.ndarray
    high: jnp.ndarray


def himmelblau(point: jnp.ndarray) -> jnp.ndarray:
    x, y = point
    return (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2


def sample_uniform(key: jax.Array, n: int, bounds: Bounds) -> jnp.ndarray:
    return jax.random.uniform(
        key,
        (n, 2),
        minval=bounds.low,
        maxval=bounds.high,
    )


def box_transform(u: jnp.ndarray, bounds: Bounds) -> jnp.ndarray:
    center = 0.5 * (bounds.low + bounds.high)
    half = 0.5 * (bounds.high - bounds.low)
    return center + half * jnp.tanh(u)


def box_inverse(x: jnp.ndarray, bounds: Bounds) -> jnp.ndarray:
    center = 0.5 * (bounds.low + bounds.high)
    half = 0.5 * (bounds.high - bounds.low)
    scaled = (x - center) / half
    scaled = jnp.clip(scaled, -0.999999, 0.999999)
    return jnp.arctanh(scaled)


def run_lbfgs(
    init_params: jnp.ndarray,
    fun: Callable[[jnp.ndarray], jnp.ndarray],
    max_iter: int,
    tol: float,
) -> tuple[jnp.ndarray, optax.OptState]:
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(
        carry: tuple[jnp.ndarray, optax.OptState],
    ) -> tuple[jnp.ndarray, optax.OptState]:
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad,
            state,
            params,
            value=value,
            grad=grad,
            value_fn=fun,
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing(carry: tuple[jnp.ndarray, optax.OptState]) -> jnp.ndarray:
        _, state = carry
        iter_num = optax.tree.get(state, "count")
        grad = optax.tree.get(state, "grad")
        err = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing, step, init_carry)
    return final_params, final_state


def adaptive_rbf_minimize(
    seed: int = 0,
    initial_samples: int = 5,
    fine_samples: int = 2000,
    outer_iters: int = 8,
    lbfgs_max_iter: int = 64,
    lbfgs_tol: float = 1e-6,
    kernel: str = "thin_plate_spline",
    smoothing: float = 1e-6,
    grid_size: int = 200,
) -> None:
    bounds = Bounds(
        low=jnp.array([-5.0, -5.0]),
        high=jnp.array([5.0, 5.0]),
    )
    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    cloud_points = sample_uniform(subkey, initial_samples, bounds)
    values = jax.vmap(himmelblau)(cloud_points)

    print("Adaptive RBF + L-BFGS on Himmelblau (domain [-5, 5]^2)")
    print(f"Initial samples: {initial_samples}")

    added_points = []
    predicted_points = []

    for step in range(outer_iters):
        rbf = RBFInterpolator(
            cloud_points,
            values,
            kernel=kernel,
            smoothing=smoothing,
        )

        key, subkey = jax.random.split(key)
        fine_cloud = sample_uniform(subkey, fine_samples, bounds)
        fine_pred = rbf(fine_cloud)
        best_idx = jnp.argmin(fine_pred)
        x0 = fine_cloud[best_idx]

        u0 = box_inverse(x0, bounds)

        def surrogate(u: jnp.ndarray) -> jnp.ndarray:
            point = box_transform(u, bounds)
            return rbf(point[None, :])[0]

        run_lbfgs_jit = jax.jit(
            lambda u: run_lbfgs(u, surrogate, lbfgs_max_iter, lbfgs_tol)
        )
        u_star, _ = run_lbfgs_jit(u0)
        x_star = box_transform(u_star, bounds)

        pred_star = rbf(x_star[None, :])[0]
        true_star = himmelblau(x_star)

        predicted_points.append(x_star)
        added_points.append(x_star)
        cloud_points = jnp.vstack([cloud_points, x_star[None, :]])
        values = jnp.concatenate([values, jnp.atleast_1d(true_star)])

        best_true_idx = jnp.argmin(values)
        best_true = cloud_points[best_true_idx]
        best_val = values[best_true_idx]

        print(
            "step={:02d} pred_min={:.4f} true_at_pred={:.4f} "
            "best_true={:.4f} @ ({:.3f}, {:.3f})".format(
                step + 1,
                float(pred_star),
                float(true_star),
                float(best_val),
                float(best_true[0]),
                float(best_true[1]),
            )
        )

    x = jnp.linspace(bounds.low[0], bounds.high[0], grid_size)
    y = jnp.linspace(bounds.low[1], bounds.high[1], grid_size)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing="xy")
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    grid_values = jax.vmap(himmelblau)(grid_points).reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(
        jnp.asarray(grid_x),
        jnp.asarray(grid_y),
        jnp.asarray(grid_values),
        levels=40,
        cmap="viridis",
    )
    fig.colorbar(contour, ax=ax, label="f(x, y)")

    initial_points = cloud_points[:initial_samples]
    ax.scatter(
        initial_points[:, 0],
        initial_points[:, 1],
        s=30,
        c="white",
        edgecolors="black",
        label="initial samples",
    )

    added_points_arr = jnp.vstack(added_points) if added_points else None
    if added_points_arr is not None:
        ax.scatter(
            added_points_arr[:, 0],
            added_points_arr[:, 1],
            s=60,
            c="orange",
            edgecolors="black",
            marker="*",
            label="added minima",
        )

    predicted_points_arr = jnp.vstack(predicted_points) if predicted_points else None
    if predicted_points_arr is not None:
        ax.plot(
            predicted_points_arr[:, 0],
            predicted_points_arr[:, 1],
            color="white",
            linewidth=1.5,
            alpha=0.7,
            label="optimization path",
        )

    ax.set_title("Adaptive sampling with RBF surrogate + L-BFGS")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(float(bounds.low[0]), float(bounds.high[0]))
    ax.set_ylim(float(bounds.low[1]), float(bounds.high[1]))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    adaptive_rbf_minimize()
