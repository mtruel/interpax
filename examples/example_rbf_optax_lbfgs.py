from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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


def normal_pdf(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)


def normal_cdf(z: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))


def expected_improvement(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    best_observed: jnp.ndarray,
    xi: float = 0.01,
) -> jnp.ndarray:
    improvement = best_observed - mu - xi
    safe_sigma = jnp.maximum(sigma, 1e-12)
    z = improvement / safe_sigma
    ei = improvement * normal_cdf(z) + safe_sigma * normal_pdf(z)
    return jnp.where(sigma > 1e-12, ei, jnp.maximum(improvement, 0.0))


def distance_uncertainty(
    query_points: jnp.ndarray,
    observed_points: jnp.ndarray,
    observed_values: jnp.ndarray,
    bounds: Bounds,
) -> jnp.ndarray:
    deltas = query_points[:, None, :] - observed_points[None, :, :]
    min_dist = jnp.sqrt(jnp.min(jnp.sum(deltas**2, axis=-1), axis=1))
    domain_diameter = jnp.linalg.norm(bounds.high - bounds.low)
    value_scale = jnp.std(observed_values) + 1e-6
    return value_scale * (min_dist / domain_diameter)


def plot_results(
    bounds: Bounds,
    initial_points: jnp.ndarray,
    added_points: list[jnp.ndarray],
    predicted_points: list[jnp.ndarray],
    lbfgs_start_points: list[jnp.ndarray],
    last_rbf: Callable[[jnp.ndarray], jnp.ndarray] | None,
    grid_size: int,
) -> None:
    x = jnp.linspace(bounds.low[0], bounds.high[0], grid_size)
    y = jnp.linspace(bounds.low[1], bounds.high[1], grid_size)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing="xy")
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    grid_values = jax.vmap(himmelblau)(grid_points).reshape(grid_size, grid_size)
    rbf_grid_values = (
        last_rbf(grid_points).reshape(grid_size, grid_size)
        if last_rbf is not None
        else None
    )

    added_points_arr = jnp.vstack(added_points) if added_points else None
    predicted_points_arr = jnp.vstack(predicted_points) if predicted_points else None
    lbfgs_start_points_arr = (
        jnp.vstack(lbfgs_start_points) if lbfgs_start_points else None
    )

    def add_overlays(ax: Axes) -> None:
        ax.scatter(
            initial_points[:, 0],
            initial_points[:, 1],
            s=30,
            c="white",
            edgecolors="black",
            label="initial samples",
        )

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

        if predicted_points_arr is not None:
            ax.plot(
                predicted_points_arr[:, 0],
                predicted_points_arr[:, 1],
                color="white",
                linewidth=1.5,
                alpha=0.7,
                label="optimization path",
            )

        if lbfgs_start_points_arr is not None and predicted_points_arr is not None:
            n_pairs = min(
                lbfgs_start_points_arr.shape[0], predicted_points_arr.shape[0]
            )
            for idx in range(n_pairs):
                start = lbfgs_start_points_arr[idx]
                minimum = predicted_points_arr[idx]
                label = "L-BFGS start -> minimum" if idx == 0 else None
                ax.plot(
                    [float(start[0]), float(minimum[0])],
                    [float(start[1]), float(minimum[1])],
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label=label,
                )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(float(bounds.low[0]), float(bounds.high[0]))
        ax.set_ylim(float(bounds.low[1]), float(bounds.high[1]))
        ax.legend(loc="upper right")

    if rbf_grid_values is None:
        fig, ax_true = plt.subplots(figsize=(8, 6))
        contour = ax_true.contourf(
            jnp.asarray(grid_x),
            jnp.asarray(grid_y),
            jnp.asarray(grid_values),
            levels=40,
            cmap="viridis",
        )
        fig.colorbar(contour, ax=ax_true, label="f(x, y)")
        add_overlays(ax_true)
        ax_true.set_title("True Himmelblau objective with selected optimization points")
        plt.tight_layout()
        plt.show()
        return

    fig, (ax_true, ax_rbf) = plt.subplots(
        1, 2, figsize=(14, 6), constrained_layout=True
    )
    contour_true = ax_true.contourf(
        jnp.asarray(grid_x),
        jnp.asarray(grid_y),
        jnp.asarray(grid_values),
        levels=40,
        cmap="viridis",
    )
    fig.colorbar(contour_true, ax=ax_true, label="f(x, y)")
    add_overlays(ax_true)
    ax_true.set_title("True Himmelblau objective")

    contour_rbf = ax_rbf.contourf(
        jnp.asarray(grid_x),
        jnp.asarray(grid_y),
        jnp.asarray(rbf_grid_values),
        levels=40,
        cmap="viridis",
    )
    fig.colorbar(contour_rbf, ax=ax_rbf, label="RBF(x, y)")
    add_overlays(ax_rbf)
    ax_rbf.set_title("RBF surrogate (last outer iteration)")
    fig.suptitle(
        "True objective vs. final RBF surrogate with L-BFGS start-to-minimum moves"
    )
    plt.show()


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
    init_params: jax.Array,
    fun: Callable[[jnp.ndarray], jnp.ndarray],
    max_iter: int,
    tol: float,
) -> tuple[jax.Array, optax.OptState]:
    """Run L-BFGS optimization using Optax."""

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
        params = cast(jax.Array, optax.apply_updates(params, updates))
        return params, state

    def continuing(carry: tuple[jnp.ndarray, optax.OptState]) -> jnp.ndarray:
        _, state = carry
        iter_num = optax.tree.get(state, "count")
        grad = optax.tree.get(state, "grad")
        err = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing, step, init_carry)
    return cast(jax.Array, final_params), final_state


def rbf_minimize(
    seed: int = 24,
    initial_samples: int = 10,
    fine_samples: int = 4000,
    outer_iters: int = 10,
    lbfgs_max_iter: int = 48,
    lbfgs_tol: float = 1e-6,
    kernel: str = "thin_plate_spline",
    smoothing: float = 1e-6,
    grid_size: int = 200,
    ei_xi: float = 0.01,
) -> None:
    bounds = Bounds(
        low=jnp.array([-5.0, -5.0]),
        high=jnp.array([5.0, 5.0]),
    )
    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    cloud_points = sample_uniform(subkey, initial_samples, bounds)
    values = jax.vmap(himmelblau)(cloud_points)

    print("RBF reconstructed inside L-BFGS objective (domain [-5, 5]x[-5, 5])")
    print(f"Initial samples: {initial_samples}")

    added_points = []
    predicted_points = []
    lbfgs_start_points = []
    last_rbf_for_scan = None

    for step in range(outer_iters):
        rbf_for_scan = RBFInterpolator(
            cloud_points,
            values,
            kernel=kernel,
            smoothing=smoothing,
        )
        last_rbf_for_scan = rbf_for_scan

        key, subkey = jax.random.split(key)
        fine_cloud = sample_uniform(subkey, fine_samples, bounds)
        fine_mu = rbf_for_scan(fine_cloud)
        fine_sigma = distance_uncertainty(fine_cloud, cloud_points, values, bounds)
        best_observed = jnp.min(values)
        fine_ei = expected_improvement(fine_mu, fine_sigma, best_observed, xi=ei_xi)
        best_idx = jnp.argmax(fine_ei)
        x0 = fine_cloud[best_idx]
        lbfgs_start_points.append(x0)
        u0 = box_inverse(x0, bounds)

        def surrogate(u: jnp.ndarray) -> jnp.ndarray:
            point = box_transform(u, bounds)
            return rbf_for_scan(point[None, :])[0]

        surrogate_jit = jax.jit(surrogate)
        u_star, _ = run_lbfgs(u0, surrogate_jit, lbfgs_max_iter, lbfgs_tol)
        x_star = box_transform(u_star, bounds)

        pred_star = rbf_for_scan(x_star[None, :])[0]
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

    initial_points = cloud_points[:initial_samples]
    plot_results(
        bounds,
        initial_points,
        added_points,
        predicted_points,
        lbfgs_start_points,
        last_rbf_for_scan,
        grid_size,
    )


if __name__ == "__main__":
    rbf_minimize()
