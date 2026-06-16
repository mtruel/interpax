"""N-D unstructured interpolation using JAX."""

from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxkd as jk
import numpy as np
from jaxtyping import Array, ArrayLike

from .utils import asarray_inexact, errorif

__all__ = ["LinearNDInterpolator", "NearestNDInterpolator"]


def _ndim_coords_from_arrays(
    points: Union[tuple[ArrayLike, ...], ArrayLike],
    ndim: Optional[int] = None,
) -> Array:
    """Convert coordinate inputs to an array with trailing dimension ``ndim``.

    Mirrors ``scipy.interpolate._interpnd._ndim_coords_from_arrays``.
    """
    raw: Union[tuple[ArrayLike, ...], ArrayLike] = points
    if isinstance(raw, tuple) and len(raw) == 1:
        raw = raw[0]

    if isinstance(raw, tuple):
        arrays = [jnp.asarray(p) for p in raw]
        broadcasted = jnp.broadcast_arrays(*arrays)
        for arr in broadcasted[1:]:
            if arr.shape != broadcasted[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        return jnp.stack(broadcasted, axis=-1).astype(jnp.result_type(*broadcasted))

    arr = jnp.asarray(raw)
    if arr.ndim == 1:
        if ndim is None:
            return arr.reshape(-1, 1).astype(jnp.float64)
        return arr.reshape(-1, ndim).astype(jnp.float64)
    return arr.astype(jnp.result_type(arr, jnp.array(1.0)))


def _barycentric_coords(
    x: Array,
    simplex_verts: Array,
    eps: Array,
) -> tuple[Array, Array]:
    """Barycentric coordinates of ``x`` w.r.t. each simplex in ``simplex_verts``.

    Parameters
    ----------
    x : (ndim,) array
        Query point.
    simplex_verts : (nsimplex, ndim + 1, ndim) array
        Vertex coordinates for each simplex.
    eps : scalar array
        Tolerance for the inside-simplex test.

    Returns
    -------
    c : (nsimplex, ndim + 1) array
        Barycentric coordinates per simplex.
    inside : (nsimplex,) bool array
        Whether ``x`` lies inside each simplex (within tolerance).
    """
    ndim = x.shape[0]
    r = simplex_verts[:, ndim, :]
    t = jnp.swapaxes(simplex_verts[:, :ndim, :] - r[:, None, :], -1, -2)
    delta = x - r
    c_head = jnp.linalg.solve(t, delta[..., None])[..., 0]
    c_last = 1.0 - jnp.sum(c_head, axis=-1, keepdims=True)
    c = jnp.concatenate([c_head, c_last], axis=-1)
    finite = jnp.all(jnp.isfinite(c), axis=-1)
    inside = (
        finite
        & jnp.all(c >= -eps, axis=-1)
        & jnp.all(c <= 1.0 + eps, axis=-1)
    )
    return c, inside


def _interp_from_barycentric(
    c: Array,
    inside: Array,
    values: Array,
    simplices: Array,
    fill_value: Array,
) -> Array:
    """Select containing simplex and linearly interpolate."""
    any_inside = jnp.any(inside)
    s_idx = jnp.argmax(inside.astype(jnp.int32))
    vals_at_simplices = values[simplices]
    if vals_at_simplices.ndim == 2:
        out_per_simplex = jnp.sum(c * vals_at_simplices, axis=-1)
    else:
        out_per_simplex = jnp.sum(c[..., None] * vals_at_simplices, axis=1)
    return jnp.where(any_inside, out_per_simplex[s_idx], fill_value)


def _barycentric_coords_from_transform(
    x: Array,
    transform: Array,
    eps: Array,
) -> tuple[Array, Array]:
    """Barycentric coordinates from precomputed SciPy-style transforms."""
    ndim = x.shape[0]
    r = transform[:, ndim, :]
    tinv = transform[:, :ndim, :ndim]
    delta = x[None, :] - r
    c_head = jnp.einsum("sij,sj->si", tinv, delta)
    c_last = 1.0 - jnp.sum(c_head, axis=-1, keepdims=True)
    c = jnp.concatenate([c_head, c_last], axis=-1)
    finite = jnp.all(jnp.isfinite(c), axis=-1)
    inside = (
        finite
        & jnp.all(c >= -eps, axis=-1)
        & jnp.all(c <= 1.0 + eps, axis=-1)
    )
    return c, inside


def _build_vertex_simplex_table(
    simplices: np.ndarray,
    npoints: int,
) -> np.ndarray:
    """Build a padded vertex-to-incident-simplices lookup table."""
    incident: list[list[int]] = [[] for _ in range(npoints)]
    for s_idx, verts in enumerate(simplices):
        for v_idx in verts:
            incident[int(v_idx)].append(int(s_idx))
    max_valence = max((len(v) for v in incident), default=1)
    table = np.full((npoints, max_valence), -1, dtype=np.int32)
    for v_idx, simplex_ids in enumerate(incident):
        table[v_idx, : len(simplex_ids)] = simplex_ids
    return table


def _interp_at_point_transform(
    x: Array,
    transform: Array,
    values: Array,
    simplices: Array,
    fill_value: Array,
    eps: Array,
) -> Array:
    """Interpolate at a query point using precomputed transforms (all simplices)."""
    c, inside = _barycentric_coords_from_transform(x, transform, eps)
    return _interp_from_barycentric(c, inside, values, simplices, fill_value)


def _interp_at_point_frozen(
    x: Array,
    transform: Array,
    values: Array,
    simplices: Array,
    candidates: Array,
    fill_value: Array,
    eps: Array,
) -> Array:
    """Interpolate using incident simplices, with full-transform fallback."""
    valid = candidates >= 0
    safe_candidates = jnp.where(valid, candidates, 0)
    t_sub = transform[safe_candidates]
    c, inside = _barycentric_coords_from_transform(x, t_sub, eps)
    inside = inside & valid
    any_cand = jnp.any(inside)

    def from_candidates() -> Array:
        s_local = jnp.argmax(inside.astype(jnp.int32))
        s_idx = candidates[s_local]
        verts = simplices[s_idx]
        c_sel = c[s_local]
        vals_at = values[verts]
        if vals_at.ndim == 1:
            return jnp.sum(c_sel * vals_at)
        return jnp.sum(c_sel[..., None] * vals_at, axis=0)

    return jax.lax.cond(
        any_cand,
        from_candidates,
        lambda: _interp_at_point_transform(
            x, transform, values, simplices, fill_value, eps
        ),
    )


def _interp_at_point(
    x: Array,
    points: Array,
    values: Array,
    simplices: Array,
    fill_value: Array,
    eps: Array,
) -> Array:
    """Interpolate at a single query point using fixed topology."""
    simplex_verts = points[simplices]
    c, inside = _barycentric_coords(x, simplex_verts, eps)
    return _interp_from_barycentric(c, inside, values, simplices, fill_value)


class LinearNDInterpolator(eqx.Module):
    """Piecewise linear interpolator in N > 1 dimensions.

    Parameters
    ----------
    points : (npoints, ndims) array_like or :class:`scipy.spatial.Delaunay`
        Data point coordinates, or a precomputed Delaunay triangulation.
    values : (npoints, ...) array_like
        Data values. The length of ``values`` along the first axis must be
        equal to the number of points.
    fill_value : float, optional
        Value used for query points outside the convex hull of the input
        points. Default is ``nan``.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
    frozen_points : bool, optional
        If ``True``, fix vertex geometry after initialization: precompute
        barycentric transforms and use ``jaxkd`` nearest-vertex seeding to
        search only incident simplices. This reduces memory and compute per
        query but gradients do not flow through vertex positions. Default is
        ``False``.

    Notes
    -----
    The Delaunay triangulation is built once at initialization using
    :class:`scipy.spatial.Delaunay` (Qhull). Connectivity is fixed; query
    evaluation is fully JAX-compatible (``jit`` and automatic differentiation).

    When ``frozen_points=False`` (default), barycentric transforms are
    recomputed from ``points`` and ``simplices`` on each evaluation so
    gradients flow through vertex positions and values. Simplex location uses
    a vectorized search over all simplices with cost ``O(n_query * n_simplex)``.

    When ``frozen_points=True``, SciPy's precomputed ``transform`` is stored,
    and each query locates a containing simplex among those incident on the
    nearest mesh vertex (via ``jaxkd``), falling back to a full simplex scan
    only when needed. Gradients w.r.t. query points and ``values`` are
    supported; vertex positions are not differentiable in this mode.

    For data on a regular grid, use :func:`interpax.interp2d` or
    :func:`interpax.interp3d` instead.
    """

    __hash__ = object.__hash__

    points: Array
    values: Array
    simplices: Array
    values_shape: tuple[int, ...]
    values_dtype: jnp.dtype = eqx.field(static=True)
    fill_value: Array
    offset: Optional[Array]
    scale: Optional[Array]
    frozen_points: bool = eqx.field(static=True)
    transform: Optional[Array]
    vertex_simplex_table: Optional[Array]
    _tree: Optional[Any]

    def __init__(
        self,
        points: ArrayLike,
        values: ArrayLike,
        fill_value: Union[float, complex] = np.nan,
        rescale: bool = False,
        *,
        frozen_points: bool = False,
    ):
        from scipy.spatial import Delaunay

        tri = None
        if isinstance(points, Delaunay):
            if rescale:
                raise ValueError(
                    "Rescaling is not supported when passing a Delaunay "
                    "triangulation as ``points``."
                )
            tri = points
            pts = asarray_inexact(tri.points)
        elif isinstance(points, tuple):
            pts = _ndim_coords_from_arrays(points)
        else:
            pts = asarray_inexact(points)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 1)

        errorif(pts.ndim != 2, ValueError, "`points` must be a 2-dimensional array.")
        if pts.shape[1] < 2:
            raise ValueError("input data must be at least 2-D")

        vals = asarray_inexact(values)
        errorif(
            vals.shape[0] != pts.shape[0],
            ValueError,
            "different number of values and points",
        )

        values_shape = vals.shape[1:]
        values_dtype = vals.dtype

        offset = None
        scale = None
        if rescale:
            offset = jnp.mean(pts, axis=0)
            scale = jnp.ptp(pts, axis=0)
            scale = jnp.where(scale > 0, scale, 1.0)
            pts = (pts - offset) / scale

        if tri is None:
            tri = Delaunay(np.asarray(pts))

        if jnp.issubdtype(values_dtype, jnp.complexfloating):
            fv = jnp.asarray(fill_value, dtype=values_dtype)
        else:
            fv = jnp.asarray(
                fill_value,
                dtype=jnp.result_type(vals.dtype, jnp.float64),
            )

        self.points = pts
        self.values = vals
        self.simplices = jnp.asarray(tri.simplices)
        self.values_shape = values_shape
        self.values_dtype = values_dtype
        self.fill_value = fv
        self.offset = offset
        self.scale = scale
        self.frozen_points = frozen_points

        if frozen_points:
            self.transform = jnp.asarray(tri.transform)
            vtx_table = _build_vertex_simplex_table(
                np.asarray(tri.simplices), pts.shape[0]
            )
            self.vertex_simplex_table = jnp.asarray(vtx_table)
            self._tree = jk.build_tree(pts)
        else:
            self.transform = None
            self.vertex_simplex_table = None
            self._tree = None

    def _scale_x(self, xi: Array) -> Array:
        if self.offset is None or self.scale is None:
            return xi
        return (xi - self.offset) / self.scale

    @eqx.filter_jit
    def _evaluate_deformable(self, xi: Array) -> Array:
        eps = jnp.asarray(100 * jnp.finfo(xi.dtype).eps)
        return jax.vmap(
            _interp_at_point,
            in_axes=(0, None, None, None, None, None),
        )(xi, self.points, self.values, self.simplices, self.fill_value, eps)

    @eqx.filter_jit
    def _evaluate_frozen(self, xi: Array) -> Array:
        transform = self.transform
        tree = self._tree
        table = self.vertex_simplex_table
        if transform is None or tree is None or table is None:
            raise RuntimeError("frozen evaluation requires precomputed geometry")
        eps = jnp.asarray(100 * jnp.finfo(xi.dtype).eps)

        def one(x: Array) -> Array:
            v_idx, _ = jk.query_neighbors(tree, x[None, :], k=1)
            v_idx = jnp.reshape(v_idx, (-1,))[0]
            candidates = table[v_idx]
            return _interp_at_point_frozen(
                x,
                transform,
                self.values,
                self.simplices,
                candidates,
                self.fill_value,
                eps,
            )

        return jax.vmap(one)(xi)

    def __call__(self, *args: ArrayLike) -> Array:
        """Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array_like
            Points where to interpolate data at. Coordinate arrays may be passed
            as separate broadcastable arrays, or as a single array with shape
            ``(..., ndim)``.
        """
        ndim = self.points.shape[1]
        xi = _ndim_coords_from_arrays(args, ndim=ndim)
        if xi.shape[-1] != ndim:
            raise ValueError("number of dimensions in xi does not match x")

        original_shape = xi.shape
        xi_flat = xi.reshape(-1, ndim)
        xi_flat = self._scale_x(xi_flat)

        if self.frozen_points:
            interp_values = self._evaluate_frozen(xi_flat)
        else:
            interp_values = self._evaluate_deformable(xi_flat)

        if self.values_shape:
            new_shape = original_shape[:-1] + self.values_shape
        else:
            new_shape = original_shape[:-1]
        return interp_values.reshape(new_shape)


class NearestNDInterpolator(eqx.Module):
    """Nearest-neighbor interpolator in N > 1 dimensions.

    Parameters
    ----------
    x : (npoints, ndims) array_like
        Data point coordinates.
    y : (npoints, ...) array_like
        Data values. The length of ``y`` along the first axis must be equal to
        the length of ``x``.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have incommensurable
        units and differ by many orders of magnitude.

    Notes
    -----
    Uses ``jaxkd`` to build a k-d tree for nearest-neighbor queries. The tree
    is built once at initialization and queries are JIT-compiled.

    Unlike :class:`scipy.interpolate.NearestNDInterpolator`, ``tree_options``,
    ``p``, ``eps``, and ``workers`` are not supported. ``distance_upper_bound``
    is supported on :meth:`__call__` and emulated via returned distances.

    For data on a regular grid, use :func:`interpax.interp2d` or
    :func:`interpax.interp3d` instead.
    """

    __hash__ = object.__hash__

    points: Array
    values: Array
    values_shape: tuple[int, ...]
    values_dtype: jnp.dtype = eqx.field(static=True)
    offset: Optional[Array]
    scale: Optional[Array]
    _tree: Any

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        rescale: bool = False,
    ):
        if isinstance(x, tuple):
            x = _ndim_coords_from_arrays(x)
        else:
            x = asarray_inexact(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        errorif(x.ndim != 2, ValueError, "`x` must be a 2-dimensional array.")

        y = asarray_inexact(y)
        errorif(
            y.shape[0] != x.shape[0],
            ValueError,
            "different number of values and points",
        )
        if x.shape[1] < 2:
            raise ValueError("input data must be at least 2-D")

        values_shape = y.shape[1:]
        values_dtype = y.dtype

        offset = None
        scale = None
        if rescale:
            offset = jnp.mean(x, axis=0)
            scale = jnp.ptp(x, axis=0)
            scale = jnp.where(scale > 0, scale, 1.0)
            x = (x - offset) / scale

        self.points = x
        self.values = y
        self.values_shape = values_shape
        self.values_dtype = values_dtype
        self.offset = offset
        self.scale = scale
        self._tree = jk.build_tree(x)

    def _scale_x(self, xi: Array) -> Array:
        if self.offset is None or self.scale is None:
            return xi
        return (xi - self.offset) / self.scale

    @eqx.filter_jit
    def _evaluate(
        self,
        xi: Array,
        distance_upper_bound: Array,
    ) -> Array:
        idx, dist = jk.query_neighbors(self._tree, xi, k=1)
        idx = jnp.reshape(idx, (-1,))
        dist = jnp.reshape(dist, (-1,))

        gathered = self.values[idx]

        valid = dist <= distance_upper_bound
        if jnp.issubdtype(self.values_dtype, jnp.complexfloating):
            nan_fill = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=self.values_dtype)
        else:
            nan_fill = jnp.asarray(
                jnp.nan, dtype=jnp.result_type(self.values.dtype, jnp.float64)
            )

        if gathered.ndim > 1:
            valid = valid.reshape(valid.shape + (1,) * (gathered.ndim - 1))
        gathered = jnp.where(valid, gathered, nan_fill)
        return gathered

    def __call__(
        self,
        *args: ArrayLike,
        distance_upper_bound: Optional[Union[float, ArrayLike]] = None,
    ) -> Array:
        """Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array_like
            Points where to interpolate data at. Coordinate arrays may be passed
            as separate broadcastable arrays, or as a single array with shape
            ``(..., ndim)``.
        distance_upper_bound : float, optional
            Return ``nan`` for query points whose nearest neighbor is farther
            than this distance. Mirrors ``scipy.spatial.KDTree.query``.
        """
        ndim = self.points.shape[1]
        xi = _ndim_coords_from_arrays(args, ndim=ndim)
        if xi.shape[-1] != ndim:
            raise ValueError("number of dimensions in xi does not match x")

        original_shape = xi.shape
        xi_flat = xi.reshape(-1, ndim)
        xi_flat = self._scale_x(xi_flat)

        if distance_upper_bound is None:
            dub = jnp.asarray(jnp.inf, dtype=xi_flat.dtype)
        else:
            dub = asarray_inexact(distance_upper_bound)
            if dub.ndim != 0:
                raise ValueError("`distance_upper_bound` must be a scalar.")

        interp_values = self._evaluate(xi_flat, dub)

        if self.values_shape:
            new_shape = original_shape[:-1] + self.values_shape
        else:
            new_shape = original_shape[:-1]
        return interp_values.reshape(new_shape)
