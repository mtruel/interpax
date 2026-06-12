"""N-D unstructured nearest-neighbor interpolation using JAX."""

from typing import Any, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jaxkd as jk
from jaxtyping import Array, ArrayLike

from .utils import asarray_inexact, errorif

__all__ = ["NearestNDInterpolator"]


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
