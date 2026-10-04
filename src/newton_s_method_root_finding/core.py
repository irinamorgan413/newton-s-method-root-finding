import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of a Newton iteration.

    Attributes:
        root: the final iterate when the solver converged; None otherwise.
        iterations: number of Newton steps taken (the initial guess is step zero,
            so the value is the count of updates applied).
        converged: True iff |f(x)| was driven below the tolerance.
        failed: True if the solver gave up without converging.
        final_x: the last x value the solver evaluated f at.
        final_fx: f(final_x); kept for diagnostics so callers can see how far off
            the function still is.
        message: short human-readable note about the stop reason.
    """
    root: Optional[float]
    iterations: int
    converged: bool
    failed: bool
    final_x: float
    final_fx: float
    message: str


def _central_difference(f: Callable[[float], float], x: float, h: float) -> float:
    # Central difference is O(h^2) and substantially more accurate than the
    # forward difference for the same step size. We use a relative step that
    # scales with |x| so the probe is well-conditioned near large and small
    # magnitudes alike.
    h = abs(h)
    return (f(x + h) - f(x - h)) / (2.0 * h)


def find_root(
    f: Callable[[float], float],
    x0: float,
    *,
    fprime: Optional[Callable[[float], float]] = None,
    tol: float = 1e-10,
    maxiter: int = 50,
    dx: float = 1e-6,
) -> NewtonResult:
    """Find a scalar root of f near x0 using Newton's method.

    The function accepts a callable returning a scalar for a scalar input.
    When fprime is not supplied, the derivative is approximated by a central
    difference with step dx scaled by max(1, |x|) at each iterate.

    Returns a NewtonResult describing the outcome. Convergence is decided on
    |f(x)| < tol rather than on step size, because the function value is the
    quantity the caller actually cares about driving to zero and it avoids a
    second tuning parameter.

    The method gives up (failed=True) if the derivative is exactly zero at an
    iterate, an iterate is non-finite, or the iteration count is exhausted
    without converging.
    """
    if not callable(f):
        raise TypeError("f must be callable")
    if fprime is not None and not callable(fprime):
        raise TypeError("fprime must be callable or None")
    if not math.isfinite(x0):
        raise ValueError("x0 must be finite")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be a positive finite number")

    x = float(x0)
    iterations = 0
    last_fx = f(x)

    # Check the trivial case first: the guess is already a root.
    if abs(last_fx) < tol:
        return NewtonResult(
            root=x,
            iterations=0,
            converged=True,
            failed=False,
            final_x=x,
            final_fx=last_fx,
            message="converged at initial guess",
        )

    for _ in range(maxiter):
        if fprime is not None:
            fp = fprime(x)
        else:
            h = dx * max(1.0, abs(x))
            fp = _central_difference(f, x, h)

        if fp == 0.0 or not math.isfinite(fp):
            return NewtonResult(
                root=None,
                iterations=iterations,
                converged=False,
                failed=True,
                final_x=x,
                final_fx=last_fx,
                message="derivative vanished or non-finite",
            )

        x_new = x - last_fx / fp
        iterations += 1

        if not math.isfinite(x_new):
            return NewtonResult(
                root=None,
                iterations=iterations,
                converged=False,
                failed=True,
                final_x=x,
                final_fx=last_fx,
                message="iterate diverged to non-finite",
            )

        try:
            last_fx = f(x_new)
        except (ValueError, OverflowError):
            return NewtonResult(
                root=None,
                iterations=iterations,
                converged=False,
                failed=True,
                final_x=x,
                final_fx=last_fx,
                message="function evaluation failed",
            )

        if not math.isfinite(last_fx):
            return NewtonResult(
                root=None,
                iterations=iterations,
                converged=False,
                failed=True,
                final_x=x_new,
                final_fx=last_fx,
                message="function value non-finite",
            )

        x = x_new

        if abs(last_fx) < tol:
            return NewtonResult(
                root=x,
                iterations=iterations,
                converged=True,
                failed=False,
                final_x=x,
                final_fx=last_fx,
                message="converged",
            )

    return NewtonResult(
        root=None,
        iterations=iterations,
        converged=False,
        failed=True,
        final_x=x,
        final_fx=last_fx,
        message="maxiter reached without convergence",
    )
