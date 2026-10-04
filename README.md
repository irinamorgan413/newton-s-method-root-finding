# Newton's Method Root Finding

A small, dependency-free scalar root finder using Newton's method, with a
numerical (central-difference) derivative as the fallback when an analytic one
is not supplied.

## Usage

```python
import math
from newton_s_method_root_finding import find_root, NewtonResult

# Analytic derivative supplied:
r = find_root(lambda x: x * x - 2.0, 1.0, fprime=lambda x: 2.0 * x)
print(r.root, r.iterations, r.converged)

# Numerical derivative fallback:
r = find_root(lambda x: math.sin(x), 3.0)
print(r.root)
```

`find_root(f, x0, *, fprime=None, tol=1e-10, maxiter=50, dx=1e-6)` returns a
`NewtonResult` with `root`, `iterations`, `converged`, `failed`, `final_x`,
`final_fx`, and `message`. `root` is `None` when the solver gives up.

## Why this exists

Newton's method is a two-line algorithm when you have the derivative, but in
practice you often don't, and wiring up a numerical derivative by hand is
repetitive. This library packages the method with a sensible central-difference
fallback so callers can drop in any `f(x) -> float` and get a root.

The trade-off is convergence robustness. Newton's method is locally quadratic
but globally fragile: a bad initial guess, a flat spot, or an inflection can
send it spiralling off. This library does **not** implement safeguards like
damped steps or bisection fallback. If your starting point is unreliable, pair
it with a bracketing method.

## The awkward edge

Convergence is decided on `|f(x)| < tol` rather than on step size. That keeps
the API to one tolerance parameter, but it means a function that is flat
near the root (small derivative magnitude) can satisfy the tolerance while
`x` is still a perceptible distance away from the true root. If that matters
for your application, tighten `tol` or check `final_x` against an independent
bound.

If the derivative is exactly zero at an iterate the solver stops with
`failed=True` and `message` mentions the derivative; it does not attempt a
rescue step.
