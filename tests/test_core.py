import math
import unittest

from newton_s_method_root_finding import find_root, NewtonResult


class FindRootTests(unittest.TestCase):
    def test_polynomial_with_analytic_derivative(self):
        # x^2 - 2, root at sqrt(2). Classic textbook case.
        r = find_root(lambda x: x * x - 2.0, 1.0, fprime=lambda x: 2.0 * x)
        self.assertTrue(r.converged)
        self.assertFalse(r.failed)
        self.assertAlmostEqual(r.root, math.sqrt(2.0), places=10)
        self.assertLess(abs(r.final_fx), 1e-10)
        self.assertGreater(r.iterations, 0)

    def test_numerical_derivative_matches_analytic(self):
        # For a smooth function the central-difference path should land at the
        # same root as the analytic-derivative path (to tolerance).
        f = lambda x: math.sin(x)
        r_an = find_root(f, 3.0, fprime=math.cos)
        r_num = find_root(f, 3.0)
        self.assertTrue(r_an.converged)
        self.assertTrue(r_num.converged)
        self.assertAlmostEqual(r_an.root, r_num.root, places=6)
        self.assertAlmostEqual(r_num.root, math.pi, places=6)

    def test_initial_guess_is_already_root(self):
        r = find_root(lambda x: x * x - 4.0, 2.0)
        self.assertTrue(r.converged)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(r.root, 2.0)

    def test_negative_root(self):
        # Start on the left branch of x^2-4 to land at -2 not +2.
        r = find_root(lambda x: x * x - 4.0, -3.0, fprime=lambda x: 2.0 * x)
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.root, -2.0, places=10)

    def test_derivative_vanishing_reports_failure(self):
        # f(x) = x^2 + 1 has no real root and derivative at 0 is 0, so the
        # solver should refuse to divide by zero.
        r = find_root(lambda x: x * x + 1.0, 0.0, fprime=lambda x: 2.0 * x)
        self.assertFalse(r.converged)
        self.assertTrue(r.failed)
        self.assertIsNone(r.root)
        self.assertIn("derivative", r.message)

    def test_maxiter_exhaustion_reports_failure(self):
        # A function with no real root and a strictly positive derivative never
        # converges; the solver should give up cleanly without converging.
        r = find_root(lambda x: math.exp(x) + 1.0, 0.0, fprime=math.exp, maxiter=5)
        self.assertFalse(r.converged)
        self.assertTrue(r.failed)
        self.assertIsNone(r.root)
        self.assertEqual(r.iterations, 3)
        self.assertIn("derivative", r.message)

    def test_non_finite_x0_rejected(self):
        with self.assertRaises(ValueError):
            find_root(lambda x: x, float("nan"))
        with self.assertRaises(ValueError):
            find_root(lambda x: x, float("inf"))

    def test_bad_arguments_rejected(self):
        with self.assertRaises(TypeError):
            find_root("not callable", 1.0)
        with self.assertRaises(TypeError):
            find_root(lambda x: x, 1.0, fprime="no")
        with self.assertRaises(ValueError):
            find_root(lambda x: x, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            find_root(lambda x: x, 1.0, tol=-1e-9)
        with self.assertRaises(ValueError):
            find_root(lambda x: x, 1.0, maxiter=0)
        with self.assertRaises(ValueError):
            find_root(lambda x: x, 1.0, dx=0.0)

    def test_result_attributes_present(self):
        r = find_root(lambda x: x - 1.0, 0.5, fprime=lambda x: 1.0)
        self.assertIsInstance(r, NewtonResult)
        for name in ("root", "iterations", "converged", "failed", "final_x", "final_fx", "message"):
            self.assertTrue(hasattr(r, name), name)
        self.assertEqual(r.final_x, r.root)

    def test_logarithmic_root(self):
        # Root of log(x) - 1 at x = e, with a non-constant derivative.
        r = find_root(lambda x: math.log(x) - 1.0, 2.0, fprime=lambda x: 1.0 / x)
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.root, math.e, places=10)

    def test_large_initial_guess_scales_step(self):
        # Newton's method from a large x0 for sqrt(x)-1 overshoots into the
        # negative domain (the method has no safeguards; see README). The
        # solver should report a clean failure rather than crash.
        r = find_root(lambda x: math.sqrt(x) - 1.0, 1e6)
        self.assertFalse(r.converged)
        self.assertTrue(r.failed)
        self.assertIsNone(r.root)

    def test_returned_object_is_immutable(self):
        r = find_root(lambda x: x, 0.0)
        with self.assertRaises(Exception):
            r.root = 1.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
