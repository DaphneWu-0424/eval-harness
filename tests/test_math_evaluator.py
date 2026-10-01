import unittest

from eval_harness.evaluators.math import MathEvaluator, compare_math_answers


class MathComparisonTests(unittest.TestCase):
    def test_fraction_decimal_equivalence(self):
        self.assertTrue(compare_math_answers("455/4", "113.75"))

    def test_radical_equivalence(self):
        self.assertTrue(compare_math_answers("sqrt(6)/2", "sqrt(3/2)"))

    def test_radical_sign_is_preserved(self):
        self.assertFalse(compare_math_answers("sqrt(6)/-2", "sqrt(3/2)"))

    def test_dfrac_and_frac_equivalence(self):
        self.assertTrue(compare_math_answers(r"\dfrac{13}{2}", r"\frac{13}{2}"))

    def test_latex_wrappers_and_spacing(self):
        self.assertTrue(compare_math_answers(r"\(a = 288, b = 48\)", "a=288,b=48"))

    def test_different_assignments_are_incorrect(self):
        self.assertFalse(compare_math_answers("a=36,b=24", r"\(a=288,b=48\)"))

    def test_assignment_order_is_irrelevant(self):
        self.assertTrue(compare_math_answers("b=48,a=288", "a=288,b=48"))

    def test_text_no_solution(self):
        self.assertTrue(compare_math_answers(r"\text{No solution}", "no solution"))

    def test_incorrect_numeric(self):
        self.assertFalse(compare_math_answers("2", "3"))

    def test_unparseable_is_unknown(self):
        self.assertIsNone(compare_math_answers("not an expression", "1"))

    def test_configured_extractor_is_created_from_registry(self):
        evaluator = MathEvaluator({"type": "math", "extractor": {"type": "composite_math"}})
        self.assertEqual(type(evaluator.extractor).__name__, "CompositeMathExtractor")

    def test_unknown_extractor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown extractor type"):
            MathEvaluator({"type": "math", "extractor": {"type": "missing"}})


if __name__ == "__main__":
    unittest.main()
