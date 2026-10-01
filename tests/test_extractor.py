import unittest

from eval_harness.extractors.math import CompositeMathExtractor


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.extractor = CompositeMathExtractor()

    def test_last_balanced_boxed(self):
        result = self.extractor.extract(r"first \boxed{1}; final \boxed{\frac{3}{4}}")
        self.assertEqual(result.value, r"\frac{3}{4}")
        self.assertEqual(result.method, "last_boxed")

    def test_final_answer(self):
        self.assertEqual(self.extractor.extract("work\nFinal Answer: 42").value, "42")

    def test_missing_output_is_unknown(self):
        self.assertIsNone(self.extractor.extract(None).value)


if __name__ == "__main__":
    unittest.main()

