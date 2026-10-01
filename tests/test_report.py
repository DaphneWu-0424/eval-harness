import unittest

from eval_harness.reports.generate import render_text_report


class ReportTests(unittest.TestCase):
    def test_length_is_not_formatted_as_percentage(self):
        report = render_text_report(
            "x",
            {
                "accuracy": 0.5,
                "length": {"mean": 12.5},
                "counts": {"samples": 1},
            },
        )
        self.assertIn("50.00%", report)
        self.assertIn("12.50", report)
        self.assertNotIn("1,250.00%", report)


if __name__ == "__main__":
    unittest.main()
