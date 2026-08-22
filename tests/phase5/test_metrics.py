import unittest

from scripts.phase5.metrics import compare_field, score_report


def make_field(
    field="total_assets",
    status="found",
    value=100000,
    unit_multiplier=1000,
    scope="individual",
    source_year=2024,
    page=10,
):
    return {
        "field": field,
        "status": status,
        "value": value,
        "unit_multiplier": unit_multiplier,
        "scope": scope,
        "source_year": source_year,
        "page": page,
    }


class TestPhase5Metrics(unittest.TestCase):

    def test_value_inside_tolerance(self):
        gold = make_field(value=100000)
        pred = make_field(value=100050)

        verdict = compare_field(gold, pred)

        self.assertTrue(verdict["correct"])

    def test_value_outside_tolerance(self):
        gold = make_field(value=100000)
        pred = make_field(value=100200)

        verdict = compare_field(gold, pred)

        self.assertFalse(verdict["correct"])
        self.assertIn(
            "value_outside_tolerance",
            verdict["failure_reasons"],
        )

    def test_wrong_unit(self):
        gold = make_field(unit_multiplier=1000)
        pred = make_field(unit_multiplier=1)

        verdict = compare_field(gold, pred)

        self.assertFalse(verdict["correct"])
        self.assertIn("unit_mismatch", verdict["failure_reasons"])

    def test_wrong_scope(self):
        gold = make_field(scope="individual")
        pred = make_field(scope="consolidated")

        verdict = compare_field(gold, pred)

        self.assertFalse(verdict["correct"])
        self.assertIn("scope_mismatch", verdict["failure_reasons"])

    def test_wrong_source_year(self):
        gold = make_field(source_year=2024)
        pred = make_field(source_year=2023)

        verdict = compare_field(gold, pred)

        self.assertFalse(verdict["correct"])
        self.assertIn(
            "source_year_mismatch",
            verdict["failure_reasons"],
        )

    def test_not_found_matches(self):
        gold = make_field(
            status="not_found",
            value=None,
            unit_multiplier=None,
            page=None,
        )
        pred = make_field(
            status="not_found",
            value=None,
            unit_multiplier=None,
            page=None,
        )

        verdict = compare_field(gold, pred)

        self.assertTrue(verdict["correct"])

    def test_ambiguous_matches(self):
        gold = make_field(
            status="ambiguous",
            value=None,
            unit_multiplier=None,
            page=None,
        )
        pred = make_field(
            status="ambiguous",
            value=None,
            unit_multiplier=None,
            page=None,
        )

        verdict = compare_field(gold, pred)

        self.assertTrue(verdict["correct"])

    def test_not_found_vs_ambiguous(self):
        gold = make_field(
            status="not_found",
            value=None,
            unit_multiplier=None,
            page=None,
        )
        pred = make_field(
            status="ambiguous",
            value=None,
            unit_multiplier=None,
            page=None,
        )

        verdict = compare_field(gold, pred)

        self.assertFalse(verdict["correct"])
        self.assertIn(
            "status_mismatch",
            verdict["failure_reasons"],
        )

    def test_document_mismatch_raises_error(self):
        fields = [
            make_field(field="total_assets"),
            make_field(field="total_equity"),
            make_field(field="net_banking_income"),
            make_field(field="operating_income"),
            make_field(field="net_income"),
            make_field(field="customer_deposits"),
            make_field(field="net_customer_loans"),
        ]

        gold = {
            "doc_id": "UIB_2024",
            "target_year": 2024,
            "target_scope": "individual",
            "currency": "TND",
            "fields": fields,
        }

        pred = {
            **gold,
            "doc_id": "STB_2024",
        }

        with self.assertRaises(ValueError):
            score_report(gold, pred)


if __name__ == "__main__":
    unittest.main()