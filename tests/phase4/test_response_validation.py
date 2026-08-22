import unittest

from scripts.phase4.response_validation import (
    parse_and_validate,
)


SCHEMA = {
    "type": "object",
    "required": ["fields"],
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["field"],
                "properties": {
                    "field": {
                        "type": "string",
                    }
                },
            },
        }
    },
}


class TestResponseValidation(unittest.TestCase):

    def test_duplicate_and_missing_fields_fail(self):
        raw = """
        {
          "fields": [
            {"field": "total_assets"},
            {"field": "total_equity"},
            {"field": "net_banking_income"},
            {"field": "operating_income"},
            {"field": "total_assets"},
            {"field": "total_equity"},
            {"field": "net_banking_income"}
          ]
        }
        """

        result = parse_and_validate(
            raw,
            SCHEMA,
        )

        self.assertFalse(
            result["valid"]
        )

        self.assertEqual(
            result["error_type"],
            "contract_validation_error",
        )


if __name__ == "__main__":
    unittest.main()