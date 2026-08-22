import unittest
from unittest.mock import patch

from scripts.phase4.generation_policy import (
    generate_with_one_repair,
)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value"],
    "properties": {
        "value": {
            "type": "integer",
        }
    },
}


def fake_result(
    text,
    input_tokens,
    output_tokens,
):
    return {
        "text": text,
        "model_id": "fake-model",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_duration_ns": 10,
        "load_duration_ns": 1,
        "prompt_eval_duration_ns": 2,
        "eval_duration_ns": 7,
        "done_reason": "stop",
    }


class TestGenerationPolicy(
    unittest.TestCase
):

    @patch(
        "scripts.phase4."
        "generation_policy.generate"
    )
    def test_valid_first_pass_no_retry(
        self,
        mock_generate,
    ):
        mock_generate.return_value = (
            fake_result(
                '{"value": 10}',
                100,
                20,
            )
        )

        result = generate_with_one_repair(
            model_id="fake-model",
            messages=[
                {
                    "role": "user",
                    "content": "PAGE_SECRET",
                }
            ],
            schema=SCHEMA,
            settings={},
        )

        self.assertFalse(
            result["retry_used"]
        )

        self.assertTrue(
            result["validation"]["valid"]
        )

        self.assertEqual(
            mock_generate.call_count,
            1,
        )

    @patch(
        "scripts.phase4."
        "generation_policy.generate"
    )
    def test_invalid_then_valid_retry(
        self,
        mock_generate,
    ):
        mock_generate.side_effect = [
            fake_result(
                '{"value": "bad"}',
                100,
                20,
            ),
            fake_result(
                '{"value": 10}',
                30,
                10,
            ),
        ]

        result = generate_with_one_repair(
            model_id="fake-model",
            messages=[
                {
                    "role": "user",
                    "content": "PAGE_SECRET",
                }
            ],
            schema=SCHEMA,
            settings={},
        )

        self.assertTrue(
            result["retry_used"]
        )

        self.assertFalse(
            result[
                "first_validation"
            ]["valid"]
        )

        self.assertTrue(
            result["validation"]["valid"]
        )

        self.assertEqual(
            mock_generate.call_count,
            2,
        )

        final_result = result[
            "generation_result"
        ]

        self.assertEqual(
            final_result[
                "input_tokens"
            ],
            130,
        )

        self.assertEqual(
            final_result[
                "output_tokens"
            ],
            30,
        )

        second_messages = (
            mock_generate
            .call_args_list[1]
            .kwargs["messages"]
        )

        retry_text = " ".join(
            message["content"]
            for message in second_messages
        )

        # The original financial-page context
        # must NOT be resent in the repair call.
        self.assertNotIn(
            "PAGE_SECRET",
            retry_text,
        )

    @patch(
        "scripts.phase4."
        "generation_policy.generate"
    )
    def test_retry_can_still_fail(
        self,
        mock_generate,
    ):
        mock_generate.side_effect = [
            fake_result(
                '{"value": "bad"}',
                100,
                20,
            ),
            fake_result(
                '{"value": "still bad"}',
                30,
                10,
            ),
        ]

        result = generate_with_one_repair(
            model_id="fake-model",
            messages=[],
            schema=SCHEMA,
            settings={},
        )

        self.assertTrue(
            result["retry_used"]
        )

        self.assertFalse(
            result["validation"]["valid"]
        )

        self.assertEqual(
            mock_generate.call_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
