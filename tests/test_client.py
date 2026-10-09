# SPDX-License-Identifier: GPL-3.0-or-later
import sys
import types
import unittest
from unittest import mock

try:
    import websocket  # noqa: F401
except ImportError:
    sys.modules["websocket"] = types.ModuleType("websocket")

from ai_image_edit.core.errors import GenerationError
from ai_image_edit.comfyui import client


def response(body, status=200):
    res = mock.Mock(status_code=status, content=b"png")
    res.json.return_value = body
    return res


class QueuePromptTest(unittest.TestCase):
    def test_returns_the_prompt_id(self):
        with mock.patch.object(client.requests, "post", return_value=response({"prompt_id": "abc"})):
            self.assertEqual(client._queue_prompt({}, "c"), "abc")

    def test_rejection_names_the_reason_and_the_node(self):
        body = {
            "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
            "node_errors": {"6": {"errors": [{"message": "Value not in list", "details": "ckpt_name: 'x'"}]}},
        }
        with mock.patch.object(client.requests, "post", return_value=response(body, 400)):
            with self.assertRaisesRegex(GenerationError, "Prompt outputs failed validation.*node 6: Value not in list"):
                client._queue_prompt({}, "c")

    def test_unreachable_server(self):
        with mock.patch.object(client.requests, "post", side_effect=client.requests.exceptions.ConnectionError("down")):
            with self.assertRaisesRegex(GenerationError, "Error sending prompt"):
                client._queue_prompt({}, "c")


class FetchImageTest(unittest.TestCase):
    def test_query_is_encoded(self):
        history = response({"p": {"outputs": {"9": {"images": [{"filename": "a b.png", "subfolder": "x&y", "type": "output"}]}}}})
        view = response(None)
        with mock.patch.object(client.requests, "get", side_effect=[history, view]) as get, \
                mock.patch.object(client, "WORK_DIR", __import__("pathlib").Path(__import__("tempfile").mkdtemp())):
            path = client.fetch_generated_image("p")
        self.assertIn("filename=a+b.png&subfolder=x%26y&type=output", get.call_args_list[1].args[0])
        self.assertTrue(path.endswith("output_p.png"))


if __name__ == "__main__":
    unittest.main()
