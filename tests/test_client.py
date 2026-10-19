# SPDX-License-Identifier: GPL-3.0-or-later
import json
import sys
import threading
import types
import unittest
from unittest import mock

try:
    import websocket  # noqa: F401
except ImportError:
    sys.modules["websocket"] = types.ModuleType("websocket")

from ai_image_edit.core.errors import GenerationCancelled, GenerationError
from ai_image_edit.comfy import client


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


class _Timeout(Exception):
    pass


class FakeSocket:
    """Plays back (seconds, message) pairs: each advances the fake clock, then is received.
    A message of None is a receive that times out after the socket's timeout."""

    def __init__(self, clock, script):
        self.clock, self.script, self.timeout = clock, list(script), None

    def connect(self, url, timeout=None):
        pass

    def settimeout(self, seconds):
        self.timeout = seconds

    def recv(self):
        if not self.script:
            self.clock[0] += self.timeout
            raise _Timeout()
        seconds, message = self.script.pop(0)
        if callable(message):
            message = message()
        if message is None:
            self.clock[0] += self.timeout
            raise _Timeout()
        self.clock[0] += seconds
        return message if isinstance(message, (str, bytes)) else json.dumps(message)

    def close(self):
        pass


def msg(kind, prompt_id="p", **data):
    return {"type": kind, "data": {"prompt_id": prompt_id, **data}}


class WaitTest(unittest.TestCase):
    def run_script(self, script, stall=600, max_time=1800, cancel=None, prompt_node=None):
        clock = [1000.0]
        events = []
        socket = FakeSocket(clock, script)
        fake_ws = types.SimpleNamespace(
            WebSocket=lambda: socket, WebSocketTimeoutException=_Timeout, WebSocketException=type("WSE", (Exception,), {}),
        )
        with mock.patch.object(client, "websocket", fake_ws), \
                mock.patch.object(client, "_queue_prompt", return_value="p"), \
                mock.patch.object(client.time, "time", lambda: clock[0]), \
                mock.patch.object(client, "STALL_TIMEOUT_SECONDS", stall), \
                mock.patch.object(client, "MAX_GENERATION_SECONDS", max_time):
            result = client.submit_workflow_and_wait(
                {}, "c", sampler_node="6", on_progress=events.append, cancel=cancel, prompt_node=prompt_node,
            )
        return result, [(e.stage.value, e.step, e.total) for e in events]

    def test_reports_the_stages_of_this_prompt_only(self):
        result, events = self.run_script([
            (0, msg("execution_start", "other")),
            (0, {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}}),
            (0, msg("execution_start")),
            (0, msg("executing", node="3")),
            (0, msg("progress", node="12", value=1, max=4)),  # not the sampler
            (0, msg("executing", node="6")),
            (0, msg("progress", node="6", value=1, max=2)),
            (0, msg("progress", "other", node="6", value=7, max=9)),
            (0, b"preview"),
            (0, msg("progress", node="6", value=2, max=2)),
            (0, msg("executing", node="8")),
            (0, msg("executing", node=None)),
        ])
        self.assertEqual(result, "p")
        self.assertEqual(events, [
            ("preparing", None, None), ("sampling", 1, 2), ("sampling", 2, 2), ("finishing", None, None),
        ])

    def test_the_prompt_enhancer_reports_its_tokens(self):
        _, events = self.run_script([
            (0, msg("execution_start")),
            (0, msg("executing", node="41")),
            (0, msg("progress", node="41", value=10, max=4096)),
            (0, msg("progress", node="41", value=20, max=4096)),
            (0, msg("executing", node="5")),
            (0, msg("executing", node="6")),
            (0, msg("progress", node="6", value=1, max=2)),
            (0, msg("executing", node=None)),
        ], prompt_node="41")
        self.assertEqual(events, [
            ("preparing", None, None), ("enhancing_prompt", None, None), ("enhancing_prompt", 10, None),
            ("enhancing_prompt", 20, None), ("preparing", None, None), ("sampling", 1, 2),
        ])

    def test_the_output_text_is_read_from_the_history(self):
        history = {"p": {"outputs": {"43": {"text": ["a better prompt"]}, "8": {"images": []}}}}
        response = mock.Mock(**{"json.return_value": history})
        with mock.patch.object(client.requests, "get", return_value=response):
            self.assertEqual(client.fetch_output_text("p", "43"), "a better prompt")
            self.assertIsNone(client.fetch_output_text("p", "8"))
            self.assertIsNone(client.fetch_output_text("other", "43"))

    def test_a_job_behind_another_is_queued(self):
        _, events = self.run_script([
            (0, {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 2}}}}),
            (0, msg("execution_start")),
            (0, msg("executing", node=None)),
        ])
        self.assertEqual(events, [("preparing", None, None), ("queued", None, None), ("preparing", None, None)])

    def test_an_execution_error_is_raised(self):
        with self.assertRaisesRegex(GenerationError, "out of memory"):
            self.run_script([(0, msg("execution_start")), (0, msg("execution_error", exception_message="out of memory"))])

    def test_an_interruption_is_raised(self):
        with self.assertRaisesRegex(GenerationError, "interrupted"):
            self.run_script([(0, msg("execution_start")), (0, msg("execution_interrupted"))])

    def test_a_running_job_that_reports_nothing_stalls(self):
        with self.assertRaisesRegex(GenerationError, "no progress for 600s"):
            self.run_script([(0, msg("execution_start")), (500, msg("progress", node="6", value=1, max=9)), (0, None)])

    def test_progress_keeps_a_long_job_alive(self):
        steps = [(300, msg("progress", node="6", value=i, max=5)) for i in range(1, 6)]  # 1500 s in all
        result, events = self.run_script([(0, msg("execution_start")), *steps, (0, msg("executing", node=None))])
        self.assertEqual(result, "p")
        self.assertEqual(events[-1], ("sampling", 5, 5))

    def test_the_wait_in_the_queue_does_not_stall(self):
        others = [(400, msg("progress", "other", node="6", value=i, max=9)) for i in range(3)]
        result, _ = self.run_script([*others, (0, msg("execution_start")), (0, msg("executing", node=None))])
        self.assertEqual(result, "p")

    def test_cancelling_stops_only_this_prompt(self):
        cancel = threading.Event()

        def cancel_then_progress():
            cancel.set()
            return msg("progress", node="6", value=2, max=9)

        posts = []
        with mock.patch.object(client.requests, "post", side_effect=lambda url, json, timeout: posts.append((url, json))):
            with self.assertRaises(GenerationCancelled):
                self.run_script([(0, msg("execution_start")), (0, cancel_then_progress)], cancel=cancel)
        self.assertEqual(posts, [
            (f"http://{client.SERVER_ADDRESS}/queue", {"delete": ["p"]}),
            (f"http://{client.SERVER_ADDRESS}/interrupt", {"prompt_id": "p"}),
        ])

    def test_a_cancel_is_noticed_while_nothing_arrives(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(client.requests, "post"):
            with self.assertRaises(GenerationCancelled):
                self.run_script([(0, None)], cancel=cancel)

    def test_the_whole_job_has_a_limit(self):
        steps = [(500, msg("progress", node="6", value=i, max=9)) for i in range(1, 9)]
        with self.assertRaisesRegex(GenerationError, "longer than 1800s"):
            self.run_script([(0, msg("execution_start")), *steps])


if __name__ == "__main__":
    unittest.main()
