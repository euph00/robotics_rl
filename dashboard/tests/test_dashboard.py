"""Behavior checks for concurrent logging and the read-only server boundary."""
import json
import gzip
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dashboard.server import DashboardServer, get_runs, read_metrics, read_records, safe_artifact, safe_report, render_report
from pnr.telemetry import RunLogger, update_project


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runs"
        self.root.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_live_jsonl_skips_partial_line_and_preserves_valid_records(self):
        path = self.root / "metrics.jsonl"
        path.write_text('{"transitions": 10, "success_rate": 0.5}\n{"transitions":', encoding="utf-8")
        self.assertEqual(read_records(path), [{"transitions": 10, "success_rate": 0.5}])
        with path.open("a") as handle:
            handle.write(' 20, "success_rate": 1}\n')
        self.assertEqual(len(read_records(path)), 2)

    def test_csv_and_nonfinite_values(self):
        (self.root / "metrics.csv").write_text("transitions,success_rate,scope\n10,0.5,skill\n20,nan,skill\n")
        records = read_metrics(self.root)
        self.assertEqual(records[0]["success_rate"], 0.5)
        self.assertIsNone(records[1]["success_rate"])
        self.assertEqual(records[0]["scope"], "skill")

    def test_logger_round_trip_and_duplicate_guard(self):
        run = RunLogger("pilot-1", root=self.root, kind="train", skill="carry",
                        inherited_transitions=40, accounting={"total_source_action_calls_known": 120})
        run.metric(100, success_rate=0.2, episodes=5, scope="skill",
                   accounting={"total_source_action_calls_known": 300})
        run.event("Physical pickup verified")
        run.finish("complete", "Pilot completed")
        runs, events = get_runs(self.root)
        self.assertEqual(runs[0]["transitions"], 100)
        self.assertEqual(runs[0]["inherited_transitions"], 40)
        self.assertEqual(runs[0]["metrics"]["accounting"]["total_source_action_calls_known"], 300)
        self.assertEqual(runs[0]["accounting"]["total_source_action_calls_known"], 120)
        self.assertEqual(runs[0]["status"], "complete")
        self.assertEqual(runs[0]["series"][0]["scope"], "skill")
        self.assertNotIn("accounting", runs[0]["series"][0])
        self.assertEqual(len(events), 2)
        with self.assertRaises(FileExistsError):
            RunLogger("pilot-1", root=self.root)
        resumed = RunLogger("pilot-1", root=self.root, resume=True, status="running")
        self.assertEqual(resumed.state["transitions"], 100)
        with self.assertRaises(ValueError):
            RunLogger("../escape", root=self.root)

    def test_project_update_preserves_prior_fields(self):
        update_project(self.root, phase="reference", summary="Proving physical grasp")
        state = update_project(self.root, summary="Lift verified")
        self.assertEqual(state["phase"], "reference")
        self.assertEqual(len(state["milestones"]), 5)

    def test_artifact_directory_cannot_alias_run_root(self):
        run = RunLogger("pilot", root=self.root)
        (run.directory / "artifacts").symlink_to(run.directory, target_is_directory=True)
        self.assertIsNone(safe_artifact(self.root, "pilot", "artifacts/status.json"))

    def test_published_report_escapes_content_and_links_only_allowed_files(self):
        artifacts = self.root / "pilot" / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "matrix.png").write_bytes(b"image fixture")
        report = self.root / "qa.md"
        report.write_text("# Test report\n\n**78/100** complete.\n\n"
                          "| Metric | Count |\n|---|---:|\n| Inclusive |63/100|\n\n"
                          "[Matrix](../runs/pilot/artifacts/matrix.png)\n\n"
                          "[Unsafe](javascript:alert(1)) <script>alert(1)</script>\n")
        with patch.dict("dashboard.server.REPORTS", {"qa.md": report}):
            self.assertEqual(safe_report("qa.md"), report)
            self.assertIsNone(safe_report("../qa.md"))
            rendered = render_report(report, self.root).decode()
            self.assertIn("<strong>78/100</strong>", rendered)
            self.assertIn("<td>63/100</td>", rendered)
            self.assertIn('href="/artifacts/pilot/artifacts/matrix.png"', rendered)
            self.assertNotIn("<script>", rendered)
            self.assertNotIn('href="javascript:', rendered)
            alias = self.root / "alias.md"
            alias.symlink_to(report)
            with patch.dict("dashboard.server.REPORTS", {"alias.md": alias}):
                self.assertIsNone(safe_report("alias.md"))

    def test_server_api_static_media_ranges_and_traversal(self):
        run = RunLogger("pilot", root=self.root, kind="reference")
        artifacts = run.directory / "artifacts"
        artifacts.mkdir()
        video = artifacts / "rollout.mp4"
        video.write_bytes(b"0123456789")
        run.artifact("artifacts/rollout.mp4", "Reference", "video")
        private = self.root / "private.json"
        private.write_text('{"secret":"must-not-be-served"}')
        (artifacts / "escape.json").symlink_to(private)
        self.assertIsNone(safe_artifact(self.root, "pilot", "artifacts/../status.json"))
        self.assertIsNone(safe_artifact(self.root, "pilot", "artifacts/escape.json"))
        server = DashboardServer(("127.0.0.1", 0), self.root)
        server.resources.cached = {"logical_cpus": 4}
        server.resources.timestamp = __import__("time").monotonic()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:%s" % server.server_port
        try:
            with urlopen(base + "/api/state") as response:
                body = json.load(response)
                self.assertEqual(body["runs"][0]["id"], "pilot")
                self.assertNotIn("pid", body["runs"][0])
                self.assertIn("Content-Security-Policy", response.headers)
            request = Request(base + "/api/state", headers={"Accept-Encoding": "gzip, deflate"})
            with urlopen(request) as response:
                self.assertEqual(response.headers["Content-Encoding"], "gzip")
                self.assertEqual(response.headers["Vary"], "Accept-Encoding")
                decoded = json.loads(gzip.decompress(response.read()))
                self.assertEqual(decoded["runs"], body["runs"])
            with urlopen(base + "/") as response:
                self.assertIn(b"Evaluation progress", response.read())
            report = self.root / "report.md"
            report.write_text("# Test result\n\nTarget not met.\n")
            with patch.dict("dashboard.server.REPORTS", {"fixture.md": report}):
                with urlopen(base + "/reports/fixture.md") as response:
                    self.assertEqual(response.headers.get_content_type(), "text/html")
                    self.assertIn(b"<h1>Test result</h1>", response.read())
                for blocked in ("/reports/../fixture.md", "/reports/private.json"):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + blocked)
                    self.assertEqual(error.exception.code, 404)
            request = Request(base + "/artifacts/pilot/artifacts/rollout.mp4", headers={"Range": "bytes=2-5"})
            with urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b"2345")
                self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            for path in ("/private.json", "/artifacts/pilot/artifacts/%2e%2e/status.json", "/artifacts/pilot/artifacts/escape.json"):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + path)
                self.assertEqual(error.exception.code, 404)
            request = Request(base + "/artifacts/pilot/artifacts/rollout.mp4", headers={"Range": "bytes=20-30"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 416)
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + "/api/state", data=b"{}", method="POST"))
            self.assertEqual(error.exception.code, 501)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
