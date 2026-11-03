# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Read synthetic official JAX console logs without starting a web server."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "loss_server.py"
SPEC = importlib.util.spec_from_file_location("pi05_loss_server", MODULE_PATH)
LOSS_SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOSS_SERVER)


class OpenpiLossLogTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory(prefix="pi05-loss-test-") as directory:
            path = Path(directory) / "console.log"
            path.write_text(text)
            return LOSS_SERVER.parse_log(str(path))

    def test_official_jax_loss_and_gradient_with_logger_prefix(self):
        data = self.parse(
            "2026-01-01 10:00:00 [INFO] Step 0: loss=0.825, grad_norm=12.5, param_norm=93.2\n"
            "Step 10: loss=0.42, grad_norm=8.25, param_norm=93.4\n"
        )
        self.assertEqual(data["steps"], [0, 10])
        self.assertEqual(data["loss"], [0.825, 0.42])
        self.assertEqual(data["grad_steps"], [0, 10])
        self.assertEqual(data["grad_norm"], [12.5, 8.25])
        self.assertEqual(data["lr"], [])
        self.assertEqual(data["vla_loss"], [])

    def test_tqdm_total_and_carriage_returns_around_metric_lines(self):
        data = self.parse(
            "\x1b[32m  0%|          | 0/20000 [00:00<?, ?it/s]\x1b[0m\r"
            "Step 0: loss=0.825, grad_norm=12.5\n"
            " 25%|##5       | 5001/20000 [1:00:00<3:00:00, 1.39it/s]\r"
            "Step 5000: loss=0.35, grad_norm=7.125\n"
        )
        self.assertEqual(data["total_steps"], 20000)
        self.assertEqual(data["steps"], [0, 5000])
        self.assertEqual(data["grad_steps"], [0, 5000])
        self.assertEqual(data["loss"], [0.825, 0.35])

    def test_repeated_step_replaces_both_metrics_without_duplicate_points(self):
        data = self.parse(
            "Step 10: loss=0.5, grad_norm=9.5\n"
            "  1%|          | 11/1000 [00:03<04:00, 4.0it/s]\r"
            "Step 10: loss=0.4, grad_norm=8.5\n"
            "Step 20: loss=0.3, grad_norm=7.5\n"
            "Step 20: loss=0.25, grad_norm=7.0\n"
        )
        self.assertEqual(data["total_steps"], 1000)
        self.assertEqual(data["steps"], [10, 20])
        self.assertEqual(data["loss"], [0.4, 0.25])
        self.assertEqual(data["grad_steps"], [10, 20])
        self.assertEqual(data["grad_norm"], [8.5, 7.0])

    def test_scientific_notation_and_metric_order(self):
        data = self.parse(
            "Step 1: grad_norm=2E+1, param_norm=9.1e2, loss=+1.25e-3\n"
            "Step 2: loss=2.5E-04, grad_norm=+9.75e+00"
        )
        self.assertEqual(data["steps"], [1, 2])
        self.assertAlmostEqual(data["loss"][0], 0.00125)
        self.assertAlmostEqual(data["loss"][1], 0.00025)
        self.assertEqual(data["grad_norm"], [20.0, 9.75])

    def test_partial_metric_updates_keep_each_curve_aligned(self):
        data = self.parse(
            "Step 10: loss=0.4\n"
            "Step 10: grad_norm=8.5\n"
            "Step 20: loss=0.3, grad_norm=7.5\n"
        )
        self.assertEqual(data["steps"], [10, 20])
        self.assertEqual(data["grad_steps"], [10, 20])
        self.assertEqual(data["loss"], [0.4, 0.3])
        self.assertEqual(data["grad_norm"], [8.5, 7.5])

    def test_missing_and_not_yet_started_logs_return_empty_curves(self):
        with tempfile.TemporaryDirectory(prefix="pi05-loss-test-") as directory:
            path = str(Path(directory) / "not_started.log")
            data = LOSS_SERVER.parse_log(path)
            self.assertEqual(data["source"], path)
            self.assertEqual(data["total_steps"], 0)
            self.assertEqual(data["steps"], [])
            self.assertEqual(data["grad_norm"], [])
        data = self.parse("Loading checkpoint...\nStarting dataset workers...\n")
        self.assertEqual(data["steps"], [])
        self.assertEqual(data["grad_steps"], [])


if __name__ == "__main__":
    unittest.main()
