"""Human-readable reporting must preserve structured scientific records."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from emulator.training.logging import (
    close_logging,
    configure_logging,
    log_batch_progress,
    log_best_checkpoint,
    log_epoch,
    logger,
)


class LoggingTests(unittest.TestCase):
    """Exercise file ownership, reporting semantics, and visible failures."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pact_v2_logging_test_")
        self.directory = Path(self.temporary.name)
        self.console = io.StringIO()
        self.redirect = ExitStack()
        self.redirect.enter_context(redirect_stdout(self.console))
        self.redirect.enter_context(redirect_stderr(self.console))
        close_logging()

    def tearDown(self):
        close_logging()
        self.redirect.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_timestamp_console_and_fresh_persistent_log(self):
        """Early console messages work before a validated directory is available."""
        configure_logging(rank=0)
        logger.info("Indexing synthetic TRAIN")
        self.assertRegex(
            self.console.getvalue(),
            r"^\[\d{4}-\d{2}-\d{2}\|\d{2}:\d{2}:\d{2}\] Indexing synthetic TRAIN\n$",
        )
        self.assertFalse((self.directory / "train.log").exists())

        path = self.directory / "train.log"
        path.write_text("previous unrelated run\n")
        configure_logging(rank=0, output_dir=self.directory)
        logger.info("Starting synthetic experiment")
        contents = path.read_text()
        self.assertNotIn("previous unrelated run", contents)
        self.assertIn("Starting synthetic experiment", contents)
        self.assertRegex(contents, r"\[\d{4}-\d{2}-\d{2}\|\d{2}:\d{2}:\d{2}\]")
        self.assertIn(contents, self.console.getvalue())

    def test_resume_appends_without_truncating_history(self):
        configure_logging(rank=0, output_dir=self.directory)
        logger.info("First session completed")
        close_logging()
        path = self.directory / "train.log"
        previous = path.read_bytes()

        configure_logging(rank=0, output_dir=self.directory, resume=True)
        logger.info("Resumed session completed")
        contents = path.read_bytes()
        self.assertTrue(contents.startswith(previous))
        self.assertIn(b"RESUM", contents[len(previous):])
        self.assertIn(b"Resumed session completed", contents[len(previous):])

    def test_reconfiguration_does_not_duplicate_handlers(self):
        configure_logging(rank=0, output_dir=self.directory)
        configure_logging(rank=0, output_dir=self.directory)
        logger.info("Unique message")
        self.assertEqual(self.console.getvalue().count("Unique message"), 1)
        self.assertEqual((self.directory / "train.log").read_text().count("Unique message"), 1)
        self.assertFalse(logger.propagate)

    def test_nonzero_rank_suppresses_normal_output_but_reports_errors(self):
        """Only rank zero owns the common file; other ranks keep tracebacks."""
        configure_logging(rank=1, output_dir=self.directory)
        logger.info("Must not appear")
        try:
            raise RuntimeError("Rank one failure")
        except RuntimeError:
            logger.exception("Worker failed")
        contents = self.console.getvalue()
        self.assertNotIn("Must not appear", contents)
        self.assertIn("Worker failed", contents)
        self.assertIn("Traceback (most recent call last):", contents)
        self.assertIn("RuntimeError: Rank one failure", contents)
        self.assertFalse((self.directory / "train.log").exists())

    def test_epoch_units_numbering_learning_rate_and_structured_record(self):
        configure_logging(rank=0, output_dir=self.directory)
        row = {
            "epoch": 0,
            "lr": 0.0001,
            "train_global_mse": 0.012345,
            "validation": {"all_rmse": 0.123456, "all_mae": 0.098765},
            "smoke_test": False,
        }
        original = deepcopy(row)
        machine_record = json.dumps(row, allow_nan=False) + "\n"
        path = self.directory / "training_metrics.jsonl"
        path.write_text(machine_record)
        log_epoch(row, epochs=300, elapsed=45.2)
        contents = self.console.getvalue()
        for expected in (
            "Epoch 001/300", "Train GlobalMSE: 0.012345 m²",
            "Val AllRMSE: 0.123456 m", "Val AllMAE: 0.098765 m",
            "LR: 1.00e-04", "Time: 45.2s",
        ):
            self.assertIn(expected, contents)
        self.assertEqual(len(contents.splitlines()), 1)
        self.assertEqual(row, original)
        self.assertEqual(path.read_text(), machine_record)
        self.assertEqual(json.dumps(row, allow_nan=False) + "\n", machine_record)

    def test_epoch_without_validation_does_not_reuse_previous_metrics(self):
        configure_logging(rank=0)
        row = {
            "epoch": 1, "lr": 0.001, "train_global_mse": 0.25,
            "validation": None, "smoke_test": False,
        }
        log_epoch(row, epochs=10, elapsed=1.0)
        contents = self.console.getvalue()
        self.assertIn("Epoch 002/10", contents)
        self.assertNotIn("Val AllRMSE:", contents)
        self.assertNotIn("Val AllMAE:", contents)
        self.assertRegex(contents.lower(), "not scheduled|skipped")

    def test_ddp_batch_progress_labels_local_population_and_missing_supervision(self):
        configure_logging(rank=0)
        log_batch_progress(
            0, 3, 1, 40, None, 2.0, world=2
        )
        contents = self.console.getvalue()
        self.assertIn("Rank 0 local Cumulative GlobalMSE", contents)
        self.assertIn("unavailable (no eligible WW node-hours)", contents)
        self.assertNotIn("nan", contents.lower())
        self.assertNotIn("0.000000 m²", contents)

    def test_best_checkpoint_notification_uses_selected_metric_and_path(self):
        configure_logging(rank=0)
        path = self.directory / "best_overall.pt"
        log_best_checkpoint(24, {"all_rmse": 0.123456, "all_mae": 0.098765}, path)
        contents = self.console.getvalue()
        self.assertIn("[Best overall] Epoch 025", contents)
        self.assertIn("Val AllRMSE: 0.123456 m", contents)
        self.assertIn(str(path), contents)

    def test_exception_traceback_reaches_console_and_log(self):
        configure_logging(rank=0, output_dir=self.directory)
        with self.assertRaisesRegex(ValueError, "Invalid synthetic forcing"):
            try:
                raise ValueError("Invalid synthetic forcing")
            except ValueError:
                logger.exception("Training failed")
                raise
        for contents in (self.console.getvalue(), (self.directory / "train.log").read_text()):
            self.assertIn("Training failed", contents)
            self.assertIn("Traceback (most recent call last):", contents)
            self.assertIn("ValueError: Invalid synthetic forcing", contents)


if __name__ == "__main__":
    unittest.main()
