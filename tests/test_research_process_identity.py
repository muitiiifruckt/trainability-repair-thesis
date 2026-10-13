import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.research_supervisor import quarantine_reused_leases, quarantine_reused_leases_locked, research_owner_alive


class ResearchProcessIdentityTest(unittest.TestCase):
    def check(self, identity, recorded=100., is_alive=True):
        with patch("scripts.research_supervisor.alive", return_value=is_alive), \
                patch("scripts.research_supervisor.process_identity", return_value=identity):
            return research_owner_alive(2400, recorded)

    def test_reused_non_python_pid_cannot_own_old_worker_lease(self):
        self.assertFalse(self.check({"image_name": "chatgpt.exe", "created_at": 200.}))

    def test_existing_python_worker_remains_owned(self):
        self.assertTrue(self.check({"image_name": "python.exe", "created_at": 90.}))

    def test_python_identity_clock_skew_remains_conservative(self):
        self.assertTrue(self.check({"image_name": "python.exe", "created_at": 200.}))

    def test_unavailable_identity_or_record_never_discards_live_owner(self):
        self.assertTrue(self.check(None))
        self.assertTrue(self.check({"image_name": "chatgpt.exe", "created_at": 200.}, recorded=None))

    def test_dead_process_is_not_owned(self):
        self.assertFalse(self.check(None, is_alive=False))

    def test_small_creation_time_disagreement_is_preserved(self):
        self.assertTrue(self.check({"image_name": "chatgpt.exe", "created_at": 101.}))

    def fixture(self, directory):
        folder = Path(directory) / "campaign"
        lease = folder / "workers/branch/example/worker_lease.json"
        lease.parent.mkdir(parents=True)
        payload = json.dumps({"pid": 2400, "started_at": 100., "task_id": "unchanged"}).encode()
        lease.write_bytes(payload)
        (folder / "worker.lock").write_bytes(b"0")
        (folder / "manifest.json").write_text(json.dumps({"worker_pid": None}), encoding="utf-8")
        scratch = lease.parent / "branch_resume.json"
        scratch.write_bytes(b"preserved")
        return folder, lease, payload, scratch

    def test_quarantine_preserves_lease_bytes_and_never_changes_resume_data(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, lease, payload, scratch = self.fixture(directory)
            with patch("scripts.research_supervisor.ROOT", Path(directory)), \
                    patch("scripts.research_supervisor.alive", side_effect=lambda pid: pid == 2400), \
                    patch("scripts.research_supervisor.process_identity", return_value={"image_name": "chatgpt.exe", "created_at": 200.}):
                records = quarantine_reused_leases(folder)
            self.assertEqual(len(records), 1)
            self.assertFalse(lease.exists())
            self.assertEqual((folder / records[0]["archive"]).read_bytes(), payload)
            self.assertEqual(scratch.read_bytes(), b"preserved")

    def test_healthy_python_lease_is_never_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, lease, payload, _ = self.fixture(directory)
            with patch("scripts.research_supervisor.ROOT", Path(directory)), \
                    patch("scripts.research_supervisor.alive", side_effect=lambda pid: pid == 2400), \
                    patch("scripts.research_supervisor.process_identity", return_value={"image_name": "python.exe", "created_at": 90.}):
                with self.assertRaises(RuntimeError):
                    quarantine_reused_leases(folder)
            self.assertEqual(lease.read_bytes(), payload)

    def test_new_campaign_without_leases_needs_no_preexisting_worker_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "new"
            folder.mkdir()
            with patch("scripts.research_supervisor.ROOT", Path(directory)):
                self.assertEqual(quarantine_reused_leases(folder), [])

    def test_standalone_wrapper_can_quarantine_under_its_existing_lock(self):
        import msvcrt
        with tempfile.TemporaryDirectory() as directory:
            folder, lease, payload, _ = self.fixture(directory)
            with (folder / "worker.lock").open("r+b") as lock:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    with patch("scripts.research_supervisor.ROOT", Path(directory)), \
                            patch("scripts.research_supervisor.alive", side_effect=lambda pid: pid == 2400), \
                            patch("scripts.research_supervisor.process_identity", return_value={"image_name": "chatgpt.exe", "created_at": 200.}):
                        records = quarantine_reused_leases_locked(folder)
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            self.assertFalse(lease.exists())
            self.assertEqual((folder / records[0]["archive"]).read_bytes(), payload)

    def test_malformed_lease_blocks_recovery_without_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, lease, _, _ = self.fixture(directory)
            lease.write_bytes(b"invalid")
            with patch("scripts.research_supervisor.ROOT", Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                    quarantine_reused_leases(folder)
            self.assertEqual(lease.read_bytes(), b"invalid")


if __name__ == "__main__":
    unittest.main()
