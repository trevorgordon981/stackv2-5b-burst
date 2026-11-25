import ast
import hashlib
import json
import os
import pickle
import tempfile
import unittest
from unittest import mock

import checkpoint_io


def _pickle_save(payload, fh):
    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path, map_location="cpu"):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class TestCheckpointResumeAndIntegrity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_resume_after_step_1000_starts_at_1001_without_repeating_step(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        payload = checkpoint_io.build_checkpoint_payload(
            model_state={"w": [1.0, 2.0]},
            opt_state={"step": 1000},
            completed_step=1000,
            cfg={"hidden_size": 256},
        )
        checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)

        loaded_path, sd, start_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, map_location="cpu", load_fn=_pickle_load
        )
        self.assertEqual(loaded_path, ckpt_path)
        self.assertEqual(sd["completed_step"], 1000)
        self.assertEqual(sd["next_step"], 1001)
        self.assertEqual(start_step, 1001)

    def test_simulated_training_loop_never_duplicates_saved_step(self):
        executed_steps = []
        save_interval = 5
        total_steps = 12

        def run_segment(stop_after_save=False):
            _, sd, start = checkpoint_io.load_latest_checkpoint(
                self.ckpt_dir, map_location="cpu", load_fn=_pickle_load
            )
            weight = sd["model"]["w"] if sd is not None else 0
            for step in range(start, total_steps):
                weight += 1
                executed_steps.append(step)
                if step > 0 and step % save_interval == 0:
                    path = os.path.join(self.ckpt_dir, f"ckpt_{step:08d}.pt")
                    payload = checkpoint_io.build_checkpoint_payload(
                        model_state={"w": weight},
                        opt_state={"updates": weight},
                        completed_step=step,
                        cfg={"seq_len": 128},
                    )
                    checkpoint_io.save_checkpoint_atomic(payload, path, save_fn=_pickle_save)
                    if stop_after_save:
                        return weight
            return weight

        w_before_restart = run_segment(stop_after_save=True)
        self.assertEqual(executed_steps, [0, 1, 2, 3, 4, 5])
        self.assertEqual(w_before_restart, 6)

        w_final = run_segment(stop_after_save=False)
        self.assertEqual(executed_steps, list(range(total_steps)))
        self.assertEqual(w_final, total_steps)

    def test_legacy_step_only_checkpoint_resumes_at_next_step(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        legacy_payload = {
            "model": {"w": 42},
            "opt": {"lr": 3e-4},
            "step": 1000,
            "cfg": {"hidden_size": 256},
        }
        checkpoint_io.save_checkpoint_atomic(legacy_payload, ckpt_path, save_fn=_pickle_save)

        _, sd, start_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, map_location="cpu", load_fn=_pickle_load
        )
        self.assertEqual(sd["step"], 1000)
        self.assertEqual(start_step, 1001)

    def test_atomic_save_creates_manifest_fsyncs_and_leaves_no_temp_files(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00000200.pt")
        payload = checkpoint_io.build_checkpoint_payload(
            model_state={"layer": [1, 2, 3]},
            opt_state={"m": [0.1, 0.2]},
            completed_step=200,
            cfg={"hidden_size": 64},
        )
        with mock.patch("checkpoint_io.os.fsync", wraps=os.fsync) as fsync_spy, mock.patch(
            "checkpoint_io.os.replace", wraps=os.replace
        ) as replace_spy:
            manifest = checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)

        self.assertGreaterEqual(fsync_spy.call_count, 2)
        self.assertEqual(replace_spy.call_count, 2)

        manifest_path = checkpoint_io.manifest_path_for(ckpt_path)
        self.assertTrue(os.path.isfile(ckpt_path))
        self.assertTrue(os.path.isfile(manifest_path))

        entries = sorted(os.listdir(self.ckpt_dir))
        self.assertEqual(entries, ["ckpt_00000200.pt", "ckpt_00000200.pt.sha256"])

        with open(ckpt_path, "rb") as fh:
            raw = fh.read()
        self.assertEqual(manifest["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(manifest["size_bytes"], len(raw))
        self.assertEqual(manifest["completed_step"], 200)
        self.assertEqual(manifest["next_step"], 201)

        verified = checkpoint_io.verify_checkpoint_integrity(ckpt_path)
        self.assertEqual(verified, manifest)

    def test_tampered_checkpoint_same_size_fails_sha256_verification(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        payload = checkpoint_io.build_checkpoint_payload({"w": b"abcdefgh"}, {"s": 1}, 1000, {})
        checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)

        with open(ckpt_path, "r+b") as fh:
            data = bytearray(fh.read())
            data[-1] ^= 0xFF
            fh.seek(0)
            fh.write(data)

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "SHA-256 mismatch"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, map_location="cpu", load_fn=_pickle_load)

    def test_truncated_checkpoint_fails_integrity_verification(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        payload = checkpoint_io.build_checkpoint_payload({"w": list(range(50))}, {"s": 1}, 1000, {})
        checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)

        orig_size = os.path.getsize(ckpt_path)
        with open(ckpt_path, "r+b") as fh:
            fh.truncate(orig_size // 2)

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "size mismatch"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, map_location="cpu", load_fn=_pickle_load)

    def test_missing_manifest_fails_closed(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        payload = checkpoint_io.build_checkpoint_payload({"w": 1}, {"s": 1}, 1000, {})
        checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)
        os.remove(checkpoint_io.manifest_path_for(ckpt_path))

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Missing integrity manifest"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, map_location="cpu", load_fn=_pickle_load)

    def test_corrupt_manifest_json_fails_closed(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        payload = checkpoint_io.build_checkpoint_payload({"w": 1}, {"s": 1}, 1000, {})
        checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)

        with open(checkpoint_io.manifest_path_for(ckpt_path), "w", encoding="utf-8") as fh:
            fh.write("{not valid json")

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Unreadable integrity manifest"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, map_location="cpu", load_fn=_pickle_load)

    def test_inconsistent_manifest_step_metadata_fails_closed(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        payload = checkpoint_io.build_checkpoint_payload({"w": 1}, {"s": 1}, 1000, {})
        checkpoint_io.save_checkpoint_atomic(payload, ckpt_path, save_fn=_pickle_save)

        manifest_path = checkpoint_io.manifest_path_for(ckpt_path)
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["next_step"] = 9999
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "next_step"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, map_location="cpu", load_fn=_pickle_load)

    def test_interrupted_save_preserves_previous_checkpoint_and_cleans_temp_files(self):
        good_ckpt = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")
        good_payload = checkpoint_io.build_checkpoint_payload({"w": 1000}, {"s": 1000}, 1000, {})
        checkpoint_io.save_checkpoint_atomic(good_payload, good_ckpt, save_fn=_pickle_save)

        interrupted_ckpt = os.path.join(self.ckpt_dir, "ckpt_00002000.pt")
        next_payload = checkpoint_io.build_checkpoint_payload({"w": 2000}, {"s": 2000}, 2000, {})

        def failing_save(_payload, fh):
            fh.write(b"partial-unflushed-bytes")
            raise OSError("simulated power loss mid-save")

        with self.assertRaises(OSError):
            checkpoint_io.save_checkpoint_atomic(next_payload, interrupted_ckpt, save_fn=failing_save)

        self.assertFalse(os.path.exists(interrupted_ckpt))
        self.assertFalse(os.path.exists(checkpoint_io.manifest_path_for(interrupted_ckpt)))
        self.assertEqual(
            sorted(os.listdir(self.ckpt_dir)),
            ["ckpt_00001000.pt", "ckpt_00001000.pt.sha256"],
        )

        loaded_path, sd, start_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, map_location="cpu", load_fn=_pickle_load
        )
        self.assertEqual(loaded_path, good_ckpt)
        self.assertEqual(sd["model"]["w"], 1000)
        self.assertEqual(start_step, 1001)

    def test_unreadable_payload_raises_integrity_error_even_if_hash_matches(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt_00001000.pt")

        def raw_garbage_save(_payload, fh):
            fh.write(b"not-a-valid-pickle-stream")

        checkpoint_io.save_checkpoint_atomic(
            {"completed_step": 1000, "next_step": 1001}, ckpt_path, save_fn=raw_garbage_save
        )

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Failed to load checkpoint payload"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, map_location="cpu", load_fn=_pickle_load)

    def test_prune_old_checkpoints_removes_both_payload_and_manifest(self):
        for step in (1000, 2000, 3000, 4000, 5000):
            p = os.path.join(self.ckpt_dir, f"ckpt_{step:08d}.pt")
            payload = checkpoint_io.build_checkpoint_payload({"w": step}, {"s": step}, step, {})
            checkpoint_io.save_checkpoint_atomic(payload, p, save_fn=_pickle_save)

        removed = checkpoint_io.prune_old_checkpoints(self.ckpt_dir, keep=3)
        self.assertEqual(len(removed), 2)
        self.assertEqual(
            sorted(os.listdir(self.ckpt_dir)),
            [
                "ckpt_00003000.pt",
                "ckpt_00003000.pt.sha256",
                "ckpt_00004000.pt",
                "ckpt_00004000.pt.sha256",
                "ckpt_00005000.pt",
                "ckpt_00005000.pt.sha256",
            ],
        )

    def test_train_script_wires_atomic_checkpoint_and_verified_resume(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(repo_root, "train.py")
        with open(train_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=train_path)

        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)

        self.assertIn("load_latest_checkpoint", called_names)
        self.assertIn("build_checkpoint_payload", called_names)
        self.assertIn("save_checkpoint_atomic", called_names)
        self.assertIn("prune_old_checkpoints", called_names)
        self.assertNotIn("save", called_names)
        self.assertNotIn("load", called_names)


if __name__ == "__main__":
    unittest.main()
