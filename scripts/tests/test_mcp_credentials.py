import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("mcp_credentials", Path(__file__).parents[1] / "provision-mcp-credentials.py")
credentials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(credentials)


class CredentialProvisioning(unittest.TestCase):
    def test_rotation_preserves_identity_and_other_clients_without_public_secrets(self):
        with tempfile.TemporaryDirectory() as base:
            path = Path(base) / "private"
            credentials.provision(path, ["openclaw-a"], ["hermes"])
            original = json.loads((path / "credentials.json").read_text())
            token = (path / "openclaw-a.env").read_text().split("'")[1]
            self.assertEqual(original["credentials"][0]["token_sha256"], hashlib.sha256(token.encode()).hexdigest())
            self.assertNotIn(token, json.dumps(original))
            credentials.rotate(path, "openclaw-a")
            rotated = json.loads((path / "credentials.json").read_text())
            self.assertEqual(original["credentials"][1], rotated["credentials"][1])
            self.assertEqual(rotated["credentials"][0]["instance_id"], "openclaw-a")
            self.assertNotEqual(original["credentials"][0]["token_sha256"], rotated["credentials"][0]["token_sha256"])
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                for file in path.iterdir():
                    self.assertEqual(file.stat().st_mode & 0o777, 0o600)

    def test_interrupted_rotation_recovers_the_same_staged_token(self):
        for failed_step in ["credentials.json", "openclaw-a.env"]:
            with self.subTest(failed_step=failed_step), tempfile.TemporaryDirectory() as base:
                path = Path(base) / "private"
                credentials.provision(path, ["openclaw-a"], ["hermes"])
                original_policy = json.loads((path / "credentials.json").read_text())
                original_env = (path / "openclaw-a.env").read_text()
                write = credentials.private_write
                def interrupted(target, text):
                    if target.name == failed_step:
                        raise OSError("synthetic interrupted write")
                    write(target, text)
                with patch.object(credentials, "private_write", side_effect=interrupted):
                    with self.assertRaises(OSError):
                        credentials.rotate(path, "openclaw-a")
                pending = json.loads((path / "openclaw-a.rotation.json").read_text())
                self.assertEqual((path / "openclaw-a.env").read_text(), original_env)
                if failed_step == "credentials.json":
                    self.assertEqual(json.loads((path / "credentials.json").read_text()), original_policy)
                with patch.object(credentials.secrets, "token_urlsafe", side_effect=AssertionError("must reuse staged token")):
                    credentials.rotate(path, "openclaw-a")
                policy = json.loads((path / "credentials.json").read_text())
                self.assertEqual(policy["credentials"][0]["token_sha256"], pending["new_hash"])
                self.assertEqual(policy["credentials"][1], original_policy["credentials"][1])
                self.assertIn(pending["token"], (path / "openclaw-a.env").read_text())
                self.assertFalse((path / "openclaw-a.rotation.json").exists())

    def test_rejects_over_service_identity_limit_before_writing(self):
        with tempfile.TemporaryDirectory() as base:
            path = Path(base) / "private"
            with self.assertRaises(ValueError):
                credentials.provision(path, [f"client-{i}" for i in range(129)], [])
            self.assertFalse(path.exists())

    @unittest.skipUnless(os.name == "posix", "private file permissions require POSIX")
    def test_rotation_refuses_nonregular_or_exposed_recovery_without_changing_tokens(self):
        for kind in ["fifo", "directory", "symlink", "broken-symlink", "exposed-file"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as base:
                path = Path(base) / "private"
                credentials.provision(path, ["openclaw-a"], ["hermes"])
                policy = path / "credentials.json"
                original_policy = policy.read_bytes()
                token_file = path / "openclaw-a.env"
                original_token = token_file.read_bytes()
                pending = path / "openclaw-a.rotation.json"
                if kind == "fifo":
                    os.mkfifo(pending, 0o600)
                elif kind == "directory":
                    pending.mkdir(mode=0o700)
                elif kind in ["symlink", "broken-symlink"]:
                    target = Path(base) / "recovery.json"
                    if kind == "symlink":
                        target.write_text("{}\n")
                        target.chmod(0o600)
                    pending.symlink_to(target)
                else:
                    pending.write_text("{}\n")
                    pending.chmod(0o644)
                # With no FIFO writer, an attempted read would block. A real
                # process and deadline prove refusal occurs before that read.
                result = subprocess.run(
                    [sys.executable, str(spec.origin), str(path), "--rotate", "openclaw-a"],
                    capture_output=True, text=True, timeout=2,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("Credential setup failed", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn(original_token.decode().split("'")[1], result.stderr)
                self.assertEqual(policy.read_bytes(), original_policy)
                self.assertEqual(token_file.read_bytes(), original_token)
                self.assertTrue(pending.exists() or pending.is_symlink())

    def test_refuses_overwrite_and_invalid_ids_before_writing(self):
        with tempfile.TemporaryDirectory() as base:
            path = Path(base) / "private"
            for writers, readers in [(["../escape"], []), (["same"], ["same"])]:
                with self.assertRaises(ValueError):
                    credentials.provision(path, writers, readers)
                self.assertFalse(path.exists())
            credentials.provision(path, ["a"], [])
            before = (path / "credentials.json").read_bytes()
            with self.assertRaises(FileExistsError):
                credentials.provision(path, ["b"], [])
            with self.assertRaises(ValueError):
                credentials.rotate(path, "missing")
            self.assertEqual(before, (path / "credentials.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
