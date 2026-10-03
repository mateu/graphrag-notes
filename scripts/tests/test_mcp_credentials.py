import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

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
