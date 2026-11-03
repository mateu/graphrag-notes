import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('name_guard', Path(__file__).resolve().parents[1] / 'check-review-gate-name.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class NameGuardTests(unittest.TestCase):
    def test_workflow_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflows = root / '.github/workflows'
            workflows.mkdir(parents=True)
            (workflows / 'review-gate.yml').write_text('name: Review gate')
            (workflows / 'ordinary.yml').write_text('name: normal')
            self.assertEqual([], guard.violations(root))
            for text in ('name: Review gate', 'context: "Review gate"', 'name: "Review gate"'):
                (workflows / 'other.yaml').write_text(text)
                self.assertEqual(['.github/workflows/other.yaml'], guard.violations(root))
