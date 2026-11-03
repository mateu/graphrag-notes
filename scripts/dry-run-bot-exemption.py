#!/usr/bin/env python3
"""Read live merged PR evidence with gh; evaluate locally without publishing checks."""
import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location('review_gate', Path(__file__).with_name('review-gate.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
for number in (222, 223, 224):
    endpoint = f'repos/mateu/graphrag-notes/pulls/{number}'
    def api(suffix):
        return json.loads(subprocess.check_output(['gh', 'api', endpoint + suffix]))
    pr = api('')
    # Pagination preserves the evaluator's completeness checks for future reuse.
    def listing(suffix):
        items = []
        page = 1
        while True:
            batch = api(suffix + f'?per_page=100&page={page}')
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1
    exempt, reason = gate.bot_exemption(pr['head']['sha'], pr, listing('/commits'), listing('/files'))
    print(f'#{number}: {"exempt" if exempt else "not-exempt"}: {reason}')
