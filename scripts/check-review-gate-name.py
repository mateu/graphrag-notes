#!/usr/bin/env python3
"""Soft guard: reserve the literal Review gate name for its publisher workflow.

This is a typo/accidental-collision check, not an adversarial security boundary.
"""
from pathlib import Path
import sys


def violations(root):
    return sorted(str(p.relative_to(root)) for p in (root / '.github/workflows').iterdir()
                  if p.suffix.lower() in ('.yml', '.yaml') and p.name != 'review-gate.yml'
                  and 'Review gate' in p.read_text(encoding='utf-8'))


def main():
    found = violations(Path(__file__).resolve().parents[1])
    if found:
        print('Reserved check name appears outside review-gate.yml: ' + ', '.join(found), file=sys.stderr)
    return bool(found)


if __name__ == '__main__':
    sys.exit(main())
