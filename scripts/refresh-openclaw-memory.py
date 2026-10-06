#!/usr/bin/env python3
"""Refresh only indexed OpenClaw memory; see docs/openclaw-memory-refresh.md."""
import sys

# Versioned installed bundles must remain identical after ordinary client use.
sys.dont_write_bytecode = True
from openclaw_memory_refresh import main

if __name__ == "__main__":
    raise SystemExit(main())
