#!/usr/bin/env python3
"""Deprecated: use ``post_ga.py`` instead."""
from __future__ import annotations

import sys

print(
    "run_client_final.py is deprecated — use post_ga.py "
    "(CLiENT ga_final + GetDist in one place).",
    file=sys.stderr,
)
from post_ga import main

raise SystemExit(main())
