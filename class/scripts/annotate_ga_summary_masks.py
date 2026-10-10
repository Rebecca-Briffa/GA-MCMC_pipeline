#!/usr/bin/env python3
"""
Add a gene-mask column to an existing GA_summary.txt (from expr).

Usage
-----
    python annotate_ga_summary_masks.py GA_summary_ga-run11.txt
    python annotate_ga_summary_masks.py GA_summary.txt -o GA_summary_with_masks.txt

Reconstructs the canonical unique-shape mask from each ``w(a)=...`` line
using the current ``chromosome.SHAPE_POOL`` / coeff mapping. Rows that cannot
be parsed keep ``mask=-``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from chromosome import N_BASIS, mask_from_expr, mask_to_bitstring


LINE_RE = re.compile(
    r"^(?P<id>gen_\d+_model_\d+)\s+"
    r"(?P<chi2>\S+)\s+"
    r"(?P<fit>\S+)\s+"
    r"(?P<k>\S+)\s+"
    r"(?P<status>\S+)\s+"
    r"(?P<backend>\S+)\s+"
    r"(?P<rest>.*)$"
)
HEADER_RE = re.compile(
    r"^ID\s+chi2\s+fitness\s+k\s+status\s+likelihood_backend\s+"
)


def annotate_line(line: str) -> str:
    raw = line.rstrip("\n")
    if HEADER_RE.match(raw.strip()) and "expr" in raw and "mask" not in raw:
        # Insert mask column before expr
        return (
            f"{'ID':<15} {'chi2':<12} {'fitness':<12} {'k':<6} {'status':<10} "
            f"{'likelihood_backend':<18} {'mask':<{N_BASIS + 2}} expr"
        )

    m = LINE_RE.match(raw.strip())
    if not m:
        return raw
    if m.group("status").lower() not in ("success", "failed"):
        return raw

    rest = m.group("rest").strip()
    # Already annotated?
    if re.match(r"^[01]{" + str(N_BASIS) + r"}\s+w\(a\)=", rest):
        return raw

    expr_m = re.search(r"(w\(a\)=.+)$", rest)
    if not expr_m:
        mask_bits = "-" * min(8, N_BASIS)
        expr = rest
    else:
        expr = expr_m.group(1)
        try:
            mask_bits = mask_to_bitstring(mask_from_expr(expr))
        except Exception:
            mask_bits = "-" * N_BASIS

    return (
        f"{m.group('id'):<15} {m.group('chi2'):<12} {m.group('fit'):<12} "
        f"{m.group('k'):<6} {m.group('status'):<10} {m.group('backend'):<18} "
        f"{mask_bits:<{N_BASIS + 2}} {expr}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path, help="Path to GA_summary.txt")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <stem>_with_masks.txt beside input)",
    )
    args = p.parse_args()
    src = args.summary.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"not found: {src}")
    out = (
        args.output.expanduser().resolve()
        if args.output
        else src.with_name(src.stem + "_with_masks.txt")
    )

    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    annotated = [annotate_line(ln) for ln in lines]
    out.write_text("\n".join(annotated) + "\n", encoding="utf-8")
    print(f"Wrote {out}  (mask length = {N_BASIS} genes)")


if __name__ == "__main__":
    main()
