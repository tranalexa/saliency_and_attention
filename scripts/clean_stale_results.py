#!/usr/bin/env python3
"""Remove out-of-scope / legacy result files without re-running experiments.

Keeps the current protocol (Class A/B/C per docs/experimental_protocol.md).
Safe to run after a full cascade + occlusion download.

Usage:
  python3 scripts/clean_stale_results.py              # dry-run (default)
  python3 scripts/clean_stale_results.py --apply        # delete locally
  python3 scripts/clean_stale_results.py --apply --modal  # local + Modal volume
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
VOLUME = "saliency-results"

# Removed methods; attention_rollout replaced raw_attn (do not delete attention_rollout_*).
STALE_STEMS = frozenset(
    {
        "smoothgrad",
        "ig_smoothgrad",
        "gbp_gc",
        "dino_attn",
        "relevancy_rollout",
        "raw_attn",
    }
)

# Legacy short name (not attention_rollout).
_ROLLOUT_RE = re.compile(
    r"^(baseline_)?rollout([_.]|$)|^occlusion_rollout([_.]|$)"
)


def _stem_from_name(name: str) -> str | None:
    if name.startswith("baseline_") and name.endswith(".npz"):
        return name[len("baseline_") : -len(".npz")]
    if name.startswith("occlusion_"):
        rest = name[len("occlusion_") :]
        for stem in sorted(STALE_STEMS, key=len, reverse=True):
            if rest.startswith(stem + "_") or rest == stem + ".npy":
                return stem
        if _ROLLOUT_RE.match("occlusion_" + rest.split("_", 1)[0] + "_"):
            return "rollout"
        return None
    for stem in sorted(STALE_STEMS, key=len, reverse=True):
        if name.startswith(stem + "_") or name == stem + ".npy":
            return stem
    if _ROLLOUT_RE.match(name):
        return "rollout"
    return None


def is_stale_file(path: Path) -> bool:
    if path.name == "baseline_maps.npz":
        return True
    stem = _stem_from_name(path.name)
    if stem is None:
        return False
    if stem == "rollout" and path.name.startswith("attention_rollout"):
        return False
    return stem in STALE_STEMS or stem == "rollout"


def nested_resnet_duplicate(path: Path, arch_dir: Path) -> bool:
    """Drop results/resnet50/resnet50/* duplicate tree."""
    try:
        rel = path.relative_to(arch_dir)
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) >= 2 and parts[0] == "resnet50"


def collect_targets(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.name == "dinov2":
            out.append(root)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if nested_resnet_duplicate(path, root):
                out.append(path)
            elif is_stale_file(path):
                out.append(path)
    return out


def _modal_bin() -> str:
    venv = REPO / ".venv" / "bin" / "modal"
    return str(venv) if venv.exists() else "modal"


def modal_list(prefix: str) -> list[dict]:
    cmd = [_modal_bin(), "volume", "ls", VOLUME, prefix, "--json"]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def modal_rm(volume_path: str, apply: bool) -> None:
    # Volume paths use leading slash: /vit/foo.npy
    path = volume_path if volume_path.startswith("/") else "/" + volume_path
    cmd = [_modal_bin(), "volume", "rm", VOLUME, path]
    if apply:
        subprocess.run(cmd, check=True)
    else:
        print("  modal:", " ".join(cmd))


def collect_modal_stale(archs: tuple[str, ...] = ("vit", "resnet50")) -> list[str]:
    """List stale file paths on the Modal results volume."""
    stale: list[str] = []
    for arch in archs:
        try:
            entries = modal_list("/" + arch)
        except subprocess.CalledProcessError:
            continue
        for entry in entries:
            if entry.get("Type") != "file":
                continue
            filename = entry["Filename"].lstrip("/")
            path = Path(filename.split("/", 1)[-1])
            if is_stale_file(path):
                stale.append("/" + filename)
    dinov2 = "/dinov2"
    try:
        modal_list(dinov2)
        stale.append(dinov2)
    except subprocess.CalledProcessError:
        pass
    return sorted(set(stale))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Also remove matching paths on Modal volume saliency-results",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Local results root (default: ./results)",
    )
    args = parser.parse_args()
    apply = args.apply

    arch_roots = [
        args.results_dir / "vit",
        args.results_dir / "resnet50",
        args.results_dir / "dinov2",
    ]
    targets = collect_targets(arch_roots)

    if not targets and not args.modal:
        print("No stale files found under", args.results_dir)
        return 0

    if not targets:
        print("No stale files locally under", args.results_dir)
    else:
        by_parent: dict[str, list[Path]] = {}
        for p in targets:
            by_parent.setdefault(str(p.parent), []).append(p)

        print(
            "%s: %d stale file(s) under %s"
            % ("DELETE" if apply else "DRY-RUN", len(targets), args.results_dir)
        )
        for parent in sorted(by_parent):
            files = by_parent[parent]
            print("  %s/ (%d)" % (parent, len(files)))
            for f in files[:5]:
                print("    -", f.name)
            if len(files) > 5:
                print("    ... +%d more" % (len(files) - 5))

        if apply:
            dinov2_dir = args.results_dir / "dinov2"
            if dinov2_dir in targets:
                import shutil

                shutil.rmtree(dinov2_dir)
                targets = [t for t in targets if t != dinov2_dir]
            for path in targets:
                path.unlink()
            print("Deleted locally.")

    if args.modal:
        print("\nModal volume (%s):" % VOLUME)
        modal_targets = collect_modal_stale()
        if not modal_targets:
            print("  No stale files on volume.")
        for vol_path in modal_targets:
            if vol_path == "/dinov2":
                cmd = [_modal_bin(), "volume", "rm", VOLUME, vol_path, "--recursive"]
                if apply:
                    subprocess.run(cmd, check=True)
                else:
                    print("  modal:", " ".join(cmd))
            else:
                modal_rm(vol_path, apply=apply)
        if not apply and modal_targets:
            print("(%d file(s); re-run with --apply --modal to delete.)" % len(modal_targets))

    if not apply:
        print("\nRe-run with --apply to delete. Add --modal to mirror on the volume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
