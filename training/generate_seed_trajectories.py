from __future__ import annotations

from pathlib import Path

from common import ROOT, collect_seed_records


def main() -> None:
    output_path = ROOT / "training" / "data" / "seed_trajectories.jsonl"
    records = collect_seed_records(output_path)
    print(f"Wrote {len(records)} seed records to {output_path}")


if __name__ == "__main__":
    main()
