"""สร้าง counts.parquet ตัวอย่าง (ไม่ต้องใช้รูปจริงหรือ YOLO)

ใช้สำหรับทดสอบ build_gold.py / dashboard.py และสำหรับ GitHub Actions

    python scripts/make_sample_counts.py --out data/silver/counts.parquet
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def camera_ids(locations: Path) -> list[str]:
    if locations.exists():
        df = pd.read_csv(locations)
        col = next((c for c in ("camera_id", "cam_id", "id") if c in df.columns), None)
        if col:
            return [str(v) for v in df[col].tolist()]
    return ["101", "102", "103"]


def main() -> None:
    ap = argparse.ArgumentParser(description="สร้างข้อมูล counts ตัวอย่าง")
    ap.add_argument("--out", type=Path, default=Path("data/silver/counts.parquet"))
    ap.add_argument("--locations", type=Path, default=Path("cctv_info.example.csv"))
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--start", default="2025-06-01", help="YYYY-MM-DD")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    cams = camera_ids(args.locations)
    t0 = datetime.strptime(args.start, "%Y-%m-%d")

    rows = []
    for i, cam in enumerate(cams):
        base = 20 - (i % 5) * 3  # แต่ละกล้องหนาแน่นไม่เท่ากัน
        for d in range(args.days):
            for h in range(24):
                rush = 2.1 if h in (7, 8, 17, 18) else (1.3 if 6 <= h <= 20 else 0.4)
                for f in range(2):
                    ts = t0 + timedelta(days=d, hours=h, minutes=30 * f)
                    n = max(0, int(random.gauss(base * rush, 2)))
                    rows.append(
                        {
                            "path": f"sample/{cam}/{ts:%Y%m%d_%H%M}.jpg",
                            "camera_id": cam,
                            "ts": ts,
                            "car": int(n * 0.6),
                            "motorcycle": int(n * 0.3),
                            "bus": int(n * 0.05),
                            "truck": n - int(n * 0.6) - int(n * 0.3) - int(n * 0.05),
                            "person": None,
                            "n_vehicles": n,
                            "n_boxes": n,
                            "infer_ms": 20.0,
                        }
                    )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[ok] {args.out} — {len(df):,} แถว, {df['camera_id'].nunique()} กล้อง, {df['ts'].min()} -> {df['ts'].max()}")


if __name__ == "__main__":
    main()
