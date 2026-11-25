"""นับยานพาหนะจากรูป CCTV ด้วย YOLO (pretrained COCO) -> ตาราง Parquet

อ่าน manifest ที่ build_manifest.py สร้าง แล้ว:
  1. สุ่มเฟรมต่อกล้อง (--every-minutes) เพื่อไม่ต้องรัน 60,000 รูป
  2. รัน YOLO นับ car / motorcycle / bus / truck (+ person ถ้าเปิด)
  3. เขียนผลเป็น part ไฟล์ใน --out (รันซ้ำ = ทำต่อจากเดิม ไม่เริ่มใหม่)

ตัวอย่าง
--------
# ลองก่อน 200 รูป ว่าตั้งค่าถูก
python detect_vehicles.py --manifest data/bronze/manifest.parquet \
        --out data/silver/detections --every-minutes 10 --limit 200

# รันเต็ม
python detect_vehicles.py --manifest data/bronze/manifest.parquet \
        --out data/silver/detections --every-minutes 10 --model yolov8n.pt --imgsz 640

# รวม part -> ไฟล์เดียวสำหรับ DuckDB
python detect_vehicles.py --out data/silver/detections --merge-to data/silver/counts.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):  # คอนโซล Windows ไม่ใช่ UTF-8 โดยดีฟอลต์
    sys.stdout.reconfigure(encoding="utf-8")

# COCO class id -> ชื่อคอลัมน์ที่เราสนใจ
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0
FLUSH_EVERY = 500  # จำนวนแถวก่อนเขียน part ไฟล์


def sample_per_camera(df: pd.DataFrame, every_minutes: int) -> pd.DataFrame:
    """เก็บเฟรมแรกของทุกช่วง N นาที ต่อกล้อง 1 ตัว"""
    if every_minutes <= 0:
        return df
    step = pd.Timedelta(minutes=every_minutes)
    keep: list[int] = []
    for _, g in df.groupby("camera_id", sort=False):
        g = g.sort_values("ts", kind="mergesort")
        last = None
        for idx, ts in zip(g.index, g["ts"]):
            if last is None or (ts - last) >= step:
                keep.append(idx)
                last = ts
    return df.loc[sorted(keep)]


def load_done(part_dir: Path) -> set[str]:
    if not part_dir.exists():
        return set()
    parts = sorted(part_dir.glob("part-*.parquet"))
    if not parts:
        return set()
    done: set[str] = set()
    for p in parts:
        done.update(pd.read_parquet(p, columns=["path"])["path"].tolist())
    return done


def merge_parts(part_dir: Path, out_file: Path, manifest: Path) -> None:
    parts = sorted(part_dir.glob("part-*.parquet"))
    if not parts:
        sys.exit(f"[x] ไม่มี part ไฟล์ใน {part_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["path"], keep="last")

    if "camera_id" not in df.columns:  # part ดิบมีแค่ผลนับ -> เติม metadata จาก manifest
        meta = pd.read_parquet(manifest)[["path", "camera_id", "ts", "rel_path"]]
        df = df.merge(meta, on="path", how="left")

    df = df.sort_values(["camera_id", "ts"]).reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    print(f"[ok] รวม {len(parts)} part -> {out_file}  ({len(df):,} แถว, {out_file.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="นับยานพาหนะในรูป CCTV ด้วย YOLO")
    ap.add_argument("--manifest", type=Path, default=Path("data/bronze/manifest.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/silver/detections"), help="โฟลเดอร์เก็บ part ไฟล์")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt หรือ path ของ weight ที่ fine-tune แล้ว")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2, help="จำนวนรูปต่อรอบ (CPU ใช้ 2 พอ ถ้าแรมเหลือน้อยให้ลดเป็น 1; เพิ่มได้ถ้าแรมว่าง)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1")
    ap.add_argument("--every-minutes", type=int, default=10, help="สุ่ม 1 เฟรมต่อ N นาทีต่อกล้อง (0 = ทุกรูป)")
    ap.add_argument("--with-person", action="store_true", help="นับคนด้วย (class person)")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนรูป (ไว้ลองก่อน)")
    ap.add_argument("--dry-run", action="store_true", help="แค่ดูว่าจะรันกี่รูป")
    ap.add_argument("--merge-to", type=Path, default=None, help="รวม part ไฟล์ทั้งหมดเป็นไฟล์เดียวแล้วจบ")
    args = ap.parse_args()

    if args.merge_to:
        merge_parts(args.out, args.merge_to, args.manifest)
        return

    if not args.manifest.exists():
        sys.exit(f"[x] ไม่พบ manifest {args.manifest} — รัน build_manifest.py ก่อน")

    df = pd.read_parquet(args.manifest)
    missing_ts = int(df["ts"].isna().sum())
    if missing_ts:
        print(f"[!] ตัด {missing_ts:,} รูปที่ไม่มี timestamp ออก (จัดเวลาไม่ได้)")
        df = df[df["ts"].notna()]

    df = df.sort_values(["camera_id", "ts"], kind="mergesort")
    total_before = len(df)
    df = sample_per_camera(df, args.every_minutes)

    done = load_done(args.out)
    if done:
        df = df[~df["path"].isin(done)]
        print(f"[i] ทำไปแล้ว {len(done):,} รูป -> เหลือ {len(df):,}")
    if args.limit:
        df = df.head(args.limit)

    print(f"[i] ทั้งหมด {total_before:,} รูป -> สุ่มทุก {args.every_minutes} นาที/กล้อง -> {len(df):,} รูป ({df['camera_id'].nunique()} กล้อง)")
    if args.dry_run or df.empty:
        print("[dry-run] จบก่อนโหลดโมเดล" if args.dry_run else "[i] ไม่มีงานค้าง")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[x] ยังไม่มี ultralytics:  python -m pip install -U ultralytics")

    device = None if args.device == "auto" else args.device
    model = YOLO(args.model)

    args.out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows: list[dict] = []
    part_seq = 0
    started = time.time()

    def flush() -> None:
        nonlocal rows, part_seq
        if not rows:
            return
        part_seq += 1
        pd.DataFrame(rows).to_parquet(args.out / f"part-{run_id}-{part_seq:04d}.parquet", index=False)
        rows = []

    results = model.predict(
        source=df["path"].tolist(),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=device,
        stream=True,
        verbose=False,
    )

    try:
        for n, res in enumerate(results, 1):
            cls_ids = res.boxes.cls.int().tolist() if res.boxes is not None else []
            row = {"path": res.path, "n_boxes": len(cls_ids)}
            for cid, name in VEHICLE_CLASSES.items():
                row[name] = cls_ids.count(cid)
            row["person"] = cls_ids.count(PERSON_CLASS) if args.with_person else None
            row["n_vehicles"] = sum(row[nm] for nm in VEHICLE_CLASSES.values())
            row["infer_ms"] = round(float(res.speed.get("inference", 0.0)), 1)
            rows.append(row)

            if n % FLUSH_EVERY == 0:
                flush()
                rate = n / (time.time() - started)
                left = (len(df) - n) / rate if rate else 0
                print(f"  {n:,}/{len(df):,}  ({rate:.1f} รูป/วินาที, เหลือ ~{left / 60:.0f} นาที)", flush=True)
    except RuntimeError as exc:
        flush()
        if "not enough memory" in str(exc) or "DefaultCPUAllocator" in str(exc):
            sys.exit(
                "[x] แรมไม่พอสำหรับ YOLO — ลดขนาด batch/imgsz แล้วรันใหม่ (ทำต่อจาก part เดิมได้):\n"
                "    python detect_vehicles.py --manifest data/bronze/manifest.parquet "
                "--out data/silver/detections --every-minutes 10 --batch 1 --imgsz 416"
            )
        raise

    flush()

    # รวมผลนับเข้ากับ metadata ของรูป
    parts = sorted(args.out.glob("part-*.parquet"))
    counts = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    counts = counts.drop_duplicates(subset=["path"], keep="last")
    meta = pd.read_parquet(args.manifest)[["path", "camera_id", "ts", "rel_path"]]
    out = counts.merge(meta, on="path", how="left")
    out = out.sort_values(["camera_id", "ts"]).reset_index(drop=True)

    final = args.out / "counts_latest.parquet"
    out.to_parquet(final, index=False)
    print(f"\n[ok] {len(out):,} แถว -> {final}")
    print(f"     รถรวมทั้งหมด {int(out['n_vehicles'].sum()):,} คัน | รูปที่ไม่พบรถเลย {int((out['n_vehicles'] == 0).sum()):,} รูป")


if __name__ == "__main__":
    main()
