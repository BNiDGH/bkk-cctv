"""สร้าง manifest (bronze layer) จากโฟลเดอร์รูป CCTV -> Parquet

อ่านชื่อไฟล์/โฟลเดอร์เพื่อดึง camera_id + timestamp แล้วเขียนตารางเดียวที่
ทุกสคริปต์ปลายน้ำใช้ต่อ (detect_vehicles.py, DuckDB, dashboard)

ตัวอย่าง
--------
python build_manifest.py --images "D:/cctv/raw" --out data/bronze/manifest.parquet
python build_manifest.py --images "D:/cctv/raw" --out data/bronze/manifest.parquet \
        --probe-size --workers 16 --csv data/bronze/manifest.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):  # คอนโซล Windows ไม่ใช่ UTF-8 โดยดีฟอลต์
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ลำดับความสำคัญ: epoch(ms) -> epoch(s) -> YYYYMMDD_HHMMSS (มี/ไม่มีตัวคั่น) -> YYYYMMDD_HHMM
_EPOCH_MS = re.compile(r"(?<!\d)(1[5-9]\d{11})(?!\d)")
_EPOCH_S = re.compile(r"(?<!\d)(1[5-9]\d{8})(?!\d)")
_FULL_TS = re.compile(
    r"(?P<Y>20\d{2})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})"
    r"[T_ -]?(?P<H>\d{2})[-_.:]?(?P<M>\d{2})[-_.:]?(?P<S>\d{2})"
)
_MIN_TS = re.compile(
    r"(?P<Y>20\d{2})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})"
    r"[T_ -](?P<H>\d{2})[-_.:]?(?P<M>\d{2})"
)
_CAM_RE = re.compile(r"(?:cam|camera|cctv)[-_ ]?0*(\d{1,4})", re.IGNORECASE)


def parse_timestamp(name: str) -> tuple[datetime | None, str]:
    """ดึงเวลาในชื่อไฟล์ คืน (datetime, แหล่งที่มา)"""
    m = _EPOCH_MS.search(name)
    if m:
        return datetime.fromtimestamp(int(m.group(1)) / 1000), "filename_epoch_ms"
    m = _EPOCH_S.search(name)
    if m:
        return datetime.fromtimestamp(int(m.group(1))), "filename_epoch_s"

    for pattern, source in ((_FULL_TS, "filename"), (_MIN_TS, "filename_minute")):
        m = pattern.search(name)
        if not m:
            continue
        g = m.groupdict()
        try:
            return (
                datetime(
                    int(g["Y"]), int(g["m"]), int(g["d"]),
                    int(g["H"]), int(g["M"]), int(g.get("S") or 0),
                ),
                source,
            )
        except ValueError:  # 20261345_256199 = ไม่ใช่วันเวลาจริง
            continue
    return None, "missing"


def exif_timestamp(path: Path) -> datetime | None:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
            if raw:
                return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def camera_id_for(path: Path, root: Path, pattern: re.Pattern[str] | None) -> str:
    """กล้องมาจากชื่อโฟลเดอร์แม่ก่อน ถ้าไม่มีค่อยดูจากชื่อไฟล์"""
    if pattern is not None:
        m = pattern.search(path.as_posix())
        if m:
            return f"cam_{int(m.group(1)):03d}" if m.groups() else m.group(0)

    rel = path.relative_to(root)
    if len(rel.parts) >= 2:
        return rel.parts[0]

    m = _CAM_RE.search(path.stem)
    if m:
        return f"cam_{int(m.group(1)):03d}"
    return "cam_unknown"


def probe_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as im:  # อ่านแค่ header ไม่ decode รูปเต็ม
            return im.width, im.height
    except Exception:
        return None, None


def build(images: Path, probe: bool, workers: int, use_mtime: bool) -> pd.DataFrame:
    files = sorted(
        p for p in images.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )
    if not files:
        sys.exit(f"[x] ไม่พบรูปใน {images}")

    rows = []
    for path in files:
        ts, ts_source = parse_timestamp(path.stem)
        if ts is None:
            ts = exif_timestamp(path)
            ts_source = "exif" if ts else ts_source
        if ts is None and use_mtime:
            ts = datetime.fromtimestamp(path.stat().st_mtime)
            ts_source = "mtime"
        rows.append(
            {
                "camera_id": camera_id_for(path, images, None),
                "path": str(path.resolve()),
                "rel_path": path.relative_to(images).as_posix(),
                "file_name": path.name,
                "ts": ts,
                "ts_source": ts_source,
                "size_bytes": path.stat().st_size,
            }
        )

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])

    if probe:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(lambda p: probe_size(Path(p)), df["path"]))
        df["width"] = [s[0] for s in sizes]
        df["height"] = [s[1] for s in sizes]

    return df.sort_values(["camera_id", "ts", "file_name"], na_position="last").reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    print(f"\nรูปทั้งหมด : {len(df):,}")
    print(f"กล้อง      : {df['camera_id'].nunique()}")
    print(f"ช่วงเวลา   : {df['ts'].min()} -> {df['ts'].max()}")
    print("\nที่มาของ timestamp:")
    for src, n in df["ts_source"].value_counts().items():
        print(f"  {src:<20} {n:>8,}")
    bad = int(df["ts"].isna().sum())
    if bad:
        print(f"\n[!] ไม่มี timestamp {bad:,} รูป -> จะถูกตัดทิ้งตอน detect (ต้องแก้ที่ชื่อไฟล์หรือ EXIF)")

    per_cam = df.groupby("camera_id").size()
    print(f"\nรูปต่อกล้อง: min {per_cam.min():,} / median {int(per_cam.median()):,} / max {per_cam.max():,}")
    thin = per_cam[per_cam < per_cam.median() * 0.5]
    if len(thin):
        print(f"[!] กล้องที่รูปน้อยผิดปกติ ({len(thin)}): {', '.join(thin.index[:10])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="สร้าง manifest จากรูป CCTV")
    ap.add_argument("--images", required=True, type=Path, help="โฟลเดอร์รากของรูป (ข้างในมีโฟลเดอร์ต่อกล้อง)")
    ap.add_argument("--out", type=Path, default=Path("data/bronze/manifest.parquet"))
    ap.add_argument("--csv", type=Path, default=None, help="เขียน CSV สำรองไว้เปิดดูด้วยตา")
    ap.add_argument("--probe-size", action="store_true", help="อ่าน width/height ของรูป (ช้ากว่า แต่ช่วย QC)")
    ap.add_argument("--workers", type=int, default=8, help="จำนวน thread ตอน probe ขนาดรูป")
    ap.add_argument("--no-mtime-fallback", action="store_true", help="ห้ามใช้ mtime ของไฟล์เป็น timestamp")
    args = ap.parse_args()

    if not args.images.is_dir():
        sys.exit(f"[x] ไม่พบโฟลเดอร์ {args.images}")

    df = build(args.images, args.probe_size, args.workers, not args.no_mtime_fallback)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    report(df)
    print(f"\n[ok] เขียนแล้ว -> {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
