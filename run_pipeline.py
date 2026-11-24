"""รัน pipeline ทั้งหมดต่อกัน: manifest -> YOLO -> gold -> weather -> QC

    python run_pipeline.py --images "D:/cctv/raw"
    python run_pipeline.py --images "D:/cctv/raw" --every-minutes 10 --limit 200   # ลองเร็ว ๆ
    python run_pipeline.py --skip manifest,detect                                   # ใช้ counts ที่มีอยู่
    python run_pipeline.py --images "D:/cctv/raw" --dry-run                         # ดูคำสั่งก่อน

หลักการ
  - ทุก stage เขียน log ลง logs/pipeline_<เวลา>.log พร้อมเวลาและ exit code
  - ตรวจว่า artifact ของแต่ละ stage เกิดขึ้นจริง (มีไฟล์ + จำนวนแถว > 0) ก่อนไปต่อ
  - รันซ้ำได้: manifest เขียนทับ, YOLO ทำต่อจาก part เดิม, weather ใช้ cache
  - weather ล้มเหลว (เน็ต/โควตา) ไม่ทำให้ทั้ง pipeline หยุด เว้นแต่ใส่ --strict
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
STAGES = ["manifest", "detect", "gold", "weather", "qc"]

PATHS = {
    "manifest": BASE / "data/bronze/manifest.parquet",
    "counts": BASE / "data/silver/counts.parquet",
    "gold": BASE / "data/gold",
}


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0


def run_stage(name: str, cmd: list[str], log: Log) -> tuple[bool, float]:
    log("")
    log("=" * 72)
    log(f"▶ {name}")
    log("  $ " + " ".join(cmd))
    log("=" * 72)
    started = time.time()

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
    code = proc.wait()

    duration = time.time() - started
    log(f"{'✓' if code == 0 else '✗'} {name} — {duration:.1f}s (exit {code})")
    return code == 0, duration


def main() -> None:
    ap = argparse.ArgumentParser(description="รัน pipeline CCTV ตั้งแต่ต้นจนจบ")
    ap.add_argument("--images", type=Path, default=None, help="โฟลเดอร์รูปดิบ (ต้องมี ถ้าไม่ skip manifest)")
    ap.add_argument("--every-minutes", type=int, default=10, help="สุ่ม 1 เฟรมต่อ N นาทีต่อกล้อง")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนรูปใน stage YOLO (ไว้ลอง)")
    ap.add_argument("--with-person", action="store_true")
    ap.add_argument("--probe-size", action="store_true", help="อ่าน width/height ตอนทำ manifest")
    ap.add_argument("--skip", default="", help="ข้าม stage เช่น manifest,detect")
    ap.add_argument("--strict", action="store_true", help="ให้ weather ล้มเหลวแล้วหยุดทั้ง pipeline")
    ap.add_argument("--weather-max-cameras", type=int, default=0, help="จำกัดกล้องตอนดึงอากาศ (ไว้ลอง)")
    ap.add_argument("--dry-run", action="store_true", help="พิมพ์คำสั่งทั้งหมด ไม่รันจริง")
    ap.add_argument("--log-dir", type=Path, default=BASE / "logs")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(STAGES)
    if unknown:
        sys.exit(f"[x] --skip ไม่รู้จัก: {sorted(unknown)} (เลือกจาก {STAGES})")

    if "manifest" not in skip and not args.images:
        sys.exit("[x] ต้องระบุ --images (หรือใส่ --skip manifest ถ้ามี manifest แล้ว)")

    py = sys.executable
    log = Log(args.log_dir / f"pipeline_{datetime.now():%Y%m%d-%H%M%S}.log")
    t_start = time.time()

    log(f"Pipeline เริ่ม {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"โฟลเดอร์ : {BASE}")
    log(f"Log      : {log.path}")
    log(f"Skip     : {sorted(skip) or 'ไม่มี'}")

    results: dict[str, tuple[str, float]] = {}

    def mark(name: str, status: str, duration: float = 0.0) -> None:
        results[name] = (status, duration)

    # ------------------------------------------------------------ 1. manifest
    if "manifest" in skip:
        mark("manifest", "skipped")
    else:
        cmd = [py, "build_manifest.py", "--images", str(args.images), "--out", str(PATHS["manifest"])]
        if args.probe_size:
            cmd.append("--probe-size")
        if args.dry_run:
            log("\n[dry-run] " + " ".join(cmd))
            mark("manifest", "dry-run")
        else:
            ok, dur = run_stage("manifest", cmd, log)
            mark("manifest", "ok" if ok else "failed", dur)
            if not ok:
                log("\n[x] manifest ล้มเหลว — หยุด pipeline")
                log.close()
                sys.exit(1)

    # ------------------------------------------------------------ 2. detect (YOLO + merge)
    if "detect" in skip:
        mark("detect", "skipped")
    else:
        if not PATHS["manifest"].exists() and not args.dry_run:
            log(f"\n[x] ไม่พบ {PATHS['manifest']} — รัน stage manifest ก่อน")
            log.close()
            sys.exit(1)
        det_dir = BASE / "data/silver/detections"
        cmd = [
            py, "detect_vehicles.py",
            "--manifest", str(PATHS["manifest"]),
            "--out", str(det_dir),
            "--every-minutes", str(args.every_minutes),
            "--model", args.model,
            "--imgsz", str(args.imgsz),
            "--conf", str(args.conf),
            "--device", args.device,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.with_person:
            cmd.append("--with-person")

        merge_cmd = [
            py, "detect_vehicles.py",
            "--out", str(det_dir),
            "--merge-to", str(PATHS["counts"]),
            "--manifest", str(PATHS["manifest"]),
        ]
        if args.dry_run:
            log("\n[dry-run] " + " ".join(cmd))
            log("[dry-run] " + " ".join(merge_cmd))
            mark("detect", "dry-run")
        else:
            ok, dur = run_stage("detect:yolo", cmd, log)
            if not ok:
                log("\n[x] YOLO ล้มเหลว — ตรวจว่าได้ติดตั้ง ultralytics/torch แล้ว")
                log.close()
                sys.exit(1)
            ok2, dur2 = run_stage("detect:merge", merge_cmd, log)
            mark("detect", "ok" if ok2 else "failed", dur + dur2)
            if not ok2:
                log("\n[x] รวมผลลัพธ์ล้มเหลว — หยุด pipeline")
                log.close()
                sys.exit(1)

    # ------------------------------------------------------------ 3. gold (+QC)
    if "gold" in skip:
        mark("gold", "skipped")
        mark("qc", "skipped")
    else:
        if not PATHS["counts"].exists() and not args.dry_run:
            log(f"\n[x] ไม่พบ {PATHS['counts']} — รัน stage detect ก่อน หรือใส่ --skip detect ถ้ามีไฟล์แล้ว")
            log.close()
            sys.exit(1)
        cmd = [py, "build_gold.py", "--counts", str(PATHS["counts"]), "--out", str(PATHS["gold"])]
        if args.dry_run:
            log("\n[dry-run] " + " ".join(cmd))
            mark("gold", "dry-run")
        else:
            ok, dur = run_stage("gold", cmd, log)
            mark("gold", "ok" if ok else "failed", dur)
            mark("qc", "ok" if ok else "failed", 0.0)
            if not ok:
                log("\n[x] gold layer ล้มเหลว — หยุด pipeline")
                log.close()
                sys.exit(1)

    # ------------------------------------------------------------ 4. weather (optional)
    if "weather" in skip:
        mark("weather", "skipped")
    else:
        cmd = [py, "fetch_weather.py", "--gold", str(PATHS["gold"])]
        if args.weather_max_cameras:
            cmd += ["--max-cameras", str(args.weather_max_cameras)]
        if args.dry_run:
            log("\n[dry-run] " + " ".join(cmd))
            mark("weather", "dry-run")
        else:
            ok, dur = run_stage("weather", cmd, log)
            mark("weather", "ok" if ok else "failed", dur)
            if not ok:
                if args.strict:
                    log("\n[x] weather ล้มเหลว (โหมด --strict) — หยุด pipeline")
                    log.close()
                    sys.exit(1)
                log("\n[!] weather ล้มเหลว — ข้ามไปทำ QC ต่อ (gold ยังใช้ได้ แค่ไม่มีข้อมูลฝน)")

    # ------------------------------------------------------------ 5. สรุป + QC
    log("")
    log("=" * 72)
    log("สรุปผล")
    log("=" * 72)

    for name in STAGES:
        status, dur = results.get(name, ("-", 0.0))
        icon = {"ok": "✓", "skipped": "–", "dry-run": "·", "failed": "✗"}.get(status, "?")
        log(f"  {icon} {name:<9} {status:<8} {dur:6.1f}s")

    gold = PATHS["gold"]
    checks = [
        ("manifest", PATHS["manifest"]),
        ("counts", PATHS["counts"]),
        ("gold/camera_hour", gold / "camera_hour.parquet"),
        ("gold/hotspot", gold / "hotspot.parquet"),
        ("gold/qc_camera", gold / "qc_camera.parquet"),
        ("gold/rain_effect", gold / "rain_effect.parquet"),
    ]
    log("\nไฟล์ผลลัพธ์:")
    for label, path in checks:
        if path.exists():
            log(f"  ✓ {label:<18} {parquet_rows(path):>8,} แถว  {path}")
        else:
            log(f"  – {label:<18} {'':>8}  {path} (ยังไม่มี)")

    qc_path = gold / "qc_camera.parquet"
    if qc_path.exists() and not args.dry_run:
        qc = pd.read_parquet(qc_path)
        counts = qc["status"].value_counts().to_dict()
        log("\nQC ข้อมูล:")
        log("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        bad = qc[qc["status"].isin(["critical", "no_data", "warn"])].sort_values("coverage_pct").head(5)
        if len(bad):
            log("  กล้องที่ควรตรวจ:")
            for row in bad.itertuples():
                log(f"    {row.camera_id} — {row.status} (coverage {row.coverage_pct}%, ขาด {row.missing_hours} ชม.)")

    hot_path = gold / "hotspot.parquet"
    if hot_path.exists() and not args.dry_run:
        hot = pd.read_parquet(hot_path).head(3)
        log("\nTop hotspot:")
        for row in hot.itertuples():
            peak = f"{int(row.peak_hour):02d}:00" if pd.notna(row.peak_hour) else "n/a"
            log(f"  {row.traffic_rank}. {row.camera_id} — {row.vehicles_per_frame:.1f} คัน/เฟรม, peak {peak}")

    total = time.time() - t_start
    log(f"\nเสร็จใน {total / 60:.1f} นาที | log: {log.path}")
    log.close()

    # weather เป็น stage เสริม: ล้มเหลวไม่ถือว่า pipeline ล้ม (ยกเว้นโหมด --strict ที่หยุดไปแล้ว)
    if any(status == "failed" and name != "weather" for name, (status, _) in results.items()):
        sys.exit(1)


if __name__ == "__main__":
    main()
