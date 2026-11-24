"""สร้าง gold layer จาก counts.parquet ด้วย DuckDB -> Parquet หลายตาราง

อินพุต : data/silver/counts.parquet  (จาก detect_vehicles.py)
เอาต์พุต: data/gold/
  hour_profile.parquet  กล้อง x ชั่วโมงของวัน  (heatmap / หาชั่วโมงเร่งด่วน)
  hour_series.parquet   อนุกรมเวลารายชั่วโมงทั้งเมือง (line chart)
  camera_hour.parquet   กล้อง x ชั่วโมงจริง (time slider บนแผนที่)
  daily.parquet         สรุปต่อวัน (trend + data coverage)
  camera_stats.parquet  สถิติต่อกล้อง (ฐานของ hotspot)
  hotspot.parquet       อันดับกล้องที่รถหนาแน่น + peak hour + z-score (+ lat/lon ถ้ามี)

ตัวอย่าง
--------
python build_gold.py --counts data/silver/counts.parquet --out data/gold
python build_gold.py --counts data/silver/counts.parquet --out data/gold \
        --locations data/reference/camera_locations.csv --db data/gold.duckdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# แต่ละตาราง: (ชื่อตาราง, SQL) — ทุกตารางสร้างจาก view ชื่อ counts
GOLD_TABLES: list[tuple[str, str]] = [
    (
        "hour_profile",
        """
        SELECT camera_id,
               hour(ts)                                    AS hour_of_day,
               count(*)                                    AS frames,
               round(avg(n_vehicles), 2)                   AS avg_vehicles,
               round(quantile_cont(n_vehicles, 0.95), 2)   AS p95_vehicles,
               max(n_vehicles)                             AS peak_vehicles,
               round(avg(car), 2)                          AS avg_car,
               round(avg(motorcycle), 2)                   AS avg_motorcycle,
               round(avg(bus), 2)                          AS avg_bus,
               round(avg(truck), 2)                        AS avg_truck
        FROM counts
        GROUP BY camera_id, hour_of_day
        """,
    ),
    (
        "hour_series",
        """
        SELECT date_trunc('hour', ts)              AS hour,
               count(DISTINCT camera_id)           AS cameras_reporting,
               count(*)                            AS frames,
               CAST(sum(n_vehicles) AS BIGINT)      AS vehicles,
               round(avg(n_vehicles), 2)           AS vehicles_per_frame,
               CAST(sum(car) AS BIGINT)            AS car,
               CAST(sum(motorcycle) AS BIGINT)     AS motorcycle,
               CAST(sum(bus) AS BIGINT)            AS bus,
               CAST(sum(truck) AS BIGINT)          AS truck
        FROM counts
        GROUP BY hour
        """,
    ),
    (
        "daily",
        """
        SELECT cast(ts AS DATE)                    AS day,
               count(DISTINCT camera_id)           AS cameras_reporting,
               count(*)                            AS frames,
               CAST(sum(n_vehicles) AS BIGINT)     AS vehicles,
               round(avg(n_vehicles), 2)           AS vehicles_per_frame,
               round(avg(infer_ms), 1)             AS avg_infer_ms
        FROM counts
        GROUP BY day
        """,
    ),
    (
        "camera_stats",
        """
        SELECT camera_id,
               count(*)                                    AS frames,
               CAST(sum(n_vehicles) AS BIGINT)             AS vehicles,
               round(avg(n_vehicles), 3)                   AS vehicles_per_frame,
               max(n_vehicles)                             AS peak_frame_vehicles,
               round(quantile_cont(n_vehicles, 0.95), 2)   AS p95_vehicles,
               round(stddev_samp(n_vehicles), 2)           AS sd_vehicles,
               count(DISTINCT cast(ts AS DATE))            AS days_covered,
               min(ts)                                     AS first_seen,
               max(ts)                                     AS last_seen
        FROM counts
        GROUP BY camera_id
        """,
    ),
    (
        "camera_hour",  # กล้อง x ชั่วโมงจริง -> ใช้ทำ time slider บนแผนที่
        """
        SELECT camera_id,
               date_trunc('hour', ts)              AS hour,
               count(*)                            AS frames,
               CAST(sum(n_vehicles) AS BIGINT)     AS vehicles,
               round(avg(n_vehicles), 2)           AS vehicles_per_frame,
               CAST(sum(car) AS BIGINT)            AS car,
               CAST(sum(motorcycle) AS BIGINT)     AS motorcycle
        FROM counts
        GROUP BY camera_id, hour
        """,
    ),
    (
        "hotspot",
        """
        WITH peak AS (  -- row_number เพื่อให้ tie ระหว่างชั่วโมงเลือกชั่วโมงน้อยกว่าเสมอ (ผลซ้ำได้)
            SELECT camera_id, hour_of_day AS peak_hour
            FROM (
                SELECT camera_id, hour_of_day,
                       row_number() OVER (PARTITION BY camera_id
                                          ORDER BY avg_vehicles DESC, hour_of_day) AS rn
                FROM hour_profile
            )
            WHERE rn = 1
        ),
        norm AS (
            SELECT avg(vehicles_per_frame) AS mean_v,
                   stddev_samp(vehicles_per_frame) AS sd_v,
                   sum(vehicles) AS total_v
            FROM camera_stats
        )
        SELECT c.camera_id,
               c.frames,
               c.vehicles,
               c.vehicles_per_frame,
               c.peak_frame_vehicles,
               c.p95_vehicles,
               p.peak_hour,
               round((c.vehicles_per_frame - n.mean_v) / nullif(n.sd_v, 0), 2) AS z_score,
               rank() OVER (ORDER BY c.vehicles_per_frame DESC)                 AS traffic_rank,
               round(100.0 * c.vehicles / nullif(n.total_v, 0), 2)              AS share_pct
        FROM camera_stats c
        CROSS JOIN norm n
        LEFT JOIN peak p USING (camera_id)
        """,
    ),
]


def load_locations(path: Path) -> pd.DataFrame:
    """อ่านไฟล์ตำแหน่งกล้อง รองรับทั้ง (lat, lon) และ (latitude, longitude)
    พร้อมคีย์ cam_num = ตัวเลขใน camera_id เพื่อ join กับชื่อโฟลเดอร์ เช่น 915 <-> cam_915
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cam = next((c for c in ("camera_id", "cam_id", "id", "camera") if c in df.columns), None)
    lat = next((c for c in ("latitude", "lat") if c in df.columns), None)
    lon = next((c for c in ("longitude", "lon", "lng") if c in df.columns), None)
    if not (cam and lat and lon):
        sys.exit(f"[x] {path} ต้องมีคอลัมน์ camera_id + latitude/longitude (พบ: {list(df.columns)})")

    out = pd.DataFrame(
        {
            "cam_num": df[cam].astype(str).str.extract(r"(\d+)", expand=False).fillna(""),
            "lat": pd.to_numeric(df[lat], errors="coerce"),
            "lon": pd.to_numeric(df[lon], errors="coerce"),
        }
    )
    for extra in ("location", "location_en", "ip_address", "image_count"):
        if extra in df.columns:
            out[extra] = df[extra]
    return out.dropna(subset=["lat", "lon"])


def build_qc(con: duckdb.DuckDBPyConnection, out: Path, has_loc: bool) -> None:
    """ตาราง QC: รูปที่ประมวลผลจริง vs image_count ใน cctv_info.csv + ความครบรายชั่วโมง

    แยก 2 ตัวเลขให้ชัด:
      sampled_pct  = รูปที่ประมวลผล / รูปทั้งหมดของกล้อง  (ต่ำได้เพราะเราสุ่มเฟรม)
      coverage_pct = ชั่วโมงที่มีข้อมูล / ชั่วโมงทั้งหมดในช่วงข้อมูล  (อันนี้คือ 'ข้อมูลขาด' จริง)
    """
    base = """
        WITH span AS (
            SELECT date_diff('hour', min(hour), max(hour)) + 1 AS hours_expected
            FROM camera_hour
        ),
        per_cam AS (
            SELECT camera_id,
                   count(DISTINCT hour)   AS hours_present,
                   count(*)               AS processed_images,
                   min(hour)              AS first_seen,
                   max(hour)              AS last_seen,
                   count(DISTINCT cast(hour AS DATE)) AS days_covered
            FROM camera_hour
            GROUP BY camera_id
        )
    """
    status_sql = """
        CASE WHEN p.camera_id IS NULL THEN 'no_data'
             WHEN 100.0 * COALESCE(p.hours_present, 0) / s.hours_expected >= 90 THEN 'ok'
             WHEN 100.0 * COALESCE(p.hours_present, 0) / s.hours_expected >= 60 THEN 'warn'
             ELSE 'critical' END AS status
    """

    if has_loc:
        sql = base + f"""
        SELECT COALESCE(p.camera_id, l.cam_num) AS camera_id,
               COALESCE(l.cam_num, regexp_extract(CAST(p.camera_id AS VARCHAR), '(\\d+)', 1)) AS cam_num,
               l.location,
               l.image_count                            AS expected_images,
               COALESCE(p.processed_images, 0)          AS processed_images,
               COALESCE(p.hours_present, 0)             AS hours_present,
               s.hours_expected,
               s.hours_expected - COALESCE(p.hours_present, 0) AS missing_hours,
               round(100.0 * COALESCE(p.hours_present, 0) / s.hours_expected, 1) AS coverage_pct,
               round(100.0 * COALESCE(p.processed_images, 0) / nullif(l.image_count, 0), 1) AS sampled_pct,
               p.first_seen, p.last_seen,
               COALESCE(p.days_covered, 0)              AS days_covered,
               {status_sql}
        FROM per_cam p
        FULL OUTER JOIN loc l
          ON l.cam_num = regexp_extract(CAST(p.camera_id AS VARCHAR), '(\\d+)', 1)
        CROSS JOIN span s
        ORDER BY cam_num
        """
    else:
        sql = base + f"""
        SELECT p.camera_id,
               regexp_extract(CAST(p.camera_id AS VARCHAR), '(\\d+)', 1) AS cam_num,
               CAST(NULL AS VARCHAR) AS location,
               CAST(NULL AS BIGINT)  AS expected_images,
               p.processed_images,
               p.hours_present,
               s.hours_expected,
               s.hours_expected - p.hours_present AS missing_hours,
               round(100.0 * p.hours_present / s.hours_expected, 1) AS coverage_pct,
               CAST(NULL AS DOUBLE) AS sampled_pct,
               p.first_seen, p.last_seen, p.days_covered,
               {status_sql}
        FROM per_cam p CROSS JOIN span s
        ORDER BY cam_num
        """

    con.execute(f"CREATE OR REPLACE TABLE qc_camera AS {sql}")
    con.execute(
        f"COPY (SELECT * FROM qc_camera) TO '{(out / 'qc_camera.parquet').as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    total, hours_expected = con.execute(
        "SELECT count(*), max(hours_expected) FROM qc_camera"
    ).fetchone()
    stats = con.execute(
        "SELECT status, count(*) FROM qc_camera GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    avg_cov, avg_cov_data = con.execute(
        "SELECT round(avg(coverage_pct), 1), round(avg(coverage_pct) FILTER (WHERE status <> 'no_data'), 1) "
        "FROM qc_camera"
    ).fetchone()
    worst = con.execute(
        "SELECT camera_id, coverage_pct, missing_hours FROM qc_camera "
        "WHERE status IN ('warn','critical') ORDER BY coverage_pct LIMIT 5"
    ).fetchall()

    print(f"  ✓ {out / 'qc_camera.parquet'}  ({total:,} กล้อง)")
    print(f"     QC: " + ", ".join(f"{s}={n}" for s, n in stats)
          + f" | coverage {avg_cov}% (เฉพาะกล้องที่มีข้อมูล {avg_cov_data}%)"
          + f" | ช่วงข้อมูล {hours_expected} ชั่วโมง")
    if worst:
        print("     กล้องที่ข้อมูลขาดมากสุด: " + ", ".join(
            f"{c} ({cov}%, ขาด {int(mh)} ชม.)" for c, cov, mh in worst))


def main() -> None:
    ap = argparse.ArgumentParser(description="สร้าง gold layer จาก counts.parquet")
    ap.add_argument("--counts", type=Path, default=Path("data/silver/counts.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/gold"))
    ap.add_argument("--locations", type=Path, default=None,
                    help="CSV ตำแหน่งกล้อง (camera_id + latitude/longitude) — ค่าเริ่มต้นใช้ cctv_info.csv ถ้ามี")
    ap.add_argument("--db", type=Path, default=None, help="เก็บตารางไว้ในไฟล์ .duckdb เพื่อ query ต่อ")
    args = ap.parse_args()

    if not args.counts.exists():
        sys.exit(f"[x] ไม่พบ {args.counts} — รัน detect_vehicles.py ก่อน")

    if args.locations is None and Path("cctv_info.csv").exists():
        args.locations = Path("cctv_info.csv")

    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.db)) if args.db else duckdb.connect()

    src = args.counts.as_posix()
    con.execute(f"CREATE OR REPLACE VIEW counts AS SELECT * FROM read_parquet('{src}')")

    cols = [r[0] for r in con.execute("DESCRIBE counts").fetchall()]
    need = {"camera_id", "ts", "n_vehicles", "car", "motorcycle", "bus", "truck", "infer_ms"}
    missing = need - set(cols)
    if missing:
        sys.exit(f"[x] counts ขาดคอลัมน์: {sorted(missing)} (มี: {cols})")

    rows = con.execute("SELECT count(*), count(DISTINCT camera_id), min(ts), max(ts) FROM counts").fetchone()
    print(f"[i] counts: {rows[0]:,} แถว | {rows[1]} กล้อง | {rows[2]} -> {rows[3]}")
    if rows[0] == 0:
        sys.exit("[x] counts ว่าง")

    for name, sql in GOLD_TABLES:
        con.execute(f"CREATE OR REPLACE TABLE {name} AS {sql}")

    # hotspot บนแผนที่: เติม lat/lon ถ้าผู้ใช้ให้ไฟล์ตำแหน่งกล้องมา
    if args.locations:
        if not args.locations.exists():
            sys.exit(f"[x] ไม่พบ {args.locations}")
        loc = load_locations(args.locations)
        con.register("loc_df", loc)
        con.execute("CREATE OR REPLACE VIEW loc AS SELECT * FROM loc_df")
        con.execute(
            """CREATE OR REPLACE TABLE hotspot AS
               SELECT h.*, l.lat, l.lon,
                      round(l.lat, 2) AS lat_grid,   -- ~1.1 กม. ไว้จับกลุ่ม hotspot บนแผนที่
                      round(l.lon, 2) AS lon_grid,
                      l.location, l.location_en, l.image_count AS expected_images
               FROM hotspot h
               LEFT JOIN loc l ON regexp_extract(CAST(h.camera_id AS VARCHAR), '(\\d+)', 1) = l.cam_num"""
        )
        con.execute(
            """CREATE OR REPLACE TABLE camera_hour AS
               SELECT c.*, l.lat, l.lon, l.location, l.location_en
               FROM camera_hour c
               LEFT JOIN loc l ON regexp_extract(CAST(c.camera_id AS VARCHAR), '(\\d+)', 1) = l.cam_num"""
        )

        matched = con.execute("SELECT count(lat) FROM hotspot").fetchone()[0]
        print(f"[i] hotspot ที่จับคู่พิกัดได้: {matched}/{rows[1]} กล้อง (จาก {args.locations.name})")
        if matched < rows[1]:
            miss = con.execute(
                "SELECT camera_id FROM hotspot WHERE lat IS NULL ORDER BY camera_id LIMIT 8"
            ).fetchall()
            print("[!] กล้องที่ไม่มีพิกัด: " + ", ".join(str(m[0]) for m in miss)
                  + (" ..." if rows[1] - matched > len(miss) else ""))

    build_qc(con, args.out, has_loc=bool(args.locations))

    # เขียน Parquet
    for name, _ in GOLD_TABLES:
        target = (args.out / f"{name}.parquet").as_posix()
        con.execute(f"COPY (SELECT * FROM {name}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"  ✓ {args.out / f'{name}.parquet'}  ({n:,} แถว)")

    print("\nTop 5 hotspot (รถหนาแน่นสุดต่อเฟรม):")
    top = con.execute(
        """SELECT traffic_rank, camera_id, vehicles_per_frame, peak_hour, z_score
           FROM hotspot ORDER BY traffic_rank LIMIT 5"""
    ).df()
    print(top.to_string(index=False))

    busiest = con.execute(
        "SELECT hour_of_day, round(sum(avg_vehicles * frames) / sum(frames), 2) AS v "
        "FROM hour_profile GROUP BY 1 ORDER BY v DESC, hour_of_day LIMIT 3"
    ).fetchall()
    print("\nชั่วโมงที่รถหนาแน่นสุด (เฉลี่ยต่อเฟรม): " + ", ".join(f"{h:02d}:00 = {v}" for h, v in busiest))

    if args.db:
        print(f"\n[i] ตารางถูกเก็บใน {args.db} ด้วย — เปิด query ต่อได้")
    con.close()


if __name__ == "__main__":
    main()
