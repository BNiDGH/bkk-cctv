"""ดึงข้อมูลอากาศย้อนหลังจาก Open-Meteo ตามพิกัดกล้อง แล้ว join เข้า gold layer

ทำ 3 อย่าง:
  1. ดึงอากาศรายชั่วโมงของทุกพิกัดกล้อง (cache ลงดิสก์ ไม่ยิงซ้ำ)
  2. เขียน data/silver/weather_camera_hour.parquet
  3. join เข้า data/gold/camera_hour.parquet + hour_series.parquet
     และสร้าง data/gold/rain_effect.parquet (ฝนตกแล้วรถติดขึ้นกี่ %)

ตัวอย่าง
--------
python fetch_weather.py                                   # ใช้วันที่จาก gold layer อัตโนมัติ
python fetch_weather.py --start 2026-11-20 --end 2026-11-22
python fetch_weather.py --no-join                         # ดึงเก็บไว้ก่อน ไม่แตะ gold

API: https://open-meteo.com/en/docs/historical-weather-api (ฟรี ไม่ต้องใช้ key)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "rain",
    "relative_humidity_2m",
    "wind_speed_10m",
    "weather_code",
    "visibility",
]
RAIN_MM_THRESHOLD = 0.1  # mm/ชม. — ต่ำกว่านี้ถือว่าไม่ตก

# WMO weather code -> ข้อความสั้น ๆ ไว้โชว์บน dashboard
WEATHER_TH = {
    0: "ท้องฟ้าแจ่มใส", 1: "แดดจัดเป็นส่วนใหญ่", 2: "มีเมฆบางส่วน", 3: "เมฆมาก",
    45: "หมอก", 48: "หมอกน้ำค้างแข็ง", 51: "ฝนปรอย", 53: "ฝนปรอยปานกลาง", 55: "ฝนปรอยหนัก",
    61: "ฝนเล็กน้อย", 63: "ฝนปานกลาง", 65: "ฝนหนัก",
    80: "ฝนตกเป็นช่วง ๆ", 81: "ฝนตกเป็นช่วง ๆ ปานกลาง", 82: "ฝนตกหนักเป็นช่วง ๆ",
    95: "พายุฝนฟ้าคะนอง", 96: "พายุฝนฟ้าคะนองกับลูกเห็บ", 99: "พายุฝนฟ้าคะนองรุนแรง",
}


def load_cameras(path: Path) -> pd.DataFrame:
    """คืนตาราง camera_id, lat, lon (พิกัดปัด 4 ตำแหน่ง เพื่อใช้ cache ร่วมกัน)"""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cam = next((c for c in ("camera_id", "cam_id", "id") if c in df.columns), None)
    lat = next((c for c in ("latitude", "lat") if c in df.columns), None)
    lon = next((c for c in ("longitude", "lon", "lng") if c in df.columns), None)
    if not (cam and lat and lon):
        sys.exit(f"[x] {path} ต้องมี camera_id + latitude/longitude (พบ: {list(df.columns)})")

    out = pd.DataFrame(
        {
            "camera_id": df[cam].astype(str).str.strip(),
            "lat": pd.to_numeric(df[lat], errors="coerce").round(4),
            "lon": pd.to_numeric(df[lon], errors="coerce").round(4),
        }
    ).dropna(subset=["lat", "lon"])
    return out.drop_duplicates(subset=["camera_id"]).reset_index(drop=True)


def fetch(lat: float, lon: float, start: str, end: str, tz: str, cache_dir: Path,
          retries: int = 3) -> dict:
    key = f"{lat:.4f}_{lon:.4f}_{start}_{end}.json"
    cached = cache_dir / key
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": tz,
        }
    )
    url = f"{API}?{params}"

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "hourly" not in data:
                raise RuntimeError(data.get("reason", str(data)[:200]))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(data), encoding="utf-8")
            return data
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"ดึงข้อมูลไม่สำเร็จ ({lat},{lon}): {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def to_frame(data: dict, camera_id: str) -> pd.DataFrame:
    h = data["hourly"]
    df = pd.DataFrame(
        {
            "camera_id": camera_id,
            "hour": pd.to_datetime(h["time"]),
            "temp_c": h.get("temperature_2m"),
            "precipitation_mm": h.get("precipitation"),
            "rain_mm": h.get("rain"),
            "humidity_pct": h.get("relative_humidity_2m"),
            "wind_kmh": h.get("wind_speed_10m"),
            "weather_code": h.get("weather_code"),
            "visibility_m": h.get("visibility"),
        }
    )
    for c in ("temp_c", "precipitation_mm", "rain_mm", "humidity_pct", "wind_kmh", "visibility_m"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["weather_code"] = df["weather_code"].astype("Int64")
    df["is_rain"] = df["precipitation_mm"].fillna(0) >= RAIN_MM_THRESHOLD
    df["weather_th"] = df["weather_code"].map(WEATHER_TH)
    return df


def join_gold(gold: Path, weather: pd.DataFrame) -> None:
    cam_hour_path = gold / "camera_hour.parquet"
    if not cam_hour_path.exists():
        print(f"[i] ยังไม่มี {cam_hour_path} — ข้ามการ join (รัน build_gold.py ก่อน)")
        return

    ch = pd.read_parquet(cam_hour_path)
    ch["hour"] = pd.to_datetime(ch["hour"])
    w = weather.copy()
    w["hour"] = pd.to_datetime(w["hour"])

    weather_cols = [c for c in w.columns if c not in ("camera_id", "hour")]
    ch = ch.drop(columns=[c for c in weather_cols if c in ch.columns])  # รันซ้ำได้
    merged = ch.merge(w, on=["camera_id", "hour"], how="left")
    # ชั่วโมงที่ไม่มีอากาศ -> is_rain = False (ไม่ใช่ NA) เพื่อให้ mask/คำนวณต่อได้
    merged["is_rain"] = merged["is_rain"].astype("boolean").fillna(False).astype(bool)
    merged.to_parquet(cam_hour_path, index=False)

    matched = int(merged["precipitation_mm"].notna().sum())
    print(f"[i] join weather เข้า camera_hour: {matched:,}/{len(merged):,} แถว "
          f"({100 * matched / max(len(merged), 1):.1f}%)")
    if matched < len(merged):
        print("[!] บางชั่วโมงไม่มีอากาศ (ช่วงวันที่ไม่ตรง หรือ Open-Meteo ยังไม่มีข้อมูลล่าสุด ~5 วัน)")

    # hour_series: เติมฝนระดับเมือง (เฉลี่ยทุกกล้อง) ไว้ทำกราฟเส้น
    series_path = gold / "hour_series.parquet"
    if series_path.exists():
        city = (
            w.groupby("hour", as_index=False)
            .agg(precipitation_mm=("precipitation_mm", "mean"),
                 temp_c=("temp_c", "mean"),
                 rain_hours_share=("is_rain", "mean"))
        )
        hs = pd.read_parquet(series_path)
        hs["hour"] = pd.to_datetime(hs["hour"])
        hs = hs.drop(columns=[c for c in city.columns if c in hs.columns and c != "hour"])
        hs.merge(city, on="hour", how="left").to_parquet(series_path, index=False)

    # ผลกระทบของฝนต่อความหนาแน่นรถ
    per_cam = (
        merged.groupby(["camera_id", "is_rain"])["vehicles_per_frame"]
        .mean()
        .unstack()
        .rename(columns={False: "dry_vpf", True: "rain_vpf"})
    )
    hours = merged.groupby("camera_id")["is_rain"].agg(rain_hours="sum", total_hours="count")
    eff = per_cam.join(hours).reset_index()
    for col in ("dry_vpf", "rain_vpf"):
        if col not in eff.columns:
            eff[col] = pd.NA
    eff["delta_pct"] = ((eff["rain_vpf"] / eff["dry_vpf"]) - 1) * 100
    eff = eff[["camera_id", "rain_hours", "total_hours", "dry_vpf", "rain_vpf", "delta_pct"]]

    # แถวสรุประดับเมือง
    rain_mask = merged["is_rain"].to_numpy(dtype=bool)
    city_dry = merged.loc[~rain_mask, "vehicles_per_frame"].mean()
    city_rain = merged.loc[rain_mask, "vehicles_per_frame"].mean()
    summary = pd.DataFrame(
        [
            {
                "camera_id": "ALL",
                "rain_hours": int(merged.loc[rain_mask, "hour"].nunique()),
                "total_hours": int(merged["hour"].nunique()),
                "dry_vpf": city_dry,
                "rain_vpf": city_rain,
                "delta_pct": (city_rain / city_dry - 1) * 100 if city_dry else None,
            }
        ]
    )
    eff = pd.concat([eff, summary], ignore_index=True).sort_values("delta_pct", ascending=False)
    eff.to_parquet(gold / "rain_effect.parquet", index=False)
    print(f"  ✓ {gold / 'rain_effect.parquet'}  ({len(eff):,} แถว)")

    if city_dry and pd.notna(city_rain):
        change = (city_rain / city_dry - 1) * 100
        print(f"\n[i] ภาพรวม: ฝนตกแล้วรถหนาแน่น {'เพิ่ม' if change >= 0 else 'ลด'} "
              f"{abs(change):.1f}% ({city_dry:.1f} -> {city_rain:.1f} คัน/เฟรม)")


def main() -> None:
    ap = argparse.ArgumentParser(description="ดึงอากาศย้อนหลัง Open-Meteo แล้ว join เข้า gold layer")
    ap.add_argument("--locations", type=Path, default=Path("cctv_info.csv"))
    ap.add_argument("--gold", type=Path, default=Path("data/gold"))
    ap.add_argument("--out", type=Path, default=Path("data/silver/weather_camera_hour.parquet"))
    ap.add_argument("--cache", type=Path, default=Path("data/bronze/weather_cache"))
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (ค่าเริ่มต้น: จาก gold/camera_hour)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (ค่าเริ่มต้น: จาก gold/camera_hour)")
    ap.add_argument("--timezone", default="Asia/Bangkok")
    ap.add_argument("--max-cameras", type=int, default=0, help="จำกัดจำนวนกล้อง (ไว้ลอง)")
    ap.add_argument("--sleep", type=float, default=0.2, help="หน่วงระหว่าง request (วินาที)")
    ap.add_argument("--no-join", action="store_true", help="ดึงเก็บเป็น Parquet อย่างเดียว ไม่แตะ gold")
    args = ap.parse_args()

    if not args.locations.exists():
        sys.exit(f"[x] ไม่พบ {args.locations}")

    cams = load_cameras(args.locations)
    if args.max_cameras:
        cams = cams.head(args.max_cameras)

    # ช่วงวันที่: ใช้จาก gold layer ถ้าไม่ได้ระบุมา
    start, end = args.start, args.end
    if not (start and end):
        ch_path = args.gold / "camera_hour.parquet"
        if not ch_path.exists():
            sys.exit("[x] ไม่ได้ระบุ --start/--end และไม่พบ gold/camera_hour.parquet ให้เดาวันที่")
        hours = pd.to_datetime(pd.read_parquet(ch_path, columns=["hour"])["hour"])
        start = start or hours.min().strftime("%Y-%m-%d")
        end = end or hours.max().strftime("%Y-%m-%d")
    print(f"[i] {len(cams)} กล้อง | {start} -> {end} | tz={args.timezone}")

    frames = []
    for i, row in enumerate(cams.itertuples(index=False), 1):
        try:
            data = fetch(row.lat, row.lon, start, end, args.timezone, args.cache)
        except RuntimeError as exc:
            print(f"  [!] {row.camera_id}: {exc}")
            continue
        frames.append(to_frame(data, row.camera_id))
        if i % 10 == 0 or i == len(cams):
            print(f"  {i}/{len(cams)} กล้อง", flush=True)
        time.sleep(args.sleep)

    if not frames:
        sys.exit("[x] ไม่ได้ข้อมูลเลย — ตรวจอินเทอร์เน็ต/ช่วงวันที่ (archive ต้องเป็นอดีต > 5 วัน)")

    weather = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(args.out, index=False)

    rain_share = 100 * weather["is_rain"].mean()
    print(f"\n[ok] {args.out}  ({len(weather):,} แถว, {weather['camera_id'].nunique()} กล้อง)")
    print(f"     ชั่วโมงที่ฝนตก {rain_share:.1f}% | ฝนรวมสูงสุด {weather['precipitation_mm'].max():.1f} มม./ชม.")

    if not args.no_join:
        join_gold(args.gold, weather)


if __name__ == "__main__":
    main()
