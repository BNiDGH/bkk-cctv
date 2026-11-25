# Bangkok CCTV → Structured Data

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/detection-YOLOv8-00BFFF)](https://docs.ultralytics.com/)
[![DuckDB](https://img.shields.io/badge/query-DuckDB-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/BNiDGH/bkk-cctv/actions/workflows/ci.yml/badge.svg)](https://github.com/BNiDGH/bkk-cctv/actions/workflows/ci.yml)

แปลงภาพจากกล้อง CCTV ในกรุงเทพฯ (ข้อมูลดิบที่ไม่มีโครงสร้าง) ให้เป็นตารางที่วิเคราะห์ได้
แล้วเล่าเป็นแผนที่ hotspot + dashboard

---

โปรเจกต์วิชา 2110403 (Bangkok CCTV Image Analysis, 15%)
ทำ pipeline ตั้งแต่รูปดิบจนถึง dashboard: **รูป → manifest → ตารางนับรถ → gold → แผนที่**

```
รูป CCTV ──build_manifest.py──> manifest.parquet ──detect_vehicles.py──> counts.parquet ──build_gold.py──> gold/*.parquet ──> dashboard.py
           (camera_id, ts)         (bronze)              (YOLO นับรถ)         (silver)      (DuckDB)      (Streamlit)
                                                                      Open-Meteo ──fetch_weather.py──┘ (join ฝน/อากาศ)
```

## โครงสร้างโปรเจกต์

```
bkk-cctv/
├─ build_manifest.py        # 1) รูป → manifest.parquet (bronze)
├─ detect_vehicles.py       # 2) YOLO นับรถ → counts.parquet (silver)
├─ build_gold.py            # 3) DuckDB → gold/*.parquet + QC
├─ fetch_weather.py         # 4) Open-Meteo → join ฝนเข้า gold
├─ dashboard.py             # 5) Streamlit + pydeck
├─ run_pipeline.py          # รันทั้งหมดต่อกัน + log
├─ cctv_info.example.csv    # แม่แบบ metadata กล้อง (ไฟล์จริงไม่ commit)
├─ scripts/
│  ├─ make_sample_counts.py # ข้อมูลตัวอย่างสำหรับ CI/ทดสอบ
│  └─ test_dashboard.py     # ทดสอบ dashboard ด้วย AppTest
├─ .github/workflows/ci.yml # CI: syntax → gold → dashboard
├─ cctv_images/             # (ไม่ commit) รูปจริง 100 กล้อง ~61k รูป (~1.6 GB)
└─ data/                    # (ไม่ commit) bronze/ silver/ gold/
```

**เปิด repo แล้วอ่านตรงไหน:**

| อยากทำอะไร | ไปที่ |
|---|---|
| เข้าใจภาพรวม | ไดอะแกรมด้านบน + หัวข้อ "ขั้นที่ 1–6" |
| รันทั้งหมด | `run_pipeline.py` (หัวข้อ "รันทั้ง pipeline ทีเดียว") |
| ดูข้อมูลที่ได้ | หัวข้อ "ขั้นที่ 4" (ตาราง gold) |
| แก้/ต่อยอด | `CONTRIBUTING.md` |

## ติดตั้ง

```bash
python -m pip install -U pandas pyarrow pillow duckdb streamlit pydeck ultralytics
```
> `ultralytics` จะดึง `torch` มาด้วย (~2 GB) — ติดตั้งครั้งเดียว ถ้าเครื่องมี GPU จะเร็วขึ้นมาก
> ตรวจแล้วว่า `build_manifest.py` ใช้กับรูปจริงได้ — manifest ของชุดข้อมูลนี้มี **61,319 รูป จาก 100 กล้อง** (25–31 ส.ค. 2026) และได้ `ts_source = filename` ครบทุกรูป

**ข้อมูลกล้อง**: ไฟล์ `cctv_info.csv` (camera_id, พิกัด, ชื่อสถานที่, ip, image_count) **ไม่ถูก commit** เพราะมี IP ภายใน
ให้คัดลอกไฟล์ตัวอย่างแล้วเติมข้อมูลจริง:

```bash
cp cctv_info.example.csv cctv_info.csv   # แล้วแก้ข้อมูลให้ตรงกับกล้องของคุณ
```

## ขั้นที่ 1 — สร้าง manifest

```bash
python build_manifest.py --images cctv_images/cctv_images --out data/bronze/manifest.parquet --probe-size --csv data/bronze/manifest.csv
```

รูปจริงอยู่ใน `cctv_images/cctv_images/` (100 กล้อง, 61,319 รูป, 25–31 ส.ค. 2026) — โฟลเดอร์ = `camera_id` และชื่อไฟล์มีเวลาเสมอ ถ้า clone ใหม่ต้องวางโฟลเดอร์รูปเองเพราะไม่ถูก commit:

```
cctv_images/cctv_images/
  1028/1028_แยกสะพานแดง_แยกสะพานแดง_20260825_061012.jpg
  1028/1028_แยกสะพานแดง_แยกสะพานแดง_20260825_062100.jpg
  1029/...
```

| คอลัมน์ | ความหมาย |
|---|---|
| `camera_id` | จากชื่อโฟลเดอร์แม่ (`1028`) ถ้าไม่มีจะเดาจากชื่อไฟล์ |
| `path`, `rel_path`, `file_name` | ที่อยู่ไฟล์ |
| `ts` | เวลา: ชื่อไฟล์ → EXIF → mtime (ตามลำดับ) |
| `ts_source` | บอกว่ามาจากไหน — **สำคัญมาก ใช้ QC** |
| `size_bytes`, `width`, `height` | ไว้เช็คไฟล์เสีย/ความละเอียดไม่เท่ากัน |

รูปแบบชื่อไฟล์ที่รองรับ: `20261124_083000`, `2026-11-24_08-30-00`, `20261124083000`, epoch (วินาที/มิลลิวินาที) — ชื่อไฟล์จริงมีชื่อสถานที่ภาษาไทยนำหน้าได้ เช่น `1028_แยกสะพานแดง_..._20260825_061012.jpg` (ตัวดึงเวลาจะหาเฉพาะส่วนวันที่)

ตอนจบสคริปต์จะรายงาน: จำนวนรูปต่อกล้อง, ช่วงเวลา, และ **จำนวนรูปที่หา timestamp ไม่ได้** — ถ้าตัวเลขนี้เยอะ แปลว่า dataset มีปัญหา ต้องแก้ตั้งแต่ขั้น scrape (ดูหัวข้อท้ายไฟล์)

## ขั้นที่ 2 — นับรถด้วย YOLO

```bash
# ลอง 200 รูปก่อน ว่าตั้งค่าถูกและโมเดลโหลดได้
python detect_vehicles.py --manifest data/bronze/manifest.parquet --out data/silver/detections --every-minutes 10 --limit 200

# รันเต็ม (รันซ้ำได้ ทำต่อจากเดิมอัตโนมัติ)
python detect_vehicles.py --manifest data/bronze/manifest.parquet --out data/silver/detections --every-minutes 10 --model yolov8n.pt --imgsz 640
```

- **`--every-minutes 10`** = สุ่ม 1 เฟรมต่อ 10 นาทีต่อกล้อง → 100 กล้อง × 144 เฟรม/วัน ≈ **14,400 รูป** แทนที่จะเป็น 60,000+ (ประหยัดเวลา ~4 เท่า และยังพอสำหรับกราฟรายชั่วโมง) ใส่ `0` ถ้าอยากได้ทุกรูป
- **resume**: ผลถูกเขียนเป็น `part-*.parquet` ทุก 500 รูป ถ้าโปรแกรมหลุด/ปิดเครื่อง รันคำสั่งเดิมต่อได้ ไม่เริ่มใหม่
- คลาสที่ใช้ (COCO pretrained): `car`, `motorcycle`, `bus`, `truck` + `person` ถ้าใส่ `--with-person`
- `--device 0` ถ้ามี GPU, `--model yolov8s.pt` ถ้าอยากแม่นขึ้น (ช้ากว่า)
- ถ้าขึ้น `DefaultCPUAllocator: not enough memory` (RAM เหลือน้อย) ให้ลด `--batch 1` และ `--imgsz 416` — ค่าเริ่มต้นใช้ `--batch 2` เพราะ batch 8 กินแรมเกินบนเครื่อง 16 GB

## ขั้นที่ 3 — รวมเป็นไฟล์เดียว แล้วใช้ต่อ

```bash
python detect_vehicles.py --out data/silver/detections --merge-to data/silver/counts.parquet --manifest data/bronze/manifest.parquet
```

> ต้องรันขั้นที่ 2 ให้มี `part-*.parquet` ใน `data/silver/detections/` ก่อน ไม่งั้นจะขึ้น `[x] ไม่พบ part`

ตาราง `counts.parquet`: `camera_id, ts, car, motorcycle, bus, truck, person, n_vehicles, infer_ms, path`

## ขั้นที่ 4 — gold layer ด้วย DuckDB

```bash
python build_gold.py --counts data/silver/counts.parquet --out data/gold

# เก็บตารางไว้ในไฟล์ .duckdb เพื่อ query ต่อ / ต่อ dashboard
python build_gold.py --counts data/silver/counts.parquet --out data/gold --db data/gold.duckdb
```

สร้างตาราง Parquet 5 ตัวใน `data/gold/`:

| ไฟล์ | เนื้อหา | เอาไปใช้ทำอะไร |
|---|---|---|
| `hour_profile.parquet` | กล้อง × ชั่วโมงของวัน (0–23) | heatmap กล้อง×ชั่วโมง, หาชั่วโมงเร่งด่วนของแต่ละกล้อง |
| `hour_series.parquet` | อนุกรมเวลารายชั่วโมงทั้งเมือง + แยกคลาสรถ | line chart ภาพรวม, ดูผลของฝน/วันหยุด |
| `daily.parquet` | สรุปต่อวัน (frames, vehicles, infer_ms) | trend รายวัน + data coverage |
| `camera_stats.parquet` | สถิติต่อกล้อง (avg, p95, sd, ช่วงเวลาที่มีข้อมูล) | ฐานของ hotspot / QC กล้อง |
| `hotspot.parquet` | อันดับกล้อง + `peak_hour` + `z_score` + `share_pct` + lat/lon | แผนที่ hotspot, เล่าเรื่อง "กล้องไหนหนักสุด" |
| `qc_camera.parquet` | QC ต่อกล้อง: `expected_images` vs `processed_images`, `hours_present/expected`, `coverage_pct`, `status` | เช็คว่ากล้องตัวไหนข้อมูลขาด ก่อนเอาไปวิเคราะห์ |

### QC ข้อมูล (`qc_camera.parquet`)

ทุกครั้งที่รัน `build_gold.py` จะได้ตาราง QC ของ **ทุกกล้องใน `cctv_info.csv`** (ไม่เฉพาะกล้องที่มีข้อมูล) พร้อม 2 ตัวเลขที่ต่างกัน:

- **`sampled_pct`** = รูปที่ประมวลผล / `image_count` — ต่ำเป็นเรื่องปกติถ้าใช้ `--every-minutes 10` (เช่น 48/613 = 7.8%)
- **`coverage_pct`** = ชั่วโมงที่มีข้อมูล / ชั่วโมงทั้งหมดในช่วงข้อมูล — **ตัวนี้แหละที่บอกว่าข้อมูลขาดจริง**

`status`: `ok` (≥90%) · `warn` (60–89%) · `critical` (<60%) · `no_data` (ไม่ถูกประมวลผลเลย)

```python
import pandas as pd
qc = pd.read_parquet("data/gold/qc_camera.parquet")
qc[qc.status != "ok"].sort_values("coverage_pct").head(10)   # กล้องที่ต้องตรวจก่อน
```

- **พิกัดกล้อง**: ถ้ามี `cctv_info.csv` (คอลัมน์ `camera_id, latitude, longitude, location`) สคริปต์จะใช้ไฟล์นั้นอัตโนมัติ แล้วเติม `lat, lon, lat_grid, lon_grid, location, location_en, expected_images` ให้ตาราง hotspot
- join กล้องด้วย **ตัวเลขในชื่อ** → `915`, `cam_915`, `CCTV-915` จับคู่กันได้หมด (ถ้าจับไม่ได้จะพิมพ์รายชื่อกล้องที่ขาดพิกัดให้)
- `lat_grid/lon_grid` = ปัดพิกัดทศนิยม 2 ตำแหน่ง (~1.1 กม.) ไว้จับกลุ่ม hotspot บนแผนที่
- ถ้าใส่ `--db` ตารางจะถูกเก็บใน `.duckdb` ด้วย เปิด query ต่อได้ทันที

```python
import duckdb
con = duckdb.connect("data/gold.duckdb")
con.sql("SELECT camera_id, location, peak_hour, vehicles_per_frame FROM hotspot LIMIT 5").df()
```

จากตรงนี้คือส่วนที่ให้คะแนน Data Science: join กับฝน (Open-Meteo) / Traffy Fondue / วันหยุด แล้วหา hotspot, ชั่วโมงเร่งด่วน, ความสัมพันธ์ฝน–รถติด

## ขั้นที่ 5 — Dashboard (Streamlit + pydeck)

```bash
streamlit run dashboard.py
```

เปิด http://localhost:8501 ประกอบด้วย:

- **แผนที่ hotspot** บน pydeck — วงกลม (Scatter) หรือแท่ง 3D (Column), ขนาด/สีตามความหนาแน่น, tooltip บอกชื่อกล้อง/สถานที่/อันดับ
- **Time slider** ช่วงชั่วโมงของวัน + เลือกวันที่ + ตัวชี้วัด (คัน/เฟรม หรือ ปริมาณรวม) — กรองแล้วตัวเลข KPI/แผนที่เปลี่ยนตามทันที
- **KPI**: รถทั้งหมด, กล้องที่มีข้อมูล, เฉลี่ยต่อเฟรม, ชั่วโมงที่หนาแน่นสุด
- **Top N กล้อง** พร้อม progress bar
- **กราฟรายชั่วโมง** (rush hour), **อนุกรมเวลา**, และ **heatmap กล้อง × ชั่วโมง**
- **สรุป insight** 3 อันดับจาก gold layer (ใช้ในสไลด์/วิดีโอได้เลย)

หมายเหตุ: ถ้าติดตั้ง `matplotlib` ด้วย heatmap จะมีสีไล่ระดับ (ถ้าไม่มีก็ยังแสดงเป็นตารางตัวเลข)

## ขั้นที่ 6 — ข้อมูลอากาศ/ฝน (Open-Meteo)

```bash
python fetch_weather.py            # ใช้วันที่จาก gold/camera_hour อัตโนมัติ
python fetch_weather.py --start 2026-11-20 --end 2026-11-22
python fetch_weather.py --no-join  # ดึงเก็บ Parquet อย่างเดียว ยังไม่แตะ gold
```

ทำอะไร:
1. ดึงอากาศรายชั่วโมงของทุกพิกัดกล้อง (`temperature_2m, precipitation, rain, humidity, wind, weather_code, visibility`) — **cache ลง `data/bronze/weather_cache/`** รันซ้ำไม่ยิง API ใหม่
2. เขียน `data/silver/weather_camera_hour.parquet`
3. join เข้า gold:
   - `camera_hour.parquet` + `precipitation_mm, rain_mm, is_rain, temp_c, weather_th, visibility_m, ...`
   - `hour_series.parquet` + ฝนระดับเมือง (`precipitation_mm, temp_c, rain_hours_share`)
   - **`rain_effect.parquet`** — ต่อกล้อง: `dry_vpf`, `rain_vpf`, `delta_pct` (+ แถว `ALL` สรุปทั้งเมือง)

ตัวอย่างผลลัพธ์: `ภาพรวม: ฝนตกแล้วรถหนาแน่น เพิ่ม 8.0% (14.2 -> 15.3 คัน/เฟรม)`

ข้อควรรู้:
- ฟรี ไม่ต้องใช้ key แต่ archive **ต้องเป็นอดีต ~5 วันขึ้นไป** (ข้อมูลล่าสุดยังไม่เข้า)
- 1 request/กล้อง → 100 กล้อง = 100 calls (โควตาฟรี 10,000/วัน)
- ถ้ารัน `build_gold.py` ใหม่ gold จะถูกเขียนทับ → รัน `fetch_weather.py` ซ้ำเพื่อ join กลับ (cache ทำให้เร็ว)

## รันทั้ง pipeline ทีเดียว

```bash
python run_pipeline.py --images cctv_images/cctv_images                      # เต็ม
python run_pipeline.py --images cctv_images/cctv_images --limit 200          # ลองเร็ว ๆ ก่อน
python run_pipeline.py --skip manifest,detect                                # มี counts อยู่แล้ว
python run_pipeline.py --images cctv_images/cctv_images --dry-run            # ดูคำสั่งที่จะรัน
```

ลำดับ stage: **manifest → detect (YOLO + merge) → gold (+QC) → weather → สรุป/QC**

- **log ทุก stage** ลง `logs/pipeline_<วันเวลา>.log` พร้อมคำสั่ง, เวลาที่ใช้ และ exit code
- **ตรวจ artifact หลังทุก stage** (มีไฟล์ + จำนวนแถว > 0) ก่อนไปต่อ — ถ้าขั้นไหนพังจะหยุดพร้อมบอกว่าต้องแก้อะไร
- **รันซ้ำได้**: manifest เขียนทับ, YOLO ทำต่อจาก part เดิม (ข้ามรูปที่ทำแล้ว), weather ใช้ cache
- **weather เป็น stage เสริม**: ถ้าเน็ต/โควต้ามีปัญหา จะเตือนแล้วทำต่อ (ไม่ทำให้ทั้ง pipeline ล้ม) — ใช้ `--strict` ถ้าอยากให้หยุด
- `--skip` รับได้หลาย stage คั่นด้วย comma, `--weather-max-cameras N` ไว้ลองโดยไม่ต้องยิง API 100 ครั้ง
- exit code: `0` = สำเร็จ, `1` = stage หลักล้มเหลว (weather ไม่นับ เว้นแต่ `--strict`)

ตัวอย่างตอนจบ:

```
  ✓ manifest  ok          1.3s
  ✓ gold      ok          1.8s
  ✓ weather   ok          4.0s

ไฟล์ผลลัพธ์:
  ✓ counts                   96 แถว  data/silver/counts.parquet
  ✓ gold/qc_camera          100 แถว  data/gold/qc_camera.parquet

QC ข้อมูล:
  no_data=98, ok=2
```

## ข้อควรระวัง

1. **ถ้า `ts_source` เป็น `mtime` เยอะ** = ชื่อไฟล์ไม่มีเวลา และรูปอาจถูกคัดลอกมาทีเดียว (mtime = เวลาก็อป ไม่ใช่เวลาถ่าย) → ต้องเก็บ timestamp ตอน scrape เอง (เช่น scrape รายชั่วโมงแล้วตั้งชื่อไฟล์ด้วยเวลาที่ดึง) ไม่งั้นการวิเคราะห์ temporal ทั้งหมดใช้ไม่ได้
2. **ภาพกลางคืน/สะท้อน** detection จะแย่ลง → รายงานเป็น limitation หรือเพิ่มขั้นปรับ brightness ก่อนเข้าโมเดล
3. `yolov8n` เป็นโมเดลทั่วไป (ไม่ใช่กล้องจราจร) — รถที่ไกล/เบลออาจหลุด ถ้าอยากได้ technical quality เพิ่ม ลอง fine-tune ด้วย subset ที่ label เอง แล้วใส่ `--model best.pt`

## ทดสอบ (CI)

GitHub Actions รันอัตโนมัติทุก push/PR (`.github/workflows/ci.yml`):

1. ตรวจ syntax ทุกสคริปต์ด้วย `py_compile`
2. สร้าง `counts.parquet` ตัวอย่าง (`scripts/make_sample_counts.py` — ไม่ต้องใช้รูปหรือ YOLO)
3. รัน `build_gold.py` กับ `cctv_info.example.csv`
4. **ทดสอบ dashboard ด้วย Streamlit `AppTest`** (`scripts/test_dashboard.py`) — รันแอปจริงแบบ headless แล้วตรวจว่าไม่มี error, KPI ขึ้นครบ, time slider กรองข้อมูลจริง และสลับโหมดแผนที่ได้
5. ตรวจว่าตารางทั้ง 7 เกิดจริง, hotspot มีพิกัดครบ, `traffic_rank` เริ่มที่ 1, `status` เป็นค่าที่อนุญาต

รันเองในเครื่อง (คำสั่งเดียวกับ CI):

```bash
python scripts/make_sample_counts.py --out data/silver/counts.parquet
python build_gold.py --counts data/silver/counts.parquet --out data/gold --locations cctv_info.example.csv
python scripts/test_dashboard.py
```

## License

[MIT](LICENSE) © 2026 BNiDGH — ใช้/แก้/ต่อยอดได้ ระบุที่มาเท่านั้น
