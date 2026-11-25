# Mock run — ทดสอบ pipeline แบบเร็ว

เอกสารนี้รวมวิธี "ลองรัน" ทั้งแบบ **ไม่ใช้รูปจริงเลย** (mock data) และแบบ **ใช้รูปจริงแค่หยิบมือ**
ไว้เช็คว่าโค้ด/path/dashboard ทำงานครบ ก่อนรันเต็ม 14,400 รูป (~30–45 นาทีบน CPU)

> คำสั่งทั้งหมดเขียนบรรทัดเดียว ใช้ได้ทั้ง PowerShell และ bash
> รันจากโฟลเดอร์ `bkk-cctv/`

---

## แบบ A — Mock run ล้วน (ไม่ใช้รูป/YOLO, ~10 วินาที)

ใช้เมื่ออยากเช็ค `build_gold.py` + `dashboard.py` ว่าทำงาน โดยไม่ต้องโหลดโมเดล

```powershell
# 0) กัน counts จริงถูกทับ (ถ้ามีอยู่แล้ว)
Copy-Item data/silver/counts.parquet data/silver/counts.real.parquet -Force

# 1) สร้าง counts ปลอม 3 กล้อง 2 วัน
python scripts/make_sample_counts.py --out data/silver/counts.parquet --locations cctv_info.example.csv

# 2) สร้าง gold layer
python build_gold.py --counts data/silver/counts.parquet --out data/gold --locations cctv_info.example.csv

# 3) เปิด dashboard
streamlit run dashboard.py
```

ผลที่ควรเห็น

- `data/gold/` มี `hour_profile`, `hour_series`, `daily`, `camera_stats`, `hotspot`, `qc_camera` (และ `camera_hour`)
- dashboard เปิดที่ http://localhost:8501 มีแผนที่ 3 จุด, KPI, time slider, heatmap
- ตัวเลขทั้งหมดเป็น **ข้อมูลปลอม** — ห้ามใช้ในสไลด์/วิดีโอ

คืนค่าข้อมูลจริงกลับ

```powershell
Move-Item -Force data/silver/counts.real.parquet data/silver/counts.parquet
```

---

## แบบ B — Smoke test ด้วยรูปจริง 200 รูป (~3–5 นาที)

ใช้เมื่ออยากเช็ค manifest → YOLO → merge → gold → weather → dashboard ว่าต่อกันได้จริง

```powershell
python run_pipeline.py --images cctv_images/cctv_images --limit 200 --every-minutes 10 --weather-max-cameras 5
```

- `--limit 200` = หยิบ 200 รูปจากกล้องตัวแรก (1028) — เร็ว แต่ไม่ได้ครอบทุกกล้อง
- ถ้ามี `part-*.parquet` ค้างจากรอบก่อน ตัวเลข `counts` จะมากกว่า 200 (resume ไม่ได้เริ่มใหม่)
- stage manifest ใช้เวลาราว 1–2 นาที เพราะสแกน 61k ไฟล์ — ไม่ใช่ความผิดปกติ
- `--weather-max-cameras 5` = ยิง Open-Meteo แค่ 5 กล้อง ไม่เปลืองโควตา
- `run_pipeline.py` จะรัน **manifest → detect (+merge) → gold → weather → สรุป/QC** ให้เอง
- log อยู่ที่ `logs/pipeline_<วันเวลา>.log`
- ตอนจบจะบอกแถวของแต่ละไฟล์, `QC ข้อมูล: ...` และ Top 3 hotspot

ถ้าอยากให้ครอบหลายกล้องแต่ยังเร็ว ให้ลดความถี่แทนการ limit

```powershell
python run_pipeline.py --images cctv_images/cctv_images --every-minutes 1440 --weather-max-cameras 5
```

`--every-minutes 1440` = 1 เฟรม/วัน/กล้อง → ประมาณ 700 รูปจากทั้ง 100 กล้อง (~5–10 นาที)

เปิด dashboard หลังจบ

```powershell
streamlit run dashboard.py
```

---

## เช็คผลแบบเร็ว (ไม่ต้องเปิด dashboard)

```powershell
python -c "import pandas as pd; c=pd.read_parquet('data/silver/counts.parquet'); print(c.shape, c.camera_id.nunique(), c.ts.min(), c.ts.max())"
python -c "import pandas as pd; h=pd.read_parquet('data/gold/hotspot.parquet'); print(h[['traffic_rank','camera_id','vehicles_per_frame']].head(5).to_string(index=False))"
python -c "import pandas as pd; q=pd.read_parquet('data/gold/qc_camera.parquet'); print(q.status.value_counts().to_dict())"
```

## ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `DefaultCPUAllocator: not enough memory` | แรมไม่พอ → ลด `--batch 1 --imgsz 416` (ค่าเริ่มต้น batch 2) |
| `unrecognized arguments: \` | ใช้ PowerShell กับ `\` ตัดบรรทัด → พิมพ์บรรทัดเดียว |
| `[x] ไม่พบ part` ตอน merge | ยังไม่ได้รัน detect หรือ `--out` คนละโฟลเดอร์ |
| dashboard ขึ้น "ยังไม่พบ gold" | ยังไม่ได้รัน `build_gold.py` หรือ `--out` ไม่ใช่ `data/gold` |
| weather fail แต่ pipeline ไปต่อ | ปกติ (stage เสริม) — ใส่ `--strict` ถ้าอยากให้หยุด |
| อยากเริ่ม mock ใหม่ทั้งหมด | ลบ `data/silver/detections/part-*.parquet` และ `data/gold/` |

## ก่อนรันเต็ม

mock run จะทิ้ง `part-*.parquet` ที่ทำไว้แล้ว ซึ่ง **resume จะข้ามรูปเหล่านี้** ถ้าต้องการรันเต็มสะอาด ๆ ให้ลบ part ก่อน

```powershell
Remove-Item data/silver/detections/part-*.parquet
python run_pipeline.py --images cctv_images/cctv_images --every-minutes 10
```
