# Contributing

ขอบคุณที่สนใจครับ 🙌 โปรเจกต์นี้เป็นงานวิชา 2110403 (DSDE-CEDT) แต่ยินดีรับ issue/PR
ถ้าอยากปรับให้ใช้กับข้อมูลกล้องชุดอื่น

## เริ่มยังไง

```bash
git clone https://github.com/BNiDGH/bkk-cctv.git
cd bkk-cctv
python -m pip install -U pandas pyarrow pillow duckdb streamlit
cp cctv_info.example.csv cctv_info.csv     # แล้วเติมข้อมูลกล้องจริง
```

ยังไม่ต้องติดตั้ง `ultralytics` (~2 GB) ถ้าไม่ได้รัน stage YOLO — ทดสอบ gold layer
และ dashboard ได้ด้วยข้อมูลตัวอย่าง:

```bash
python scripts/make_sample_counts.py --out data/silver/counts.parquet
python build_gold.py --counts data/silver/counts.parquet --out data/gold --locations cctv_info.example.csv
python scripts/test_dashboard.py
```

## ก่อนเปิด PR

1. รันสิ่งที่ CI รัน (คำสั่งเดียวกัน ดู `.github/workflows/ci.yml`)
   - `python -m py_compile *.py scripts/*.py`
   - `python scripts/test_dashboard.py` ต้องผ่าน
2. อย่า commit ไฟล์เหล่านี้ (อยู่ใน `.gitignore` แล้ว)
   - `cctv_info.csv` — มี IP ภายในของกล้อง
   - `data/` — ข้อมูลที่ generate ได้ (manifest, counts, gold)
   - `logs/`, `*.pt` (weight)
3. ถ้าแก้ schema ของตาราง gold ให้อัปเดตหัวข้อ "ขั้นที่ 4" ใน README ด้วย

## สไตล์โค้ด

- Python 3.10+ , type hint เท่าที่จำเป็น, docstring ภาษาไทยอธิบาย "ทำไม" ไม่ใช่ "ทำอะไร"
- ทุกสคริปต์ตั้ง `sys.stdout.reconfigure(encoding="utf-8")` เพื่อให้พิมพ์ไทยบนคอนโซล Windows ได้
- สคริปต์ต้อง **รันซ้ำได้** (idempotent) — เขียนทับหรือข้ามของที่ทำแล้ว ไม่สร้างข้อมูลซ้ำ
- ถ้าเพิ่มตาราง/คอลัมน์ใหม่ ใส่ลงใน `GOLD_TABLES` ของ `build_gold.py` เพื่อให้ QC และ CI เห็นด้วย

## Commit message

ขึ้นต้นด้วยสิ่งที่เปลี่ยน แล้วอธิบายเหตุผลสั้น ๆ เช่น

```
เพิ่มตาราง qc_camera เทียบรูปที่ประมวลผลกับ image_count

เดิมรู้แค่ว่ากล้องตัวไหนมีข้อมูล ไม่รู้ว่าขาดไปกี่ชั่วโมง
```

## ข้อมูลและความเป็นส่วนตัว

ข้อมูลภาพ CCTV เป็นข้อมูลสาธารณะของ กทม. แต่ `cctv_info.csv` มี IP ภายในของกล้อง
ห้าม commit หรือแนบใน issue — ใช้ `cctv_info.example.csv` แทนเสมอ
