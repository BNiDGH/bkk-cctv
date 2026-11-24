"""ทดสอบ dashboard ด้วย streamlit.testing.AppTest (ใช้ใน CI)

รัน dashboard จริงแบบ headless แล้วตรวจว่า
  - ไม่มี exception ตอนเรนเดอร์
  - KPI 4 ตัวขึ้นครบ
  - เลื่อน time slider แล้วตัวเลขเปลี่ยน (กรองจริง)
  - สลับตัวชี้วัดและโหมดแผนที่ได้

ต้องมี data/gold/camera_hour.parquet ก่อน (รัน build_gold.py)

    python scripts/test_dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # dashboard อ่าน data/gold แบบ path สัมพันธ์กับ cwd จึงต้องยืนที่ root ของโปรเจกต์
    os.chdir(ROOT)
    if not Path("data/gold/camera_hour.parquet").exists():
        sys.exit("[x] ไม่พบ data/gold/camera_hour.parquet — รัน build_gold.py ก่อน")

    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=180).run()
    assert len(at.exception) == 0, f"dashboard มี error: {[e.value for e in at.exception]}"
    assert len(at.error) == 0, f"dashboard ขึ้น error: {[e.value for e in at.error]}"
    assert len(at.metric) == 4, f"ต้องมี KPI 4 ตัว (พบ {len(at.metric)})"
    assert len(at.slider) >= 1 and len(at.multiselect) >= 1, "ไม่พบตัวกรอง"

    print("✓ เรนเดอร์ได้ — KPI:", [(m.label, m.value) for m in at.metric])

    full_day = at.metric[0].value
    at.slider[0].set_value((7, 9)).run()
    assert len(at.exception) == 0, f"error หลังเลื่อน slider: {[e.value for e in at.exception]}"
    morning = at.metric[0].value
    assert morning != full_day, f"time slider ไม่กรองข้อมูล (ยังได้ {morning})"
    print(f"✓ time slider กรองจริง — ทั้งวัน {full_day} → 07:00–09:00 {morning}")

    at.radio[0].set_value("vehicles").run()
    at.radio[1].set_value("แท่ง 3D (Column)").run()
    assert len(at.exception) == 0, f"error หลังสลับโหมด: {[e.value for e in at.exception]}"
    print("✓ สลับตัวชี้วัดและโหมดแผนที่ 3D ได้")

    print("✓ ทดสอบ dashboard ผ่านทั้งหมด")


if __name__ == "__main__":
    main()
