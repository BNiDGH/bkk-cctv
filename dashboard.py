"""Bangkok CCTV Traffic Dashboard (Streamlit + pydeck)

อ่าน data/gold/*.parquet ที่ build_gold.py สร้าง
รัน:  streamlit run dashboard.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

GOLD = Path("data/gold")
METRICS = {
    "vehicles_per_frame": "ความหนาแน่น (คัน/เฟรม)",
    "vehicles": "ปริมาณรวม (คัน)",
}
BANGKOK = {"lat": 13.7563, "lon": 100.5018, "zoom": 10.7}

st.set_page_config(page_title="Bangkok CCTV Traffic", page_icon="🚦", layout="wide")


@st.cache_data(show_spinner=False)
def load(name: str, _mtime: float) -> pd.DataFrame | None:
    path = GOLD / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def read(name: str) -> pd.DataFrame | None:
    path = GOLD / f"{name}.parquet"
    if not path.exists():
        return None
    return load(name, path.stat().st_mtime)


def heat_color(v: float) -> list[int]:
    """0 = น้ำเงิน (รถน้อย) -> 1 = แดง (รถหนาแน่น)"""
    v = max(0.0, min(1.0, float(v)))
    return [int(40 + 215 * v), int(90 - 40 * v), int(230 - 200 * v), 200]


ch = read("camera_hour")
hotspot = read("hotspot")

if ch is None:
    st.error(
        "ยังไม่พบ `data/gold/camera_hour.parquet`\n\n"
        "รันก่อน: `python build_gold.py --counts data/silver/counts.parquet --out data/gold`"
    )
    st.stop()

if "lat" not in ch.columns or ch["lat"].isna().all():
    st.warning("gold layer ยังไม่มีพิกัดกล้อง — วาง `cctv_info.csv` ไว้ข้างสคริปต์ แล้วรัน build_gold.py ใหม่")

ch["hour"] = pd.to_datetime(ch["hour"])
ch["hour_of_day"] = ch["hour"].dt.hour
ch["date"] = ch["hour"].dt.date
metric = "vehicles_per_frame"

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("ตัวกรอง")
    metric = st.radio(
        "ตัวชี้วัดบนแผนที่",
        options=list(METRICS),
        format_func=lambda k: METRICS[k],
        index=0,
    )
    hour_range = st.slider("ช่วงชั่วโมงของวัน", 0, 23, (0, 23), help="เลื่อนดูรถติดช่วงเช้า/เย็น")
    all_dates = sorted(ch["date"].unique())
    picked_dates = st.multiselect("วันที่", all_dates, default=all_dates[-7:])
    top_n = st.slider("แสดงกล้องกี่อันดับแรก", 5, 100, 25, step=5)
    layer_kind = st.radio("รูปแบบแผนที่", ["วงกลม (Scatter)", "แท่ง 3D (Column)"], index=0)
    st.caption("ข้อมูลจาก gold layer · build_gold.py")

# ---------------------------------------------------------------- filter
mask = ch["hour_of_day"].between(*hour_range) & ch["date"].isin(picked_dates or all_dates)
df = ch.loc[mask]

if df.empty:
    st.info("ไม่มีข้อมูลในช่วงที่เลือก ลองขยายช่วงชั่วโมงหรือเพิ่มวันที่")
    st.stop()

agg = (
    df.groupby("camera_id", as_index=False)
    .agg(
        vehicles=("vehicles", "sum"),
        frames=("frames", "sum"),
        vpf=("vehicles_per_frame", "mean"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        location=("location", "first") if "location" in df.columns else ("camera_id", "first"),
    )
    .sort_values(metric if metric != "vehicles_per_frame" else "vpf", ascending=False)
    .reset_index(drop=True)
)
agg["rank"] = range(1, len(agg) + 1)
value_col = "vpf" if metric == "vehicles_per_frame" else "vehicles"

# ---------------------------------------------------------------- KPI
hourly = df.groupby("hour_of_day", as_index=False)["vehicles"].sum()
peak_hour = int(hourly.loc[hourly["vehicles"].idxmax(), "hour_of_day"]) if len(hourly) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("รถทั้งหมด (คัน)", f"{int(df['vehicles'].sum()):,}")
c2.metric("กล้องที่มีข้อมูล", f"{agg['camera_id'].nunique():,}")
c3.metric("เฉลี่ยต่อเฟรม", f"{df['vehicles_per_frame'].mean():.1f} คัน")
c4.metric("ชั่วโมงที่หนาแน่นสุด", f"{peak_hour:02d}:00")

st.title("🚦 Bangkok CCTV — Traffic Hotspot")
st.caption(
    f"{len(df):,} กล้อง-ชั่วโมง · {len(picked_dates) or len(all_dates)} วัน · "
    f"ชั่วโมง {hour_range[0]:02d}:00–{hour_range[1]:02d}:59 · ตัวชี้วัด: {METRICS[metric]}"
)

# ---------------------------------------------------------------- map + table
left, right = st.columns([2.1, 1])

with left:
    geo = agg.dropna(subset=["lat", "lon"]).copy()
    if geo.empty:
        st.info("ไม่มีพิกัดกล้องให้วาดบนแผนที่ (ตรวจคอลัมน์ lat/lon ใน gold layer)")
    else:
        vmax = geo[value_col].max() or 1
        geo["radius"] = 250 + 1100 * (geo[value_col] / vmax)
        geo["color"] = geo[value_col].apply(lambda v: heat_color(v / vmax))
        geo["elev"] = 4000 * (geo[value_col] / vmax)

        tooltip = {
            "html": (
                "<b>{camera_id}</b><br/>{location}<br/>"
                "รถ/เฟรม: {vpf:.1f}<br/>รวม: {vehicles:,} คัน<br/>อันดับ: {rank}"
            )
        }
        if layer_kind.startswith("วงกลม"):
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=geo,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color="color",
                radius_min_pixels=4,
                radius_max_pixels=45,
                pickable=True,
                opacity=0.75,
                stroked=True,
                get_line_color=[255, 255, 255, 120],
            )
        else:
            layer = pdk.Layer(
                "ColumnLayer",
                data=geo,
                get_position=["lon", "lat"],
                get_elevation="elev",
                elevation_scale=1,
                radius=180,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=BANGKOK["lat"], longitude=BANGKOK["lon"],
                    zoom=BANGKOK["zoom"], pitch=35 if layer_kind.startswith("แท่ง") else 0,
                ),
                tooltip=tooltip,
                map_style="light",
            ),
            width="stretch",
        )
        st.caption("ขนาด/สี = ระดับความหนาแน่นของรถ (น้ำเงิน = น้อย, แดง = มาก)")

with right:
    st.subheader(f"Top {min(top_n, len(agg))} กล้อง")
    show = agg.head(top_n)[["rank", "camera_id", "location", "vpf", "vehicles"]].copy()
    show.columns = ["อันดับ", "กล้อง", "สถานที่", "คัน/เฟรม", "รวม (คัน)"]
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "คัน/เฟรม": st.column_config.ProgressColumn(
                "คัน/เฟรม", min_value=0, max_value=float(show["คัน/เฟรม"].max() or 1), format="%.1f"
            ),
            "รวม (คัน)": st.column_config.NumberColumn(format="%d"),
        },
    )

# ---------------------------------------------------------------- charts
st.divider()
g1, g2 = st.columns(2)

with g1:
    st.subheader("รถตามชั่วโมงของวัน")
    by_hour = hourly.set_index("hour_of_day").reindex(range(24)).fillna(0)
    by_hour.index.name = "ชั่วโมง"
    st.bar_chart(by_hour, height=280, color="#e4572e")
    st.caption("รวมทุกวันที่เลือก — ใช้หาช่วง rush hour")

with g2:
    st.subheader("แนวโน้มตามเวลา")
    ts = df.groupby("hour", as_index=False)["vehicles"].sum().set_index("hour")
    st.line_chart(ts, height=280, color="#2e86ab")
    st.caption("อนุกรมเวลาจริง (เห็นผลของวันหยุด/ฝน ถ้า join ข้อมูลอากาศเพิ่ม)")

st.divider()
st.subheader("Heatmap กล้อง × ชั่วโมง")
pivot = (
    df.pivot_table(index="camera_id", columns="hour_of_day", values="vehicles_per_frame", aggfunc="mean")
    .reindex(agg["camera_id"].head(top_n))
)
pivot.columns = [f"{h:02d}" for h in pivot.columns]
# สีไล่ระดับต้องใช้ matplotlib — ต้องเช็คก่อนสร้าง Styler เพราะ error จะเกิดตอน
# st.dataframe ประมวลผล ไม่ใช่ตอนเรียก background_gradient
if importlib.util.find_spec("matplotlib") is not None:
    heat = pivot.style.background_gradient(cmap="YlOrRd", axis=None).format("{:.1f}")
else:
    heat = pivot.round(1)
st.dataframe(
    heat,
    width="stretch",
    height=min(600, 24 * len(pivot) + 60),
)
st.caption("ค่า = คันต่อเฟรมเฉลี่ย · แถวเรียงตามอันดับความหนาแน่น")

if hotspot is not None and {"lat", "lon"} <= set(hotspot.columns):
    st.divider()
    st.subheader("สรุปจาก gold layer (ทั้งช่วงข้อมูล)")
    h = hotspot.dropna(subset=["lat"]).head(3)
    for _, r in h.iterrows():
        st.markdown(
            f"- **{r['camera_id']}** · {r.get('location', '')} — "
            f"{r['vehicles_per_frame']:.1f} คัน/เฟรม · peak {int(r['peak_hour']):02d}:00 · "
            f"คิดเป็น {r['share_pct']:.1f}% ของรถทั้งหมด"
        )
