import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import math
import re
import time
import os
import sys

# 라이브러리 체크
try:
    from korean_romanizer.romanizer import Romanizer
    import pykakasi
    from serpapi import GoogleSearch
except ImportError as e:
    print("❌ 필요 라이브러리가 설치되지 않았습니다.")
    print(f"   에러: {e}")
    sys.exit(1)

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
SERPAPI_KEY = "api key"
INPUT_FILE = 'survey.csv'
CLEAN_FILE = 'clean.csv'
VOLUME_FILE = 'place_volumes.csv'
OUTPUT_HTML = 'index.html'

japanese = pykakasi.kakasi()

# ==========================================
# [STEP 1] 데이터 전처리
# ==========================================
def auto_convert(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).strip()
    if re.search('[가-힣]', text):
        return Romanizer(text).romanize().lower().replace(" ", "")
    result = japanese.convert(text)
    converted = "".join([item['hepburn'] for item in result])
    return converted.lower().replace(" ", "")

def process_survey_data():
    print("\n[1/3] 🧹 데이터 전처리 시작...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 파일이 없습니다.")
        sys.exit(1)
    try:
        df = pd.read_csv(INPUT_FILE)
        target_columns = [col for col in df.columns if '추천' in col or '장소' in col or 'location' in col]
        for col in target_columns:
            df[col] = df[col].apply(auto_convert)
        df.to_csv(CLEAN_FILE, index=False, encoding='utf-8-sig')
        print(f"   ✅ 변환 완료! '{CLEAN_FILE}' 저장됨.")
        return df
    except Exception as e:
        print(f"❌ 전처리 오류: {e}")
        sys.exit(1)

# ==========================================
# [STEP 2] 검색량 수집
# ==========================================
def fetch_search_volumes():
    print("\n[2/3] 🔍 구글 검색량 수집 시작...")
    if os.path.exists(VOLUME_FILE):
        print(f"   ℹ️ 기존 '{VOLUME_FILE}' 파일을 사용합니다.")
        return

    try:
        df = pd.read_csv(CLEAN_FILE)
        target_columns = [
            col for col in df.columns 
            if ('추천' in col or '장소' in col or 'location' in col) 
            and ('이유' not in col and '理由' not in col)
        ]
        all_places = pd.unique(df[target_columns].values.ravel('K'))
        places = [p for p in all_places if pd.notna(p) and p != "" and not str(p).startswith('#')]

        print(f"   - 총 {len(places)}개 장소 검색 시작...")
        results_data = []

        for idx, place in enumerate(places):
            print(f"   - ({idx+1}/{len(places)}) '{place}'...", end=" ")
            params = {"q": place, "location": "Global", "hl": "en", "gl": "us", "api_key": SERPAPI_KEY}
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                total_count = results.get("search_information", {}).get("total_results", 0)
                results_data.append({"place": place, "search_volume": total_count})
                print(f"{total_count:,}개")
            except Exception as e:
                print(f"실패 ({e})")
                results_data.append({"place": place, "search_volume": 0})
            time.sleep(0.5)

        volume_df = pd.DataFrame(results_data)
        volume_df.to_csv(VOLUME_FILE, index=False, encoding='utf-8-sig')
        print("   ✅ 수집 완료.")
    except Exception as e:
        print(f"❌ 검색량 오류: {e}")

# ==========================================
# [STEP 3] 인터랙티브 맵
# ==========================================
def _stable_angle(place: str) -> float:
    h = hashlib.md5(place.encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return 2 * math.pi * u

def _safe_distance(volume: float, count: int, k: float = 30.0, min_d: float = 2.0, max_d: float = 25.0):
    if volume is None or pd.isna(volume): return None
    try: v = float(volume)
    except: return None
    if v <= 1: return None 

    denom = math.log10(v)
    if denom <= 0: return None
    base_distance = k / denom 

    vote_bonus = (count - 1) * 1.5 
    final_distance = base_distance - vote_bonus
    return max(min_d, min(final_distance, max_d))

def _compute_marker_size(count: int, base: float = 12.0, scale: float = 18.0, alpha: float = 0.90, max_size: float = 100.0):
    if count <= 0: return base
    s = base + scale * (count ** alpha)
    return min(s, max_size)

def _separate_points(x, y, sizes, iters=170, padding=2.25, repel_strength=0.065, pull_strength=0.02):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    sizes = np.array(sizes, dtype=float)
    r0 = np.sqrt(x**2 + y**2) + 1e-9
    rad = 0.10 + 0.012 * sizes
    n = len(x)
    if n <= 1: return x.tolist(), y.tolist()

    for _ in range(iters):
        dx = np.zeros(n)
        dy = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                vx = x[i] - x[j]
                vy = y[i] - y[j]
                dist = math.hypot(vx, vy) + 1e-9
                min_dist = (rad[i] + rad[j]) * padding
                if dist < min_dist:
                    overlap = (min_dist - dist) / min_dist
                    push = repel_strength * overlap
                    ux, uy = vx / dist, vy / dist
                    dx[i] += ux * push
                    dy[i] += uy * push
                    dx[j] -= ux * push
                    dy[j] -= uy * push
        x += dx
        y += dy
        r = np.sqrt(x**2 + y**2) + 1e-9
        scale = (r0 / r)
        x = x * (1 - pull_strength) + (x * scale) * pull_strength
        y = y * (1 - pull_strength) + (y * scale) * pull_strength
    return x.tolist(), y.tolist()

def _add_center_marker_only(fig, row, col, center_label):
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="circle", size=52, color="black", opacity=0.05, line=dict(width=0)), hoverinfo="skip", showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="star", size=22, color="black", line=dict(width=1, color="white")), hoverinfo="text", hovertext=f"<b>{center_label} CENTER</b><br>Reference point", showlegend=False), row=row, col=col)

def generate_interactive_map():
    print("\n[3/3] 🎨 인터랙티브 웹 맵 생성 중...")

    try:
        df = pd.read_csv(CLEAN_FILE)
        volumes_df = pd.read_csv(VOLUME_FILE)
        volumes = volumes_df.set_index("place")["search_volume"].to_dict()
    except FileNotFoundError:
        print("❌ CSV 파일 없음.")
        return

    configs = [
        {"title": "Seoul · Male",   "gender": "남성", "pairs": [(7, 8), (9, 10)], "row": 1, "col": 1, "color": "#1f77b4", "center_label": "SEOUL"},
        {"title": "Seoul · Female", "gender": "여성", "pairs": [(7, 8), (9, 10)], "row": 1, "col": 2, "color": "#ff7f0e", "center_label": "SEOUL"},
        {"title": "Tokyo · Male",   "gender": "남성", "pairs": [(3, 4), (5, 6)],  "row": 2, "col": 1, "color": "#2ca02c", "center_label": "TOKYO"},
        {"title": "Tokyo · Female", "gender": "여성", "pairs": [(3, 4), (5, 6)],  "row": 2, "col": 2, "color": "#d62728", "center_label": "TOKYO"},
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[c["title"] for c in configs], horizontal_spacing=0.08, vertical_spacing=0.10)
    all_x, all_y = [], []
    gender_col_idx = 2
    top_picks_per_subplot = []

    for cfg in configs:
        sub_df = df[df.iloc[:, gender_col_idx].astype(str).str.contains(cfg["gender"], na=False)]
        place_data = {}
        for place_idx, reason_idx in cfg["pairs"]:
            for p, r in zip(sub_df.iloc[:, place_idx], sub_df.iloc[:, reason_idx]):
                if pd.isna(p): continue
                p = str(p).strip()
                if not p: continue
                if p not in place_data: place_data[p] = {"count": 0, "reasons": []}
                place_data[p]["count"] += 1
                if pd.notna(r) and str(r).strip(): place_data[p]["reasons"].append(str(r).strip())

        x_vals, y_vals, sizes, hover_texts, labels = [], [], [], [], []
        items_with_distance = []

        for place, info in place_data.items():
            vol = volumes.get(place, None)
            count = info["count"]
            
            d = _safe_distance(vol, count)
            if d is None: continue
            
            items_with_distance.append({"place": place, "distance": d})
            
            angle = _stable_angle(place)
            x, y = d * math.cos(angle), d * math.sin(angle)
            size = _compute_marker_size(count)
            
            unique_reasons = list(dict.fromkeys(info["reasons"]))
            display = unique_reasons[:6]
            if len(unique_reasons) > 6: display.append("…and more")
            reasons_html = "<br>".join([f"• {t}" for t in display]) if display else "• (no reason provided)"
            
            hover_texts.append(f"<b>{place}</b><br><span style='color:#6b7280'>Votes</span> · {count}명<br><span style='color:#6b7280'>Search volume</span> · {vol:,}<br><span style='color:#6b7280'>Distance</span> · {d:.2f}<br><br><b>Reasons</b><br>{reasons_html}")
            x_vals.append(x); y_vals.append(y); sizes.append(size); labels.append(place)

        # 거리 기준 오름차순 정렬
        items_with_distance.sort(key=lambda x: x["distance"])

        # [수정] 공동 2위 포함 로직
        final_top_items = []
        if len(items_with_distance) > 0:
            # 1위 추가
            final_top_items.append(items_with_distance[0])
            
            # 2위부터 확인 (공동 순위 포함)
            if len(items_with_distance) > 1:
                # 2위의 거리값 확인
                second_place_dist = items_with_distance[1]["distance"]
                
                # 2위와 같거나 (공동 2위), 1위와 같은 경우 (공동 1위) 모두 추가
                # 이미 추가한 0번 인덱스는 제외하고 1번부터 순회
                for item in items_with_distance[1:]:
                    # 거리 차이가 거의 없으면(부동소수점 오차 고려) 같은 순위 그룹으로 인정
                    # 2위 거리보다 작거나 같으면 Top 2 그룹에 포함
                    if item["distance"] <= second_place_dist + 1e-9:
                        final_top_items.append(item)
                    else:
                        # 정렬되어 있으므로 더 이상 볼 필요 없음
                        break

        top_picks_per_subplot.append({
            "row": cfg["row"],
            "col": cfg["col"],
            "top_items": final_top_items
        })

        x_vals, y_vals = _separate_points(x_vals, y_vals, sizes)
        all_x += x_vals; all_y += y_vals

        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers+text", text=labels, textposition="top center", textfont=dict(size=11), marker=dict(size=sizes, color=cfg["color"], opacity=0.82, line=dict(width=1, color="rgba(255,255,255,0.95)")), hoverinfo="text", hovertext=hover_texts, showlegend=False), row=cfg["row"], col=cfg["col"])
        _add_center_marker_only(fig, cfg["row"], cfg["col"], cfg["center_label"])

    r = max(10, max(max(abs(min(all_x or [0])), abs(max(all_x or [0]))), max(abs(min(all_y or [0])), abs(max(all_y or [0])))) * 1.25)
    
    # [수정] Top Trends 박스 표기 개선 (공동 순위 반영)
    for pick in top_picks_per_subplot:
        if not pick["top_items"]: continue
        
        text_lines = ["<b>🔥 Top Trends</b>"]
        
        # 1위의 거리 (기준점)
        best_dist = pick["top_items"][0]["distance"]

        for item in pick["top_items"]:
            # 1위와 거리가 같으면(공동 1위) 🥇, 아니면(2위 그룹) 🥈
            if abs(item["distance"] - best_dist) < 1e-9:
                medal = "🥇"
            else:
                medal = "🥈"
            text_lines.append(f"{medal} {item['place']}")
        
        fig.add_annotation(
            text="<br>".join(text_lines),
            x=r*0.90, y=r*0.90,
            showarrow=False,
            row=pick["row"], col=pick["col"],
            font=dict(size=11, color="#333333"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            borderpad=8,
            xanchor="right", yanchor="top"
        )

    fig.update_layout(shapes=[], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="system-ui, sans-serif", size=12, color="#111827"), margin=dict(l=18, r=18, t=64, b=18), showlegend=False, height=860, width=1120, dragmode=False)
    fig.update_xaxes(visible=False, range=[-r, r]); fig.update_yaxes(visible=False, range=[-r, r])

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"dragmode": False, "displaylogo": False, "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]})
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Trend-KNN</title><style>:root{{--bg:#ffffff;--card:#ffffff;--text:#111827;--muted:#6b7280;--border:rgba(17,24,39,0.08);--shadow:0 10px 24px rgba(17,24,39,0.06);--radius:18px;}}body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;}}.wrap{{max-width:1200px;margin:0 auto;padding:30px 18px 44px;}}.header{{max-width:860px;margin-bottom:16px;}}.title{{font-size:26px;font-weight:760;margin:0 0 8px;}}.subtitle{{margin:0;color:var(--muted);font-size:14px;line-height:1.6;}}.card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:14px 14px 10px;}}.footer{{margin-top:10px;color:var(--muted);font-size:12px;}}.divider{{height:1px;background:var(--border);margin:10px 0 0;}}</style></head><body><div class="wrap"><div class="header"><h1 class="title">Trend-KNN Interactive Map</h1><p class="subtitle">Dot size represents <b>survey popularity</b>. Distance from center represents <b>trend strength</b> (Search Volume).<br>Hover a dot to see reasons.</p></div><div class="card">{plot_div}<div class="divider"></div><div class="footer">Center star is the reference point (SEOUL/TOKYO). Larger circles mean more mentions.</div></div></div></body></html>"""
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(html)
    print(f"   ✅ 완성되었습니다! '{OUTPUT_HTML}' 파일을 확인하세요.")

def main():
    print("🚀 [Trend-KNN] 전체 파이프라인 실행 시작...")
    start_time = time.time()
    process_survey_data()
    fetch_search_volumes()
    generate_interactive_map()
    end_time = time.time()
    print(f"\n🎉 모든 작업 완료! (소요 시간: {end_time - start_time:.2f}초)")

if __name__ == "__main__":
    main()