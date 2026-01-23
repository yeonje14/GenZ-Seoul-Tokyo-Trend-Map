import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import math


def _stable_angle(place: str) -> float:
    h = hashlib.md5(place.encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return 2 * math.pi * u


def _safe_distance(volume: float, k: float = 30.0, min_d: float = 2.0, max_d: float = 25.0):
    if volume is None or pd.isna(volume):
        return None
    try:
        v = float(volume)
    except Exception:
        return None
    if v <= 1:
        return None
    denom = math.log10(v)
    if denom <= 0:
        return None
    d = k / denom
    d = max(min_d, min(d, max_d))
    return d


def _compute_marker_size(
    count: int,
    base: float = 10.0,
    scale: float = 16.0,   # 크기 변화 체감 강화
    alpha: float = 0.90,   # 큰 count 더 강조
    max_size: float = 92.0
):
    if count <= 0:
        return base
    s = base + scale * (count ** alpha)
    return min(s, max_size)


def _separate_points(
    x, y,
    sizes,
    iters: int = 170,
    padding: float = 2.25,
    repel_strength: float = 0.065,
    pull_strength: float = 0.02,
):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    sizes = np.array(sizes, dtype=float)

    r0 = np.sqrt(x**2 + y**2) + 1e-9

    # size(픽셀)를 좌표계로 매핑하는 경험적 스케일
    rad = 0.10 + 0.012 * sizes

    n = len(x)
    if n <= 1:
        return x.tolist(), y.tolist()

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
                    ux = vx / dist
                    uy = vy / dist

                    dx[i] += ux * push
                    dy[i] += uy * push
                    dx[j] -= ux * push
                    dy[j] -= uy * push

        x += dx
        y += dy

        # 반지름(트렌드 거리) 의미를 약하게 유지
        r = np.sqrt(x**2 + y**2) + 1e-9
        scale = (r0 / r)
        x = x * (1 - pull_strength) + (x * scale) * pull_strength
        y = y * (1 - pull_strength) + (y * scale) * pull_strength

    return x.tolist(), y.tolist()


def _add_center_marker_only(fig, row: int, col: int, center_label: str):
    # halo (매우 은은하게)
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(symbol="circle", size=52, color="black", opacity=0.05, line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row, col=col
    )

    # star (hover only)
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(symbol="star", size=22, color="black", line=dict(width=1, color="white")),
            hoverinfo="text",
            hovertext=f"<b>{center_label} CENTER</b><br>Reference point",
            showlegend=False,
        ),
        row=row, col=col
    )


def create_interactive_web_map_toss_style_no_ellipses(
    clean_path: str = "clean.csv",
    volumes_path: str = "place_volumes.csv",
    output_file: str = "index.html",
    distance_k: float = 30.0,
):
    print("🚀 데이터 로딩 및 분석 중...")

    try:
        df = pd.read_csv(clean_path)
        volumes_df = pd.read_csv(volumes_path)
    except FileNotFoundError:
        print(f"❌ 오류: '{clean_path}' 또는 '{volumes_path}' 파일을 찾을 수 없습니다.")
        return

    if not {"place", "search_volume"}.issubset(set(volumes_df.columns)):
        print("❌ 오류: place_volumes.csv는 최소한 place, search_volume 컬럼이 필요합니다.")
        return

    volumes = volumes_df.set_index("place")["search_volume"].to_dict()

    gender_col_idx = 2

    configs = [
        {"title": "Seoul · Male",   "gender": "남성", "pairs": [(7, 8), (9, 10)], "row": 1, "col": 1, "color": "#1f77b4", "center_label": "SEOUL"},
        {"title": "Seoul · Female", "gender": "여성", "pairs": [(7, 8), (9, 10)], "row": 1, "col": 2, "color": "#ff7f0e", "center_label": "SEOUL"},
        {"title": "Tokyo · Male",   "gender": "남성", "pairs": [(3, 4), (5, 6)],  "row": 2, "col": 1, "color": "#2ca02c", "center_label": "TOKYO"},
        {"title": "Tokyo · Female", "gender": "여성", "pairs": [(3, 4), (5, 6)],  "row": 2, "col": 2, "color": "#d62728", "center_label": "TOKYO"},
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[c["title"] for c in configs],
        horizontal_spacing=0.08, vertical_spacing=0.10
    )

    all_x, all_y = [], []

    for cfg in configs:
        gender_series = df.iloc[:, gender_col_idx].astype(str)
        sub_df = df[gender_series.str.contains(cfg["gender"], na=False)]

        place_data = {}
        for place_idx, reason_idx in cfg["pairs"]:
            places = sub_df.iloc[:, place_idx]
            reasons = sub_df.iloc[:, reason_idx]

            for p, r in zip(places, reasons):
                if pd.isna(p):
                    continue
                p = str(p).strip()
                if not p:
                    continue

                if p not in place_data:
                    place_data[p] = {"count": 0, "reasons": []}

                place_data[p]["count"] += 1

                if pd.notna(r):
                    rr = str(r).strip()
                    if rr:
                        place_data[p]["reasons"].append(rr)

        x_vals, y_vals, sizes, hover_texts, labels = [], [], [], [], []

        for place, info in place_data.items():
            vol = volumes.get(place, None)
            d = _safe_distance(vol, k=distance_k, min_d=2.0, max_d=25.0)
            if d is None:
                continue

            angle = _stable_angle(place)
            x = d * math.cos(angle)
            y = d * math.sin(angle)

            count = info["count"]
            size = _compute_marker_size(count)

            unique_reasons = list(dict.fromkeys(info["reasons"]))
            display = unique_reasons[:6]
            if len(unique_reasons) > 6:
                display.append("…and more")

            reasons_html = "<br>".join([f"• {t}" for t in display]) if display else "• (no reason provided)"
            vol_str = f"{vol:,}" if isinstance(vol, (int, float)) and not pd.isna(vol) else "N/A"

            hover_texts.append(
                f"<b>{place}</b><br>"
                f"<span style='color:#6b7280'>Votes</span> · {count}명<br>"
                f"<span style='color:#6b7280'>Search volume</span> · {vol_str}<br>"
                f"<span style='color:#6b7280'>Distance</span> · {d:.2f}<br>"
                f"<br><b>Reasons</b><br>{reasons_html}"
            )

            x_vals.append(x)
            y_vals.append(y)
            sizes.append(size)
            labels.append(place)

        x_vals, y_vals = _separate_points(x_vals, y_vals, sizes)

        all_x += x_vals
        all_y += y_vals

        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(
                    size=sizes,
                    color=cfg["color"],
                    opacity=0.82,
                    line=dict(width=1, color="rgba(255,255,255,0.95)"),
                ),
                hoverinfo="text",
                hovertext=hover_texts,
                showlegend=False,
            ),
            row=cfg["row"], col=cfg["col"]
        )

        _add_center_marker_only(fig, cfg["row"], cfg["col"], cfg["center_label"])

    # 축 범위 자동
    if all_x and all_y:
        max_abs = max(
            max(abs(min(all_x)), abs(max(all_x))),
            max(abs(min(all_y)), abs(max(all_y))),
        )
        r = max(10, max_abs * 1.25)
    else:
        r = 10

    # ✅ 타원/원형 장식(가이드) 완전 제거
    fig.update_layout(shapes=[])

    # Plotly 내부 스타일(최대한 미니멀/정돈)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Apple SD Gothic Neo', 'Helvetica Neue', Arial, sans-serif",
            size=12,
            color="#111827"
        ),
        margin=dict(l=18, r=18, t=64, b=18),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(17,24,39,0.14)",
            font=dict(color="#111827", size=12),
            align="left",
            namelength=-1,
        ),
        showlegend=False,
        height=860,
        width=1120,
    )
    fig.update_annotations(font=dict(size=13, color="#111827"))

    fig.update_xaxes(visible=False, range=[-r, r])
    fig.update_yaxes(visible=False, range=[-r, r])

    # ✅ 드래그로 인한 “점 사라짐” 방지
    fig.update_layout(dragmode=False)

    plotly_config = {
        "dragmode": False,
        "scrollZoom": False,
        "doubleClick": False,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
        ],
    }

    # ===== 토스 톤(브런치 예시처럼) HTML 레이아웃 =====
    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", config=plotly_config)

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Trend-KNN</title>
  <style>
    :root {{
      --bg: #ffffff;                /* 브런치 예시처럼 '흰 화면' */
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: rgba(17,24,39,0.08);
      --shadow: 0 10px 24px rgba(17,24,39,0.06);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display","Apple SD Gothic Neo","Helvetica Neue", Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 30px 18px 44px;
    }}
    .header {{
      max-width: 860px;
      margin-bottom: 16px;
    }}
    .title {{
      font-size: 26px;
      font-weight: 760;
      letter-spacing: -0.02em;
      line-height: 1.15;
      margin: 0 0 8px;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px 14px 10px;
    }}
    .footer {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }}
    .divider {{
      height: 1px;
      background: var(--border);
      margin: 10px 0 0;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1 class="title">Trend-KNN Interactive Map</h1>
      <p class="subtitle">
        Dot size represents <b>survey popularity</b>. Distance from center represents <b>trend strength</b> (higher volume → closer).
        Hover a dot to see reasons. (No decorative ellipse lines.)
      </p>
    </div>

    <div class="card">
      {plot_div}
      <div class="divider"></div>
      <div class="footer">
        Center star is the reference point (SEOUL/TOKYO). Larger circles mean more mentions.
      </div>
    </div>
  </div>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ 완성되었습니다! '{output_file}' 파일을 열어 확인하세요.")


if __name__ == "__main__":
    create_interactive_web_map_toss_style_no_ellipses()