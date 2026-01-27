# 📍 Trend-KNN
### Psychological Distance-based Interactive Map of Popular Spots in Seoul & Tokyo

> **"A data-driven exploration of Gen Z travel preferences using weighted survey insights and global search trends."**

---

## 🇰🇷 한국어 (Korean)

### 🌟 프로젝트 동기
- **배경**: 소카대학교 단기 연수 중 도쿄 도심에서 실제로 “어디가 인기 있는지”를 판단하기 어려운 경험을 함. 기존 여행 정보는 광고성 추천이나 물리적 위치 중심이 많아 또래 대학생들의 실질적 선호를 반영하지 못함.
- **목적**: 20대 대학생 설문 데이터를 기반으로, 단순한 물리적 거리가 아닌 **인지도와 선호도가 결합된 ‘심리적 거리 지도’**를 제작하여 직관적인 트렌드 가이드를 제공함.

### 🧠 분석 방법론 (Algorithm Evolution)
본 프로젝트는 데이터 왜곡을 방지하기 위해 초기 모델을 개선한 **Hybrid Scoring** 기법을 적용함.

- **데이터 편향 해결 (Bias Correction)**: 초기 모델에서 검색량이 압도적인 장소가 투표 수와 무관하게 중심에 배치되는 문제를 발견(예: 인천 vs 강남). 이를 해결하기 위해 **Vote-First Bonus** 로직을 도입함.
- **수식 설계**: 
  $$
  Distance = \text{Base (Search Volume)} - (\text{Vote Count} \times 1.5)
  $$
  구글 검색량으로 기본 거리를 산정하되, 투표 수에 강력한 가중치를 부여하여 실제 인기 장소가 중심(🥇)에 오도록 설계.
- **공동 순위 시스템**: 점수가 동일한 경우 공동 순위(Joint Rank)로 처리하여 정보의 누락 없이 시각화.

### 📊 주요 결과 및 해석
- **초기 가설**: “20대 남성과 여성의 여행 취향은 대체로 비슷할 것이다.”
- **분석 결과**: **가설 부분 기각.** 장소(거점)는 공유하나 소비하는 **맥락(Context)**에서 뚜렷한 차이 발생.
  - **남성**: 활동성·에너지 중심 (#클럽, #번화가, #특정목적지)
  - **여성**: 감성·분위기 중심 (#카페, #야경, #로맨틱)

---

## 🇯🇵 日本語 (Japanese)

### 🌟 プロジェクトの動機
- **背景**: 創価大学での研修中、既存の観光情報が同世代のリアルなトレンドを反映していないと感じた。広告的な情報ではなく、大学生の生の声を可視化する必要性を痛感。
- **目的**: アンケートデータと検索トレンドを統合し、**「心理的な近さ」を基準としたインタラクティブマップ**を構築。

### 🧠 分析手法の高度化 (Algorithm Evolution)
データの歪みを防ぎ、信頼性を高めるための**ハイブリッド・スコアリング**を採用。

- **データ偏向の修正**: 検索量のみを基準にすると、実際の人気（投票数）が反映されない問題を確認。これを解決するため、**投票優先重み付け（Vote-First Bonus）**を導入。
- **ロジックの設計**: 投票数1票ごとに中心方向へ強力なボーナス距離（1.5ユニット）を付与。知名度よりも「選ばれた回数」を優先してランク付け。
- **共同順位の処理**: 同一スコアのスポットを共同順位として表示し、トレンドの厚みを可視化。

### 📊 主な結果と解釈
- **結論**: トレンドの拠点は共有しているが、消費の目的には性別による明確な差異が存在する。
  - **男性**: 「何をするか」という**活動性（Activity）**を重視。
  - **女性**: 「どのような空間か」という**雰囲気（Atmosphere）**を重視。

---

## 🇺🇸 English

### 🌟 Project Motivation
- **Background**: Identified a gap between ad-based travel guides and real Gen Z trends during a program at Soka University.
- **Objective**: To visualize **"Psychological Distance"** by integrating survey data from university students with global Google search trends.

### 🧠 Methodology (Algorithm Evolution)
Enhanced the recommendation logic by applying **Hybrid Scoring** to eliminate data bias.

- **Bias Correction**: Identified a flaw where high-search-volume locations overshadowed actual votes (e.g., Incheon vs. Gangnam). Resolved this by implementing **Vote-First Bonus** logic.
- **Formula Design**: 
  $$
  Distance = \text{Base (Search Volume)} - (\text{Vote Count} \times 1.5)
  $$
  This prioritizes actual **user preference (Votes)** over general fame (Search Volume).
- **Tie-Ranking System**: Implemented logic to handle and display joint rankings (🥇, 🥈) for spots with identical psychological scores.

### 📊 Key Findings & Interpretation
- **Hypothesis**: "Travel preferences are similar regardless of gender."
- **Result**: **Partially Rejected.** While sharing general hotspots, the **context of consumption** differs significantly.
  - **Men**: Prioritize **Activity and Purpose** (#Nightlife, #Action).
  - **Women**: Prioritize **Mood and Experience** (#Atmosphere, #Aesthetics).

---

## 🛠 Tech Stack
- **Language / Libs**: Python, Pandas, NumPy, Plotly
- **Data Mining**: SerpApi (Google Search Volume)
- **Deployment**: GitHub Pages (Interactive Web)

## 🖥 Interactive Web Map
- 🔗 [Explore the Map]([https://yeonje14.github.io/seoul-tokyo-research/](https://yeonje14.github.io/GenZ-Seoul-Tokyo-Trend-Map/))

---

© 2026 Yeonje Lee  
Computer Engineering, Changwon National University
