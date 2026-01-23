# 📍 Trend-KNN  
### Psychological Distance-based Interactive Map of Popular Spots in Seoul & Tokyo

> A data-driven exploration of Gen Z travel preferences using survey data and search trends

---

## 🇰🇷 한국어 (Korean)

### 🌟 프로젝트 동기
- **배경**  
  소카대학교 단기 연수 중 도쿄 도심에서 실제로 “어디가 인기 있는지”를 판단하기 어려운 경험을 함.  
  기존 여행 정보는 광고성 추천이나 물리적 위치 중심이 많아, 또래 대학생들의 실제 선호를 충분히 반영하지 못함.

- **목적**  
  20대 대학생 설문 데이터를 기반으로  
  **‘많이 언급된 장소’가 아닌 ‘심리적으로 가까운 인기 스팟’을 시각화한 지도**를 제작.

---

### 🛠 기술 스택
- **언어 / 라이브러리**: Python, Pandas, NumPy, Plotly  
- **데이터 처리**: 설문 데이터 전처리 및 외부 트렌드 데이터 결합  
- **배포**: GitHub Pages (Interactive Web)

---

### 🧠 분석 방법론 (KNN 개념 응용)
본 프로젝트는 전통적인 분류 목적의 KNN이 아닌,  
**거리 기반 사고방식(Distance-based Thinking)**을 분석 구조에 적용함.

- **가중치 설계**  
  설문 응답에서 1순위·2순위를 구분하여 점수를 합산함으로써  
  단순 빈도(count)가 아닌 **선호 강도(weighted preference)**를 반영.

- **심리적 거리 (Psychological Distance)**  
  장소의 대중적 인지도를 Google 검색량으로 정의하고,  
  다음 수식을 통해 중심과의 거리로 변환.
  ---

### 📊 주요 결과
- **초기 가설**  
“20대 남성과 여성의 도심 여행 취향은 대체로 비슷할 것이다.”

- **분석 결과**
- 남성: 활동성·에너지 중심 (#클럽, #번화가)
- 여성: 감성·분위기 중심 (#카페, #낭만)

→ 성별에 따른 명확한 취향 차이 확인.

---

## 🇯🇵 日本語 (Japanese)

### 🌟 プロジェクトの動機
- **背景**  
創価大学での短期研修中、東京の都心で「実際にどこが人気なのか」を判断するのが難しいと感じた。  
既存の観光情報は広告的・地理的な推薦が多く、同世代のリアルな嗜好を反映していない。

- **目的**  
20代大学生のアンケートデータを基に、  
**心理的距離によって可視化された“本当に人気のあるスポットマップ”**を制作。

---

### 🛠 技術スタック
- **言語・ライブラリ**: Python, Pandas, NumPy, Plotly  
- **データ処理**: アンケート前処理＋外部トレンドデータ統合  
- **デプロイ**: GitHub Pages

---

### 🧠 分析手法（KNN的発想）
- **重み付け**: 回答順位に基づくスコア合算  
- **心理的距離**: Google検索量の逆数として定義

---

### 📊 主な結果
- 男性: 活動性・エネルギー重視（クラブ、賑やかなエリア）
- 女性: 感性・雰囲気重視（カフェ、ロマンチックな空間）

---

## 🇺🇸 English

### 🌟 Project Motivation
- **Background**  
During a short-term program at Soka University, it was difficult to identify places that were genuinely popular among peers.  
Existing travel guides often rely on advertisements or physical proximity rather than real preferences.

- **Objective**  
To visualize **psychological popularity** based on real survey data from Gen Z university students.

---

### 🛠 Tech Stack
- **Language & Libraries**: Python, Pandas, NumPy, Plotly  
- **Data Processing**: Survey preprocessing + external trend data integration  
- **Deployment**: GitHub Pages

---

### 🧠 Methodology (KNN-inspired Logic)
- **Weighted Preferences**: Ranking-based score aggregation  
- **Psychological Distance**: Reciprocal of Google Search Volume

---

### 📊 Key Findings
- **Hypothesis**  
“Gen Z travel preferences are similar regardless of gender.”

- **Result**
- Men prioritize **activity and energy**
- Women prioritize **mood and emotional atmosphere**

---

## 🖥 Interactive Web Map
- 🔗 https://yeonje14.github.io/seoul-tokyo-research/

---

© 2026 Yeonje Lee  
Computer Engineering, Changwon National University
