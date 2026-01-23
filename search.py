import pandas as pd
from serpapi import GoogleSearch
import time

API_KEY = 

def get_search_volume():

    try:
        df = pd.read_csv('clean.csv')
    except FileNotFoundError:
        print("❌ clean.csv 파일이 없습니다. soka.py를 먼저 실행해 주세요.")
        return

    # 2. 진짜 장소 이름만 골라내기 
    target_columns = [
        col for col in df.columns 
        if ('추천' in col or '장소' in col or 'location' in col) 
        and ('이유' not in col and '理由' not in col)
    ]
    all_places = pd.unique(df[target_columns].values.ravel('K'))
    places = [p for p in all_places if pd.notna(p) and p != "" and not str(p).startswith('#')]

    print(f"🔍 총 {len(places)}개의 장소를 찾았습니다. 검색량 수집을 시작합니다.")

    results_data = []

    for place in places:
        print(f"📡 '{place}' 검색 중...")
        
        # SerpApi로 구글 검색 결과 수 가져오기
        params = {
            "q": place,
            "location": "Global",
            "hl": "en",
            "gl": "us",
            "api_key": API_KEY
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # 'total_results'가 검색 결과 개수입니다.
            total_count = results.get("search_information", {}).get("total_results", 0)
            
            results_data.append({
                "place": place,
                "search_volume": total_count
            })
            print(f"   ✅ 결과: {total_count}개")
            
        except Exception as e:
            print(f"   ❌ {place} 검색 중 오류 발생: {e}")
            results_data.append({"place": place, "search_volume": 0})
        
        # API 부하를 줄이기 위해 아주 잠깐 쉽니다.
        time.sleep(0.5)

    # 3. 결과를 엑셀(CSV)로 저장
    volume_df = pd.DataFrame(results_data)
    volume_df.to_csv('place_volumes.csv', index=False, encoding='utf-8-sig')
    print("\n🎉 모든 검색량 수집 완료! 'place_volumes.csv' 파일을 확인하세요.")

if __name__ == "__main__":
    get_search_volume()