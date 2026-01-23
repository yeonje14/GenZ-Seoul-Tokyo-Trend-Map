import pandas as pd
from korean_romanizer.romanizer import Romanizer
import pykakasi
import re

japanese = pykakasi.kakasi()
def auto_convert(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).strip()

    if re.search('[가-힣]', text):
        return Romanizer(text).romanize().lower().replace(" ", "")
    result = japanese.convert(text)
    converted = "".join([item['hepburn'] for item in result])

    return converted.lower().replace(" ", "")

def main():
    input_file = 'survey.csv'
    try:
        print("read file")
        df = pd.read_csv(input_file)

        target_columns = [col for col in df.columns if '추천' in col or '장소' in col or 'location' in col]

        for col in target_columns:
            print(f"convert : {col}")
            df[col] = df[col].apply(auto_convert)

        df.to_csv('clean.csv', index=False, encoding='utf-8-sig')
        print("save file: clean.csv")

        print("divide groups")

        nation_col = [c for c in df.columns if '국적' in c][0]
        gender_col = [c for c in df.columns if '성별' in c][0]      

        groups = {
            'kr_male': df[df[nation_col].str.contains('한국') & df[gender_col].str.contains('남성')],
            'kr_female': df[df[nation_col].str.contains('한국') & df[gender_col].str.contains('여성')],
            'jp_male': df[df[nation_col].str.contains('일본') & df[gender_col].str.contains('남성')],
            'jp_female': df[df[nation_col].str.contains('일본') & df[gender_col].str.contains('여성')]
        }

        
        for name, data in groups.items():
            filename = f"{name}.csv"
            data.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"   save finish: {filename} ({len(data)}명)")

        print("\n all tasks completed successfully")

    except FileNotFoundError:
        print(f" '{input_file}' can not fine the file. check it.")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()