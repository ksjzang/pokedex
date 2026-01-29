import pandas as pd
import pyttsx3
import os
import time

def create_pokemon_tts():
    # --- 설정 부분 ---
    excel_file = 'sentences.xlsx'  # 엑셀 파일 이름
    output_folder = 'pokemon_voice' # 결과물이 저장될 폴더 이름
    # ----------------

    # 1. 엑셀 파일 불러오기
    if not os.path.exists(excel_file):
        print(f"❌ 오류: '{excel_file}' 파일을 찾을 수 없습니다.")
        return

    print("📂 엑셀 파일을 읽는 중입니다...")
    # openpyxl 엔진 사용, 데이터가 없는 행은 건너뜀
    df = pd.read_excel(excel_file, engine='openpyxl', header=None)
    
    # 2. 저장할 폴더 만들기
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3. 변환 시작
    success_count = 0
    
    for index, row in df.iterrows():
        # 데이터 가져오기 (첫 번째 열이 텍스트)
        text_val = str(row.iloc[0]).strip() if len(row) > 0 else ""
        # 파일명은 인덱스 번호 사용
        file_name_val = str(index + 1)

        # 내용이 비어있으면 건너뛰기
        if not file_name_val or not text_val or text_val == 'nan':
            continue

        # 저장할 경로 설정 (예: pokemon_voice/1.mp3)
        save_path = os.path.join(output_folder, f"{file_name_val}.mp3")
        
        try:
            print(f"🎙️ 변환 중: {file_name_val}.mp3 (내용: {text_val[:15]}...)")
            
            # pyttsx3로 음성 변환
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)  # 음성 속도 (기본값: 200, 더 높으면 더 빠름)
            engine.save_to_file(text_val, save_path)
            engine.runAndWait()
            
            success_count += 1
            
        except Exception as e:
            print(f"⚠️ 실패 ({file_name_val}): {e}")

    print(f"\n🎉 완료! 총 {success_count}개의 MP3 파일이 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    create_pokemon_tts()