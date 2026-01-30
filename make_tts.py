import pandas as pd
from gtts import gTTS
import os
import time

def create_pokemon_tts():
    # 1. 파일 설정
    csv_file = 'pokemon_1_to_898.csv'
    output_folder = 'pokemon_tts_output' # 결과물 저장 폴더
    
    # 2. 데이터 불러오기
    if not os.path.exists(csv_file):
        print(f"❌ 오류: '{csv_file}' 파일이 없습니다.")
        return

    df = pd.read_csv(csv_file)
    
    # 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"🔄 총 {len(df)}마리의 포켓몬 변환을 시작합니다...")

    # 3. 한 줄씩 변환
    for index, row in df.iterrows():
        # 데이터 추출 (문자열로 변환)
        p_num = str(row['번호'])
        name = str(row['이름'])
        category = str(row['분류'])
        p_type = str(row['타입'])
        desc = str(row['설명'])
        
        # [타입 처리 로직]
        # 데이터가 '노말'이면 -> '노말 타입'으로 변경
        # 데이터가 '풀, 독 타입'이면 -> 그대로 유지
        if not p_type.endswith('타입'):
            p_type += " 타입"

        # [읽을 내용 구성] - 번호 제외!
        # 예: "레트라. 쥐포켓몬. 노말 타입. 뒷발의 발가락에는..."
        text_to_speak = f"{name}. {category}. {p_type}. {desc}"
        
        # [파일 이름 설정] - 정렬을 위해 파일명에는 번호 포함 (원치 않으면 제거 가능)
        # 예: 21_레트라.mp3
        filename = f"{p_num}_{name}.mp3"
        save_path = os.path.join(output_folder, filename)

        try:
            # TTS 생성 (한국어)
            tts = gTTS(text=text_to_speak, lang='ko')
            tts.save(save_path)
            
            print(f"[{index+1}/{len(df)}] 저장됨: {filename}")
            # print(f"   ㄴ 내용: {text_to_speak[:30]}...") # 확인용 출력
            
            # 구글 차단 방지 딜레이 (1초)
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️ 에러 발생 ({name}): {e}")

    print("\n🎉 변환 작업이 모두 끝났습니다!")

if __name__ == "__main__":
    create_pokemon_tts()