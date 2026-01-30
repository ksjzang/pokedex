import pandas as pd
import edge_tts
import asyncio
import os

# ==========================================
# 🎛️ 여기서 목소리, 속도, 톤을 조절하세요
# ==========================================

# 1. 목소리 선택 (주석을 해제/설정해서 선택)
VOICE = "ko-KR-SunHiNeural"  # 여자 목소리 (선희) - 기본
# VOICE = "ko-KR-InJoonNeural" # 남자 목소리 (인준)

# 2. 속도 조절 (기본: "+0%")
# 예: "-10%"(느리게), "+20%"(빠르게)
RATE = "+0%" 

# 3. 톤(높낮이) 조절 (기본: "+0Hz")
# 예: "-5Hz"(굵게/낮게), "+10Hz"(가늘게/높게)
PITCH = "+0Hz" 

# ==========================================

async def create_pokemon_tts_advanced():
    csv_file = 'pokemon_1_to_898.csv'
    output_folder = 'pokemon_voice_pro' # 폴더명 변경
    
    if not os.path.exists(csv_file):
        print(f"❌ '{csv_file}' 파일이 없습니다.")
        return

    df = pd.read_csv(csv_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"🔄 변환 설정: 목소리[{VOICE}], 속도[{RATE}], 톤[{PITCH}]")
    print(f"🔄 총 {len(df)}마리의 포켓몬 변환을 시작합니다...")

    for index, row in df.iterrows():
        p_num = str(row['번호'])
        name = str(row['이름'])
        category = str(row['분류'])
        p_type = str(row['타입'])
        desc = str(row['설명'])
        
        # 타입 글자 처리
        if not p_type.endswith('타입'):
            p_type += " 타입"

        # 읽을 내용 (번호 제외)
        text_to_speak = f"{name}. {category}. {p_type}. {desc}"
        
        # 파일명 (번호 포함)
        filename = f"{p_num}_{name}.mp3"
        save_path = os.path.join(output_folder, filename)

        try:
            # edge-tts 통신 객체 생성
            communicate = edge_tts.Communicate(text_to_speak, VOICE, rate=RATE, pitch=PITCH)
            
            # 파일 저장 (비동기 처리)
            await communicate.save(save_path)
            
            print(f"[{index+1}/{len(df)}] 저장됨: {filename}")
            
        except Exception as e:
            print(f"⚠️ 에러 발생 ({name}): {e}")

    print("\n🎉 모든 변환 작업이 완료되었습니다!")

if __name__ == "__main__":
    # 비동기 함수 실행을 위한 코드
    asyncio.run(create_pokemon_tts_advanced())