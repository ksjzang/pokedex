import cv2
import numpy as np
import onnxruntime as ort
import time
import sys

# === 설정 ===
MODEL_PATH = "pokemon.onnx"
LABEL_PATH = "labels.txt"
INPUT_SIZE = 224

# 1. 라벨 로드
classes = []
try:
    with open(LABEL_PATH, "r", encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: labels.txt not found.")
    exit()

# 2. 모델 로드 (최적화)
print("모델 로딩 중... (1~2분 소요될 수 있음)")
try:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# 3. 카메라 설정 (V4L2 필수)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 카메라 예열
print("카메라 예열 중...")
time.sleep(2)

def predict(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (INPUT_SIZE, INPUT_SIZE))
    input_data = np.transpose(resized, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    input_data /= 255.0
    
    outputs = session.run(None, {input_name: input_data})
    scores = outputs[0][0]
    max_idx = np.argmax(scores)
    return classes[max_idx], scores[max_idx]

# 4. 메인 루프 (인터랙티브 모드)
print("\n=== [AI 포켓몬 도감 대기 중] ===")
print("종료하려면 'Ctrl + C'를 누르세요.\n")

try:
    while True:
        # 1단계: 사용자 입력 대기
        # PuTTY에서는 스페이스바 감지가 어려우므로 '엔터(Enter)'로 대체합니다.
        input("⌨️  촬영하려면 [Enter] 키를 누르세요...")

        # 2단계: 탐지 시작 알림
        print("📸  탐지 중...", end='', flush=True)

        # [중요] 버퍼 비우기
        # 엔터를 누르기 전까지 쌓여있던 옛날 이미지를 버려야
        # 방금 찍은 따끈따끈한 사진을 분석합니다.
        for _ in range(5):
            cap.read()
        
        # 실제 촬영
        ret, frame = cap.read()
        if not ret:
            print("\n❌ 카메라 오류: 다시 시도해주세요.")
            continue

        # 3단계: 분석 및 결과 출력
        try:
            label, conf = predict(frame)
            
            # \r을 사용하여 '탐지 중...' 글자를 덮어씁니다.
            if conf > 0.5:
                print(f"\r👉  탐지됨: {label} (정확도: {conf*100:.1f}%)       ")
            else:
                print(f"\r❓  탐지됨: 확실하지 않음 ({label}?)             ")
                
        except Exception as e:
            print(f"\n에러 발생: {e}")
            
        print("-" * 40) # 구분선

except KeyboardInterrupt:
    print("\n👋 프로그램을 종료합니다.")

cap.release()