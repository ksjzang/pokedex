import cv2
import numpy as np
import onnxruntime as ort
import os

# ==========================================
# 👇 여기에 테스트할 이미지 파일 이름을 적으세요
IMAGE_FILE = "test.png" 
# ==========================================

MODEL_PATH = "pokemon.onnx"
LABEL_PATH = "labels.txt"
INPUT_SIZE = 224  # 기본값 (모델에 따라 자동 조정됨)

# 1. 라벨 로드
classes = []
try:
    with open(LABEL_PATH, "r", encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: labels.txt 파일이 없습니다.")
    exit()

# 2. 이미지 파일 확인
if not os.path.exists(IMAGE_FILE):
    print(f"❌ 오류: '{IMAGE_FILE}' 파일을 찾을 수 없습니다.")
    print("👉 같은 폴더에 이미지 파일이 있는지, 이름이 정확한지 확인해주세요.")
    exit()

# 3. 모델 로드
print("모델 로딩 중...")
try:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 입력 사이즈 자동 감지
    try:
        shape = session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int):
            INPUT_SIZE = shape[2]
            print(f"ℹ️ 모델 입력 크기 자동 감지: {INPUT_SIZE}x{INPUT_SIZE}")
    except:
        pass
        
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# 4. 이미지 전처리 및 예측 함수
def predict_image(filename):
    # 이미지 읽기 (OpenCV는 BGR로 읽습니다)
    img = cv2.imread(filename)
    
    if img is None:
        print("❌ 이미지를 읽을 수 없습니다. (손상된 파일일 수 있음)")
        return

    print(f"📸 이미지 로드 성공: {filename} ({img.shape[1]}x{img.shape[0]})")

    # [전처리 1] BGR -> RGB 변환 (모델은 보통 RGB를 원함)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # [전처리 2] 리사이즈 (224x224)
    resized = cv2.resize(rgb_img, (INPUT_SIZE, INPUT_SIZE))
    
    # [전처리 3] 차원 변경 (H,W,C) -> (C,H,W)
    input_data = np.transpose(resized, (2, 0, 1))
    
    # [전처리 4] 배치 차원 추가 및 정규화 (0~1)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    input_data /= 255.0

    # 추론 실행
    print("🧠 분석 중...")
    outputs = session.run(None, {input_name: input_data})
    
    # 결과 처리
    scores = outputs[0][0] # Softmax 확률값들
    max_idx = np.argmax(scores)
    confidence = scores[max_idx]
    
    return classes[max_idx], confidence, scores

# 5. 실행 및 결과 출력
label, conf, all_scores = predict_image(IMAGE_FILE)

print("\n" + "="*30)
print(f"👉 결 과: {label}")
print(f"📊 정확도: {conf*100:.2f}%")
print("="*30)

# (선택사항) 상위 3개 후보 보여주기
top3_indices = np.argsort(all_scores)[::-1][:3]
print("\n[상위 3개 후보]")
for i in top3_indices:
    print(f"- {classes[i]}: {all_scores[i]*100:.1f}%")