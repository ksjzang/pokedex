import cv2
import numpy as np
import onnxruntime as ort

# === 설정 ===
MODEL_PATH = "pokemon.onnx"
LABEL_PATH = "labels.txt"
INPUT_SIZE = 224  # ViT 모델은 보통 224x224 크기를 사용합니다.

# 1. 라벨 로드
classes = []
try:
    with open(LABEL_PATH, "r", encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: labels.txt 파일이 없습니다.")
    exit()

# 2. 모델 로드 (ONNX Runtime 사용)
print("모델을 로딩 중입니다... (Zero 2W에서는 1~2분 걸릴 수 있습니다)")
try:
    # CPU 모드로 세션 시작
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 모델의 입력 사이즈 정보가 있다면 가져오기
    try:
        shape = session.get_inputs()[0].shape
        # shape가 [1, 3, H, W] 형태인지 확인
        if len(shape) == 4 and isinstance(shape[2], int):
            INPUT_SIZE = shape[2]
            print(f"모델 입력 크기 자동 감지됨: {INPUT_SIZE}")
    except:
        pass
        
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# 3. 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def predict(image):
    # 전처리 1: OpenCV(BGR) -> 모델 입력(RGB) 변환
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 전처리 2: 리사이즈
    resized = cv2.resize(rgb_img, (INPUT_SIZE, INPUT_SIZE))
    
    # 전처리 3: 차원 변경 (H, W, C) -> (C, H, W)
    input_data = np.transpose(resized, (2, 0, 1))
    
    # 전처리 4: 배치 차원 추가 및 정규화 [1, 3, 224, 224], float32
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    input_data /= 255.0  # 0~1 사이 값으로 스케일링

    # 추론 실행 (Inference)
    outputs = session.run(None, {input_name: input_data})
    
    # 결과 처리
    scores = outputs[0][0]     # 첫 번째 배치의 결과값
    max_idx = np.argmax(scores) # 가장 높은 점수의 인덱스 찾기
    confidence = scores[max_idx]
    
    result_text = f"{classes[max_idx]} ({confidence:.2f})"
    print(f"👉 분석 결과: {result_text}")
    return result_text

# 4. 메인 루프
print("준비 완료! 카메라 창을 클릭하고 '스페이스바'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 화면 표시
    cv2.imshow("Pokedex (Press SPACE)", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 누르면 종료
    if key == ord('q'):
        break
    
    # '스페이스바' 누르면 분석
    elif key == 32: 
        print("\n📸 찰칵! 분석 중...")
        try:
            # 분석 실행
            label = predict(frame)
            
            # 화면에 결과 잠시 보여주기
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Pokedex (Press SPACE)", frame)
            
            # 결과 확인을 위해 2초간 멈춤
            cv2.waitKey(2000) 
            print("다시 대기 중...")
            
        except Exception as e:
            print(f"에러 발생: {e}")

cap.release()
cv2.destroyAllWindows()