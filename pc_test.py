import cv2
import numpy as np
import onnxruntime as ort

# === 설정 ===
MODEL_PATH = "pokemon.onnx"
LABEL_PATH = "labels.txt"
INPUT_SIZE = 224  # 모델에 맞게 자동 조정되지만 기본값 224

# 1. 라벨 로드
classes = []
try:
    with open(LABEL_PATH, "r", encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"라벨 {len(classes)}개 로드 완료")
except FileNotFoundError:
    print("Error: labels.txt 파일이 없습니다.")
    exit()

# 2. 모델 로드
print("모델 로딩 중...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    # 입력 사이즈 자동 감지
    try:
        shape = session.get_inputs()[0].shape
        # shape가 [1, 3, H, W] 형태라고 가정
        if len(shape) == 4 and isinstance(shape[2], int):
            INPUT_SIZE = shape[2]
            print(f"모델 입력 크기 자동 감지: {INPUT_SIZE}x{INPUT_SIZE}")
    except:
        pass
    print("✅ 모델 로딩 성공!")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit()

# 3. PC 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("\n=== 실행 방법 ===")
print("👉 스페이스바: 화면 캡쳐 및 분석")
print("👉 q: 종료")

def predict(image):
    # 전처리: BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 리사이즈
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    # 차원 변경 (HWC -> CHW)
    transposed = np.transpose(resized, (2, 0, 1))
    # 배치 차원 추가 및 정규화 (0~1)
    input_data = np.expand_dims(transposed, axis=0).astype(np.float32) / 255.0
    
    # 추론
    outputs = session.run(None, {input_name: input_data})
    
    # 결과 처리
    scores = outputs[0][0]
    max_idx = np.argmax(scores)
    confidence = scores[max_idx]
    
    return classes[max_idx], confidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("PC ONNX Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32: # 스페이스바
        print("\n🔍 분석 중...")
        try:
            label, conf = predict(frame)
            print(f"👉 결과: {label} (확률: {conf:.2f})")
            
            # 화면에 결과 띄우기
            result_frame = frame.copy()
            text = f"{label} ({conf*100:.1f}%)"
            cv2.putText(result_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
            cv2.imshow("PC ONNX Test", result_frame)
            cv2.waitKey(2000) # 2초간 멈춤
            
        except Exception as e:
            print(f"에러 발생: {e}")

cap.release()
cv2.destroyAllWindows()