import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model_path = "pokemon.onnx"
output_model_path = "pokemon_quant.onnx" # 다이어트된 모델 이름

print(f"변환 시작: {input_model_path} -> {output_model_path}")

# float32 -> uint8 (8비트 정수)로 변환
# 용량은 1/4로 줄고, Zero 2W에서 연산 속도는 훨씬 빨라집니다.
quantize_dynamic(
    input_model_path,
    output_model_path,
    weight_type=QuantType.QUInt8
)

print("✅ 변환 완료! 'pokemon_quant.onnx' 파일을 라즈베리파이로 옮기세요.")