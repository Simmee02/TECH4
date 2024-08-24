from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# 파인튜닝된 모델과 프로세서 로드
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_blip")
processor = BlipProcessor.from_pretrained("fine_tuned_blip")

# 모델을 평가 모드로 전환
model.eval()

# 예측할 이미지 파일 경로
image_path = "/Users/simmee/Documents/GitHub/tech4/Testdata/image12.png"

# 이미지 로드 및 전처리
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,       
        num_beams=5,           
        temperature=0.5,     
        early_stopping=True  
    )

# 생성된 캡션을 디코딩
caption = processor.decode(outputs[0], skip_special_tokens=True)
final_caption = f"This is a damaged block. {caption}"

# 포맷팅된 캡션 출력
print(f"Generated caption: {final_caption}")
