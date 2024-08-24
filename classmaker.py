import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AdamW
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, captions_file, image_dir, processor):
        with open(captions_file, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = self.captions[idx]
        image_path = os.path.join(self.image_dir, item['file_name'])
        image = Image.open(image_path).convert("RGB")
        text = item['text']

        inputs = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

# 필요한 경로와 프로세서 초기화
image_dir = "Data/"
captions_file = "captions.json"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# 데이터셋 및 DataLoader 생성
dataset = CustomDataset(captions_file, image_dir, processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 로드 및 학습 준비
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

# 파인튜닝 학습 루프
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.1
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# 파인튜닝된 모델 저장
model.save_pretrained("fine_tuned_blip")
processor.save_pretrained("fine_tuned_blip")

# 파인튜닝된 모델로 예측 수행
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_blip")
processor = BlipProcessor.from_pretrained("fine_tuned_blip")
image_path = "Data/image1.jpg"
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs)

caption = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Generated caption: {caption}")
