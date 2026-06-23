"""
앙상블 모델(exp4+exp6+exp7) test 데이터 전체 예측 사전 계산
결과: masking/ensemble/predictions/{stem}.npy (512x512, 값 0-3)

실행:
  python masking/ensemble/precompute_predictions.py \
      --data_root dataset/split_data \
      --exp4_path masking/exp4/best_seg_model_exp4.pth \
      --exp6_path masking/exp6/best_seg_model_exp6.pth \
      --exp7_path masking/exp7/best_seg_model_exp7.pth
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--exp4_path", type=str, required=True)
parser.add_argument("--exp6_path", type=str, required=True)
parser.add_argument("--exp7_path", type=str, required=True)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR   = os.path.join(SCRIPT_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

NUM_CLASSES = 4
IMG_SIZE    = 512
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

IMAGE_DIR = os.path.join(args.data_root, "images", "test")


# ── Dataset (라벨 없이 이미지만) ───────────────────────────────────────────────
class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir   = img_dir
        self.transform = transform
        self.files = sorted(
            fn for fn in os.listdir(img_dir)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = np.array(PILImage.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        out   = self.transform(image=img)
        return out["image"], fname


tf = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

ds     = TestImageDataset(IMAGE_DIR, tf)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
print(f"Test 이미지: {len(ds)}장")


# ── 키 변환 ────────────────────────────────────────────────────────────────────
def remap_state_dict(old_state):
    new_state = {}
    for k, v in old_state.items():
        nk = k
        nk = re.sub(r'^segformer\.encoder\.patch_embeddings\.(\d+)\.',
                    lambda m: f'segformer.stages.{m.group(1)}.patch_embeddings.', nk)
        nk = re.sub(r'^segformer\.encoder\.block\.(\d+)\.(\d+)\.',
                    lambda m: f'segformer.stages.{m.group(1)}.blocks.{m.group(2)}.', nk)
        nk = re.sub(r'^segformer\.encoder\.layer_norm\.(\d+)\.',
                    lambda m: f'segformer.stages.{m.group(1)}.layer_norm.', nk)
        nk = nk.replace('.attention.self.query.',      '.attention.q_proj.')
        nk = nk.replace('.attention.self.key.',        '.attention.k_proj.')
        nk = nk.replace('.attention.self.value.',      '.attention.v_proj.')
        nk = nk.replace('.attention.output.dense.',    '.attention.o_proj.')
        nk = nk.replace('.attention.self.sr.',         '.attention.sequence_reduction.sequence_reduction.')
        nk = nk.replace('.attention.self.layer_norm.', '.attention.sequence_reduction.layer_norm.')
        nk = nk.replace('.layer_norm_1.', '.layernorm_before.')
        nk = nk.replace('.layer_norm_2.', '.layernorm_after.')
        nk = nk.replace('.mlp.dense1.', '.mlp.fc1.')
        nk = nk.replace('.mlp.dense2.', '.mlp.fc2.')
        nk = re.sub(r'^decode_head\.linear_c\.(\d+)\.',
                    lambda m: f'decode_head.linear_projections.{m.group(1)}.', nk)
        new_state[nk] = v
    return new_state


def load_model(backbone, ckpt_path):
    base = SegformerForSemanticSegmentation.from_pretrained(
        backbone, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
    )
    state = remap_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    base.load_state_dict(state, strict=False)
    base.eval()
    return base.to(DEVICE)


print("\n모델 로드 중...")
model_exp4 = load_model("nvidia/mit-b4", args.exp4_path)
model_exp6 = load_model("nvidia/mit-b5", args.exp6_path)
model_exp7 = load_model("nvidia/mit-b2", args.exp7_path)
print("모델 로드 완료")


# ── 예측 및 저장 ───────────────────────────────────────────────────────────────
print("\n예측 중...")
with torch.no_grad():
    for imgs, fnames in loader:
        imgs = imgs.to(DEVICE)
        size = (IMG_SIZE, IMG_SIZE)

        def get_prob(model):
            logits = model(pixel_values=imgs).logits
            return F.softmax(
                F.interpolate(logits, size=size, mode="bilinear", align_corners=False),
                dim=1
            )

        preds = ((get_prob(model_exp4) + get_prob(model_exp6) + get_prob(model_exp7)) / 3)
        preds = preds.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, 512, 512)

        for pred, fname in zip(preds, fnames):
            stem = os.path.splitext(fname)[0]
            np.save(os.path.join(PRED_DIR, f"{stem}.npy"), pred)
            print(f"  저장: {stem}.npy")

print(f"\n완료! {PRED_DIR} 에 {len(ds)}개 저장됨")
