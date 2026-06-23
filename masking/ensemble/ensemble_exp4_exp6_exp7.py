"""
exp4 (SegFormer-B4) + exp6 (SegFormer-B5) + exp7 (SegFormer-B2) 앙상블
소프트맥스 확률 평균 후 argmax → test mIoU 계산
최종 결과: test_mIoU=0.4013

실행:
  python masking/ensemble/ensemble_exp4_exp6_exp7.py \
      --data_root dataset/split_data \
      --exp3_path masking/exp4/best_seg_model_exp4.pth \
      --exp6_path masking/exp6/best_seg_model_exp6.pth \
      --exp7_path masking/exp7/best_seg_model_exp7.pth
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--exp3_path", type=str, required=True, help="best_seg_model_exp4.pth 경로")
parser.add_argument("--exp6_path", type=str, required=True, help="best_seg_model_exp6.pth 경로")
parser.add_argument("--exp7_path", type=str, required=True, help="best_seg_model_exp7.pth 경로")
args = parser.parse_args()

NUM_CLASSES  = 4
IMG_SIZE     = 512
BATCH        = 8
NUM_WORKERS  = 0
CLASS_NAMES  = ["background", "dental_calculus", "dental_caries", "gingivitis"]

IMAGE_DIR = os.path.join(args.data_root, "images")
LABEL_DIR = os.path.join(args.data_root, "labels")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── YOLO polygon → pixel mask ──────────────────────────────────────────────────
def yolo_poly_to_mask(label_path, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = [
                (int(coords[i] * img_w), int(coords[i + 1] * img_h))
                for i in range(0, len(coords) - 1, 2)
            ]
            if len(points) < 3:
                continue
            poly_img = PILImage.new("L", (img_w, img_h), 0)
            ImageDraw.Draw(poly_img).polygon(points, fill=cls_id + 1)
            poly_arr = np.array(poly_img)
            mask[poly_arr > 0] = poly_arr[poly_arr > 0]
    return mask


# ── Dataset ────────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    def __init__(self, split, transform):
        self.img_dir   = os.path.join(IMAGE_DIR, split)
        self.lbl_dir   = os.path.join(LABEL_DIR, split)
        self.transform = transform
        self.files = sorted(
            fn for fn in os.listdir(self.img_dir)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        stem  = os.path.splitext(fname)[0]
        img   = np.array(PILImage.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        h, w  = img.shape[:2]
        mask  = yolo_poly_to_mask(os.path.join(self.lbl_dir, f"{stem}.txt"), w, h)
        out   = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
tf = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

test_ds     = SegDataset("test", tf)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
print(f"Test: {len(test_ds)}장")


# ── 키 변환 (구 transformers → 신 transformers) ────────────────────────────────
def remap_state_dict(old_state):
    import re
    new_state = {}
    for k, v in old_state.items():
        nk = k
        # encoder.patch_embeddings.{i}.* → stages.{i}.patch_embeddings.*
        nk = re.sub(
            r'^segformer\.encoder\.patch_embeddings\.(\d+)\.',
            lambda m: f'segformer.stages.{m.group(1)}.patch_embeddings.',
            nk
        )
        # encoder.block.{i}.{j}.* → stages.{i}.blocks.{j}.*
        nk = re.sub(
            r'^segformer\.encoder\.block\.(\d+)\.(\d+)\.',
            lambda m: f'segformer.stages.{m.group(1)}.blocks.{m.group(2)}.',
            nk
        )
        # encoder.layer_norm.{i}.* → stages.{i}.layer_norm.*
        nk = re.sub(
            r'^segformer\.encoder\.layer_norm\.(\d+)\.',
            lambda m: f'segformer.stages.{m.group(1)}.layer_norm.',
            nk
        )
        # attention 키 이름 변경
        nk = nk.replace('.attention.self.query.', '.attention.q_proj.')
        nk = nk.replace('.attention.self.key.',   '.attention.k_proj.')
        nk = nk.replace('.attention.self.value.', '.attention.v_proj.')
        nk = nk.replace('.attention.output.dense.', '.attention.o_proj.')
        nk = nk.replace('.attention.self.sr.', '.attention.sequence_reduction.sequence_reduction.')
        nk = nk.replace('.attention.self.layer_norm.', '.attention.sequence_reduction.layer_norm.')
        # block 내부 layer norm 이름 변경
        nk = nk.replace('.layer_norm_1.', '.layernorm_before.')
        nk = nk.replace('.layer_norm_2.', '.layernorm_after.')
        # MLP 키 이름 변경
        nk = nk.replace('.mlp.dense1.', '.mlp.fc1.')
        nk = nk.replace('.mlp.dense2.', '.mlp.fc2.')
        # decode_head.linear_c.{i}.* → decode_head.linear_projections.{i}.*
        nk = re.sub(
            r'^decode_head\.linear_c\.(\d+)\.',
            lambda m: f'decode_head.linear_projections.{m.group(1)}.',
            nk
        )
        new_state[nk] = v
    return new_state


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(backbone, ckpt_path):
    base = SegformerForSemanticSegmentation.from_pretrained(
        backbone,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    old_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    new_state  = remap_state_dict(old_state)
    missing, unexpected = base.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [경고] missing keys: {len(missing)}개 — {missing[:3]}...")
    if unexpected:
        print(f"  [경고] unexpected keys: {len(unexpected)}개 — {unexpected[:3]}...")
    base.eval()
    return base.to(DEVICE)

print(f"\nexp4 (mit-b4) 로드: {args.exp3_path}")
model_exp4 = load_model("nvidia/mit-b4", args.exp3_path)

print(f"exp6 (mit-b5) 로드: {args.exp6_path}")
model_exp6 = load_model("nvidia/mit-b5", args.exp6_path)

print(f"exp7 (mit-b2) 로드: {args.exp7_path}")
model_exp7 = load_model("nvidia/mit-b2", args.exp7_path)


# ── 앙상블 추론 ────────────────────────────────────────────────────────────────
print("\n앙상블 추론 중...")
conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        # 각 모델 logits → 원본 해상도로 upsample → softmax
        def get_prob(model, imgs, size):
            logits = model(pixel_values=imgs).logits
            return F.softmax(
                F.interpolate(logits, size=size, mode="bilinear", align_corners=False),
                dim=1
            )

        size  = masks.shape[-2:]
        prob4 = get_prob(model_exp4, imgs, size)
        prob6 = get_prob(model_exp6, imgs, size)
        prob7 = get_prob(model_exp7, imgs, size)

        # 3개 모델 확률 평균 → argmax
        preds = ((prob4 + prob6 + prob7) / 3).argmax(dim=1).cpu().view(-1)
        tgts  = masks.cpu().view(-1)

        conf += torch.bincount(
            NUM_CLASSES * tgts.long() + preds.long(),
            minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        del imgs, masks, prob4, prob6, prob7, preds, tgts


# ── 결과 ───────────────────────────────────────────────────────────────────────
ious = []
for c in range(1, NUM_CLASSES):
    tp = conf[c, c].item()
    fp = conf[:, c].sum().item() - tp
    fn = conf[c, :].sum().item() - tp
    ious.append(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None)

valid     = [v for v in ious if v is not None]
test_miou = sum(valid) / len(valid) if valid else 0.0

print("\n=== Ensemble Test 결과 ===")
print(f"test_mIoU: {test_miou:.4f}")
for i, iou in enumerate(ious):
    name = CLASS_NAMES[i + 1]
    print(f"  {name}: {iou:.4f}" if iou is not None else f"  {name}: N/A")

print("\n=== 단일 모델 비교 ===")
print(f"  exp4 (B4) 단독:       0.3789")
print(f"  exp6 (B5) 단독:       0.3682")
print(f"  exp7 (B2) 단독:       0.3848")
print(f"  exp4+exp7 앙상블:     0.3942")
print(f"  exp4+exp6+exp7 앙상블: {test_miou:.4f}  {'↑ 개선' if test_miou > 0.3942 else '↓ 하락'}")
