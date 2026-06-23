"""
구강 질환 세그멘테이션 데모 웹앱
사전 조건: precompute_predictions.py 실행 완료 (masking/ensemble/predictions/ 폴더 존재)

실행:
  streamlit run masking/ensemble/app.py
"""

import os
import numpy as np
from PIL import Image as PILImage, ImageDraw
import streamlit as st

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT  = os.path.join(BASE_DIR, "dataset", "split_data")
IMAGE_DIR  = os.path.join(DATA_ROOT, "images", "test")
LABEL_DIR  = os.path.join(DATA_ROOT, "labels", "test")
PRED_DIR   = os.path.join(SCRIPT_DIR, "predictions")

CLASS_NAMES = ["Background", "Dental Calculus (치석)", "Dental Caries (충치)", "Gingivitis (치은염)"]
CLASS_COLORS = [
    None,                    # background: 표시 안 함
    (220, 50,  50,  160),   # calculus: 빨강
    (50,  100, 220, 160),   # caries: 파랑
    (50,  200, 80,  160),   # gingivitis: 초록
]


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


# ── 컬러 오버레이 생성 ─────────────────────────────────────────────────────────
def apply_overlay(img_pil, mask):
    img_rgba    = img_pil.convert("RGBA")
    overlay_arr = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        if color is None:
            continue
        overlay_arr[mask == cls_id] = color
    overlay = PILImage.fromarray(overlay_arr, "RGBA")
    result  = PILImage.alpha_composite(img_rgba, overlay)
    return result.convert("RGB")


# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Oral Disease Segmentation", layout="wide")


# ── 모델 설명 섹션 ─────────────────────────────────────────────────────────────
st.title("Oral Disease Segmentation")
with st.expander("📋 모델 및 성능 정보", expanded=False):
    st.markdown("""

### 데이터셋


이미지 1장에 여러 클래스가 동시에 존재할 수 있어 클래스별 합계 ≠ 전체 이미지 수.

| 구분 | 전체 이미지 | dental_calculus | dental_caries | gingivitis |
|---|---|---|---|---|
| Train | 2,563 | 532 | 798 | 1,544 |
| Val   | 405   | 56  | 243 | 130  |
| Test  | 407   | 58  | 250 | 135  |
| **합계** | **3,375** | **646** | **1,291** | **1,809** |

> **Split 방식**: pHash 기반 Group split — 중복·증강 이미지가 다른 split에 배치되는 data leakage 제거

---

### 앙상블 구성 모델

3개 SegFormer 모델의 소프트맥스 확률 평균 후 argmax.

| 실험 | 모델 | 파라미터 | Batch | 입력 크기 | Epochs (조기종료) |
|---|---|---|---|---|---|
| exp4 | SegFormer-B4 | 64M | 8  | 512×512 | 100 (25에서 종료) |
| exp6 | SegFormer-B5 | 84M | 32 | 512×512 | 100 (64에서 종료) |
| exp7 | SegFormer-B2 | 25M | 16 | 512×512 | 120 (113에서 종료) |

**exp4 vs exp6 vs exp7**

| | exp4 | exp6 | exp7 |
|---|---|---|---|
| Loss | 0.5×CE + 0.5×Dice | 0.5×CE + 0.5×Dice | 0.5×Focal + 0.5×Dice |
| LR (backbone / head) | 6e-5 / 6e-5 | 1e-5 / 1e-4 | 6e-5 / 6e-4 |
| Calculus 오버샘플링 | 3× | 3× | 5× |
| 추가 Augmentation | — | — | ElasticTransform, ShiftScaleRotate, CoarseDropout |

---

### 실험별 성능 (test 기준)

| 실험 | test mIoU | calculus IoU | caries IoU | gingivitis IoU |
|---|---|---|---|---|
| exp4 (B4) | 0.3789 | 0.272 | 0.483 | 0.381 |
| exp6 (B5) | 0.3682 | 0.292 | 0.447 | 0.366 |
| exp7 (B2) | 0.3848 | 0.262 | 0.512 | 0.380 |
| **앙상블 (exp4+exp6+exp7)** | **0.4013** | **0.294** | **0.514** | **0.396** |


""")

# ── 범례 ───────────────────────────────────────────────────────────────────────
st.markdown(
    '<span style="background-color:rgba(220,50,50,0.7);padding:2px 10px;border-radius:4px;color:white">Calculus</span>'
    "&nbsp;&nbsp;"
    '<span style="background-color:rgba(50,100,220,0.7);padding:2px 10px;border-radius:4px;color:white">Caries</span>'
    "&nbsp;&nbsp;"
    '<span style="background-color:rgba(50,200,80,0.7);padding:2px 10px;border-radius:4px;color:white">Gingivitis</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── 이미지 목록 로드 ───────────────────────────────────────────────────────────
if not os.path.exists(PRED_DIR) or not os.listdir(PRED_DIR):
    st.error("예측 파일이 없습니다. 먼저 precompute_predictions.py를 실행하세요.")
    st.stop()

test_files = sorted(
    fn for fn in os.listdir(IMAGE_DIR)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
)
stems_with_pred = {os.path.splitext(fn)[0] for fn in os.listdir(PRED_DIR) if fn.endswith(".npy")}
test_files = [fn for fn in test_files if os.path.splitext(fn)[0] in stems_with_pred]

if not test_files:
    st.error("예측 파일과 매칭되는 이미지가 없습니다.")
    st.stop()


st.markdown("")

col_header_gt, col_header_pred = st.columns(2)
with col_header_gt:
    st.subheader("Ground Truth")
with col_header_pred:
    st.subheader("Model Prediction")

# ── 전체 이미지 스크롤 뷰 ──────────────────────────────────────────────────────
for fname in test_files:
    stem = os.path.splitext(fname)[0]

    img_path  = os.path.join(IMAGE_DIR, fname)
    lbl_path  = os.path.join(LABEL_DIR, f"{stem}.txt")
    pred_path = os.path.join(PRED_DIR,  f"{stem}.npy")

    orig_img = PILImage.open(img_path).convert("RGB")
    w, h     = orig_img.size

    gt_mask   = yolo_poly_to_mask(lbl_path, w, h)
    pred_mask = np.load(pred_path)
    pred_mask_orig = np.array(
        PILImage.fromarray(pred_mask).resize((w, h), resample=PILImage.NEAREST)
    )

    gt_overlay   = apply_overlay(orig_img, gt_mask)
    pred_overlay = apply_overlay(orig_img, pred_mask_orig)

    display_w = min(w, 700)
    col_gt, col_pred = st.columns(2)
    with col_gt:
        st.image(gt_overlay, caption=fname, width=display_w)
    with col_pred:
        st.image(pred_overlay, width=display_w)

    st.divider()
