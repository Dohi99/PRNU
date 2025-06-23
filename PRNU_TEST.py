import numpy as np
import cv2
import os
import glob
try:
    from skimage import img_as_float
except ImportError:
    def img_as_float(img):
        return img.astype(np.float32) / 255.0
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import matplotlib
import platform
import tkinter as tk
from tkinter import filedialog

# 한글 폰트 설정
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

def extract_residual(img, wavelet='db2', level=2, method='BayesShrink', mode='soft'):
    img_f = img_as_float(img)
    den = denoise_wavelet(
        img_f[...,1],
        wavelet=wavelet,
        mode=mode,
        method=method,
        wavelet_levels=level,
        rescale_sigma=True
    )
    residual = img_f[...,1] - den
    return residual

def patch_prnu_power_map(residual, patch_size=32):
    H, W = residual.shape
    h_patch, w_patch = patch_size, patch_size
    h_blocks = H // h_patch
    w_blocks = W // w_patch
    power_map = np.zeros((h_blocks, w_blocks))
    for i in range(h_blocks):
        for j in range(w_blocks):
            patch = residual[i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch]
            power_map[i,j] = np.var(patch)
    return power_map

def select_folder(title="폴더를 선택하세요"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def find_faces_folders(root):
    faces_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath).lower() == "faces":
            faces_dirs.append(dirpath)
    return sorted(faces_dirs)

def get_image_files(folder):
    img_types = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff')
    files = []
    for t in img_types:
        files += glob.glob(os.path.join(folder, t))
    return sorted(files)

def safe_folder_name(path):
    # 폴더명에서 경로 구분자 등 특수문자 제거
    return path.replace(':', '').replace('/', '_').replace('\\', '_')

if __name__ == "__main__":
    print("상위 폴더 선택 → 하위 faces 폴더 자동탐색, 결과 저장 폴더 선택 후 파일로 저장")

    # 1. 상위(루트) 폴더 선택
    root_folder = select_folder("상위 폴더(루트)를 선택하세요")
    if not root_folder:
        print("취소됨.")
        exit()
    # 2. 결과 저장 폴더 선택
    save_dir = select_folder("결과 이미지/요약 저장 폴더를 선택하세요")
    if not save_dir:
        print("취소됨.")
        exit()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    faces_folders = find_faces_folders(root_folder)
    if len(faces_folders) == 0:
        print("하위 faces 폴더를 찾지 못했습니다.")
        exit()
    print(f"\n총 {len(faces_folders)}개의 faces 폴더 분석 시작\n")

    threshold_power = 1e-7
    summary_lines = []

    for faces_dir in faces_folders:
        files = get_image_files(faces_dir)
        if len(files) == 0:
            summary_lines.append(f"{faces_dir}\n    이미지 없음\n")
            continue

        face_powers = []
        failed_imgs = []
        rep_img_idx = None
        min_power = None

        for i, img_path in enumerate(files):
            img = cv2.imread(img_path)
            if img is None:
                face_powers.append(np.nan)
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_show = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
            faces = detect_face(img_show)
            if len(faces) == 0:
                face_powers.append(np.nan)
                continue
            areas = [w*h for (x, y, w, h) in faces]
            idx = np.argmax(areas)
            x, y, w, h = faces[idx]
            pad = int(0.1 * max(w,h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_show.shape[1], x + w + pad)
            y2 = min(img_show.shape[0], y + h + pad)
            face_patch = img_show[y1:y2, x1:x2]
            residual = extract_residual(face_patch)
            power_val = np.var(residual)
            face_powers.append(power_val)

            if (min_power is None) or (power_val < min_power):
                min_power = power_val
                rep_img_idx = i

        powers_arr = np.array(face_powers)
        valid_powers = powers_arr[~np.isnan(powers_arr)]
        mean_power = np.mean(valid_powers) if len(valid_powers) > 0 else 0

        if mean_power < threshold_power:
            prnu_message = ">>> 평균 PRNU 파워가 매우 낮음 → 딥페이크/변조 강하게 의심!"
        else:
            prnu_message = ">>> PRNU 파워 정상범위 → 원본(비변조) 가능성 높음!"
        summary_lines.append(f"{faces_dir}\n    평균 PRNU 파워: {mean_power:.2e}   판정: {prnu_message}\n")

        # 대표 이미지(최저 파워) 시각화 및 저장
        if rep_img_idx is not None:
            img_path = files[rep_img_idx]
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_show = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
            faces = detect_face(img_show)
            areas = [w*h for (x, y, w, h) in faces]
            idx = np.argmax(areas)
            x, y, w, h = faces[idx]
            pad = int(0.1 * max(w,h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_show.shape[1], x + w + pad)
            y2 = min(img_show.shape[0], y + h + pad)
            face_patch = img_show[y1:y2, x1:x2]
            residual = extract_residual(face_patch)
            patch_size = 32
            power_map = patch_prnu_power_map(residual, patch_size=patch_size)
            mean_p, std_p = power_map.mean(), power_map.std()
            th = mean_p - 2*std_p
            mask = (power_map < th).astype(np.float32)
            import matplotlib.cm as cm
            normed_map = (power_map - power_map.min())/(np.ptp(power_map) + 1e-8)
            heatmap = cv2.resize(normed_map, (face_patch.shape[1], face_patch.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_img = cv2.resize(mask, (face_patch.shape[1], face_patch.shape[0]), interpolation=cv2.INTER_NEAREST)
            heatmap_colored = cm.jet(heatmap)[...,:3]
            overlay = (0.7*face_patch/255.0 + 0.3*heatmap_colored)
            result_img = face_patch.copy()
            result_img[mask_img > 0.5] = [255,0,0]
            img_face_box = img_show.copy()
            cv2.rectangle(img_face_box, (x1, y1), (x2, y2), (255,0,0), 3)

            fig, axes = plt.subplots(2, 3, figsize=(15,8))
            axes[0,0].set_title("대표 원본 이미지 (얼굴 검출)")
            axes[0,0].imshow(img_face_box)
            axes[0,0].axis('off')
            axes[0,1].set_title("대표 얼굴 패치")
            axes[0,1].imshow(face_patch)
            axes[0,1].axis('off')
            axes[0,2].set_title("PRNU 잔차 (G채널)")
            axes[0,2].imshow(residual, cmap='gray')
            axes[0,2].axis('off')
            axes[1,0].set_title("PRNU Patch Power Map")
            im = axes[1,0].imshow(power_map, cmap='jet')
            plt.colorbar(im, ax=axes[1,0], fraction=0.03, pad=0.02)
            axes[1,0].axis('off')
            axes[1,1].set_title("PRNU 히트맵 오버레이")
            axes[1,1].imshow(np.clip(overlay,0,1))
            axes[1,1].axis('off')
            axes[1,2].set_title(f"변조 의심 마스크/표시\n{prnu_message}")
            axes[1,2].imshow(result_img)
            axes[1,2].axis('off')
            plt.tight_layout()
            folder_tag = safe_folder_name(faces_dir[-80:])  # 너무 긴 경로 줄임
            save_path = os.path.join(save_dir, f"{folder_tag}_prnu_result.png")
            plt.savefig(save_path)
            plt.close(fig)

    # 전체 요약 summary.txt 저장
    summary_txt_path = os.path.join(save_dir, "summary.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.writelines([line+'\n' for line in summary_lines])

    print("\n분석 및 저장 완료!")
    print(f"결과 시각화 이미지 및 summary.txt가 다음 폴더에 저장됨: \n{save_dir}")
