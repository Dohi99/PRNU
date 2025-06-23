import numpy as np
import cv2
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

def select_image(title="이미지 파일을 선택하세요"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title,
        filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
    root.destroy()
    return file_path

if __name__ == "__main__":
    print("PRNU 차맵 시각화: 이미지 두 개를 선택하세요")

    img1_path = select_image("첫 번째 이미지 파일을 선택하세요")
    if not img1_path:
        print("취소됨.")
        exit()
    img2_path = select_image("두 번째 이미지 파일을 선택하세요")
    if not img2_path:
        print("취소됨.")
        exit()

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("이미지 로드 실패.")
        exit()

    # 해상도 맞추기 (작은 쪽 기준)
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # PRNU 잔차 추출
    residual1 = extract_residual(img1_rgb)
    residual2 = extract_residual(img2_rgb)
    # 차맵 계산
    diff_map = np.abs(residual1 - residual2)
    # 표준화(0~1)
    diff_map_norm = (diff_map - diff_map.min()) / (np.ptp(diff_map) + 1e-8)

    plt.figure(figsize=(16,5))
    plt.subplot(1,4,1)
    plt.title("이미지1")
    plt.imshow(img1_rgb)
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.title("이미지2")
    plt.imshow(img2_rgb)
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.title("PRNU 차맵 (abs diff)")
    plt.imshow(diff_map_norm, cmap='hot')
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.title("PRNU 차맵 오버레이")
    import matplotlib.cm as cm
    overlay = img1_rgb/255.0 * 0.7 + cm.jet(diff_map_norm)[...,:3]*0.3
    plt.imshow(np.clip(overlay,0,1))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
