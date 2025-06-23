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

def select_image(title="이미지 파일을 선택하세요"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title,
        filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
    root.destroy()
    return file_path

def detect_face(img_rgb):
    # OpenCV haarcascade 얼굴 검출기 사용
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # cascade 파일 경로는 OpenCV가 제공하는 기본 파일로 지정
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # faces: [ [x,y,w,h], ... ]
    return faces

if __name__ == "__main__":
    print("PRNU 기반 얼굴 영역 변조 탐지 예시")
    img_path = select_image()
    if not img_path:
        print("취소됨.")
        exit()
    img = cv2.imread(img_path)
    if img is None:
        print("이미지 로드 실패.")
        exit()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_show = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)

    # 얼굴 자동 검출
    faces = detect_face(img_show)
    if len(faces) == 0:
        print("얼굴이 검출되지 않았습니다.")
        plt.imshow(img_show)
        plt.title("얼굴 미검출")
        plt.axis('off')
        plt.show()
        exit()

    print(f"검출된 얼굴 개수: {len(faces)}")
    # 여러 얼굴이 검출되면 가장 큰 얼굴 선택
    areas = [w*h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]

    # 얼굴 패치 crop & 정사각형 보정
    pad = int(0.1 * max(w,h))  # 얼굴 주변 약간 여유 있게
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_show.shape[1], x + w + pad)
    y2 = min(img_show.shape[0], y + h + pad)
    face_patch = img_show[y1:y2, x1:x2]

    # 얼굴 패치 PRNU 잔차 분석
    residual = extract_residual(face_patch)
    patch_size = 32
    power_map = patch_prnu_power_map(residual, patch_size=patch_size)
    mean_p, std_p = power_map.mean(), power_map.std()
    th = mean_p - 2*std_p
    mask = (power_map < th).astype(np.float32)

    # 시각화
    import matplotlib.cm as cm
    normed_map = (power_map - power_map.min())/(np.ptp(power_map) + 1e-8)
    heatmap = cv2.resize(normed_map, (face_patch.shape[1], face_patch.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_img = cv2.resize(mask, (face_patch.shape[1], face_patch.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap_colored = cm.jet(heatmap)[...,:3]
    overlay = (0.7*face_patch/255.0 + 0.3*heatmap_colored)
    result_img = face_patch.copy()
    result_img[mask_img > 0.5] = [255,0,0]

    # 전체 이미지에 얼굴 bounding box 시각화
    img_face_box = img_show.copy()
    cv2.rectangle(img_face_box, (x1, y1), (x2, y2), (255,0,0), 3)

    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1)
    plt.title("원본 이미지 (얼굴 검출)")
    plt.imshow(img_face_box)
    plt.axis('off')
    plt.subplot(2,3,2)
    plt.title("검출된 얼굴 패치")
    plt.imshow(face_patch)
    plt.axis('off')
    plt.subplot(2,3,3)
    plt.title("PRNU 잔차 (G채널)")
    plt.imshow(residual, cmap='gray')
    plt.axis('off')
    plt.subplot(2,3,4)
    plt.title("PRNU Patch Power Map")
    plt.imshow(power_map, cmap='jet')
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.axis('off')
    plt.subplot(2,3,5)
    plt.title("PRNU 히트맵 오버레이")
    plt.imshow(np.clip(overlay,0,1))
    plt.axis('off')
    plt.subplot(2,3,6)
    plt.title("변조 의심 마스크/표시")
    plt.imshow(result_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
