"""
Face detector - YuNet (có landmarks, chính xác) hoặc Haar cascade fallback.
Trả về: list of (bbox, landmarks) với bbox=(x1,y1,x2,y2), landmarks=(5,2) hoặc None.
"""
import cv2
import numpy as np
import os

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def _download_yunet(models_dir: str) -> str | None:
    """Tải YuNet nếu chưa có."""
    path = os.path.join(models_dir, "yunet.onnx")
    if os.path.exists(path):
        return path
    try:
        import urllib.request
        os.makedirs(models_dir, exist_ok=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            path
        )
        if os.path.getsize(path) > 100000:
            return path
    except Exception:
        pass
    return None


class FaceDetector:
    def __init__(self, model_path_or_dir: str):
        self.model_path = model_path_or_dir
        self._yunet = None
        self._cascade = cv2.CascadeClassifier(CASCADE_PATH)

        # Thử load YuNet (có landmarks, chính xác hơn Haar)
        models_dir = os.path.dirname(model_path_or_dir) if "/" in model_path_or_dir or "\\" in model_path_or_dir else "models"
        if not os.path.isdir(models_dir):
            models_dir = "models"
        yunet_path = os.path.join(models_dir, "yunet.onnx")
        if not os.path.exists(yunet_path):
            yunet_path = _download_yunet(models_dir)

        if yunet_path and os.path.exists(yunet_path):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    yunet_path, "", (320, 320),
                    score_threshold=0.6, nms_threshold=0.3, top_k=5000
                )
            except Exception:
                self._yunet = None

    def detect(self, image, return_landmarks=False):
        """Detect faces. Return list of (bbox, landmarks). bbox=(x1,y1,x2,y2), landmarks=(5,2) or None."""
        h, w = image.shape[:2]

        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(image)
            if faces is not None and len(faces) > 0:
                result = []
                for f in faces:
                    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    bbox = (x, y, x + fw, y + fh)
                    # YuNet landmarks: re(4,5), le(6,7), nose(8,9), rm(10,11), lm(12,13)
                    # ArcFace order: le, re, nose, lm, rm
                    lm = np.array([
                        [f[6], f[7]],   # left_eye
                        [f[4], f[5]],   # right_eye
                        [f[8], f[9]],   # nose
                        [f[12], f[13]], # left_mouth
                        [f[10], f[11]], # right_mouth
                    ], dtype=np.float32)
                    result.append((bbox, lm))
                return result

        return self._detect_cascade(image)

    def _detect_cascade(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5,
            minSize=(max(30, min(w, h) // 10), max(30, min(w, h) // 10)),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        result = []
        for (x, y, fw, fh) in faces:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + fw), int(y + fh)
            result.append(((x1, y1, x2, y2), None))

        if not result:
            margin = min(w, h) // 4
            cx, cy = w // 2, h // 2
            size = min(w, h) - 2 * margin
            size = max(size, 50)
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            result.append(((x1, y1, x2, y2), None))
        return result
