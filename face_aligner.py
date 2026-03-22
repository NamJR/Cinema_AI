"""
Face alignment chuẩn 5 landmark - tương thích ArcFace/InsightFace.
Dùng SimilarityTransform để map 5 landmarks vào template cố định.
"""
import cv2
import numpy as np

# ArcFace template (112x112) - chuẩn InsightFace
# Thứ tự: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Ước lượng ma trận Similarity (scale, rotate, translate) từ cặp điểm.
    Giải least-squares: dst = M @ [src | 1]
    M là 2x3 affine với ràng buộc similarity (a²+b² = scale²).
    """
    num = src.shape[0]
    # Build hệ Ax = b: với M = [a -b tx; b a ty]
    # x' = a*x - b*y + tx,  y' = b*x + a*y + ty
    A = np.zeros((num * 2, 4), dtype=np.float64)
    b = np.zeros((num * 2,), dtype=np.float64)
    for i in range(num):
        x, y = src[i, 0], src[i, 1]
        A[i * 2] = [x, -y, 1, 0]
        A[i * 2 + 1] = [y, x, 0, 1]
        b[i * 2] = dst[i, 0]
        b[i * 2 + 1] = dst[i, 1]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, ty = x

    M = np.array([
        [a, -b_val, tx],
        [b_val, a, ty]
    ], dtype=np.float32)
    return M


def estimate_norm(landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Ước lượng ma trận transform từ landmarks -> template ArcFace.
    landmarks: (5, 2) - left_eye, right_eye, nose, left_mouth, right_mouth
    """
    assert landmarks.shape == (5, 2), f"Expected (5,2), got {landmarks.shape}"

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x

    M = _similarity_transform(landmarks.astype(np.float64), dst)
    return M


def norm_crop(img: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Cắt và align face theo 5 landmarks chuẩn ArcFace.
    """
    M = estimate_norm(landmarks, image_size)
    warped = cv2.warpAffine(
        img, M, (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped


def _estimate_landmarks_from_bbox(bbox: tuple, img_shape: tuple) -> np.ndarray:
    """
    Ước lượng 5 landmarks từ bbox (khi không có detector trả landmarks).
    Tỷ lệ chuẩn: eyes ~35% từ trên, nose ~55%, mouth ~82%.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    landmarks = np.array([
        [x1 + w * 0.26, y1 + h * 0.33],   # left_eye
        [x1 + w * 0.74, y1 + h * 0.33],   # right_eye
        [x1 + w * 0.50, y1 + h * 0.55],   # nose
        [x1 + w * 0.35, y1 + h * 0.80],   # left_mouth
        [x1 + w * 0.65, y1 + h * 0.80],   # right_mouth
    ], dtype=np.float32)

    h_img, w_img = img_shape[:2]
    landmarks[:, 0] = np.clip(landmarks[:, 0], 0, w_img - 1)
    landmarks[:, 1] = np.clip(landmarks[:, 1], 0, h_img - 1)
    return landmarks


def _expand_bbox(bbox: tuple, img_shape: tuple, expand_ratio: float = 0.25) -> tuple:
    """Mở rộng bbox để lấy đủ vùng mặt."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    h_img, w_img = img_shape[:2]

    exp_w = w * expand_ratio
    exp_h = h * expand_ratio

    x1 = max(0, int(x1 - exp_w))
    y1 = max(0, int(y1 - exp_h * 0.5))
    x2 = min(w_img, int(x2 + exp_w))
    y2 = min(h_img, int(y2 + exp_h))
    return (x1, y1, x2, y2)


def align_face(
    image: np.ndarray,
    bbox: tuple,
    landmarks: np.ndarray = None,
    output_size: int = 112
) -> np.ndarray:
    """
    Align face chuẩn 5 landmark ArcFace.
    - Nếu có landmarks thật: dùng norm_crop trực tiếp.
    - Nếu chỉ có bbox: ước lượng landmarks rồi norm_crop.
    """
    if landmarks is None or landmarks.size < 10:
        landmarks = _estimate_landmarks_from_bbox(bbox, image.shape)

    landmarks = np.array(landmarks, dtype=np.float32).reshape(5, 2)
    return norm_crop(image, landmarks, output_size)


def align_face_from_bbox(image: np.ndarray, bbox: tuple, output_size: int = 112) -> np.ndarray:
    """Align từ bbox (ước lượng 5 landmarks)."""
    bbox = _expand_bbox(bbox, image.shape)
    return align_face(image, bbox, landmarks=None, output_size=output_size)
