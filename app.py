from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import io
from database import SessionLocal, Face, RecognitionLog
from insightface_service import InsightFaceService, try_create_fallback_service

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# ArcFace/InsightFace cosine similarity threshold (embeddings L2-normalized)
# buffalo_l: 0.5 | ONNX fallback: 0.45
COSINE_THRESHOLD_INSIGHTFACE = 0.50
COSINE_THRESHOLD_ONNX        = 0.45
MAX_EMBEDDINGS_PER_PERSON    = 5   # Tối đa bao nhiêu ảnh đăng ký mỗi người

app = FastAPI(title="Cinema Face Recognition API", version="2.0")
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Khởi tạo service (InsightFace ưu tiên, fallback ONNX)
# ---------------------------------------------------------------------------
face_service      = None
face_service_mode = "none"
COSINE_THRESHOLD  = COSINE_THRESHOLD_ONNX  # default, sẽ override bên dưới

try:
    svc = InsightFaceService(model_name="buffalo_l")
    if svc.is_ready():
        face_service      = svc
        face_service_mode = "insightface"
        COSINE_THRESHOLD  = COSINE_THRESHOLD_INSIGHTFACE
        print("[INFO] InsightFace (buffalo_l) loaded successfully.")
    else:
        print(f"[WARN] InsightFace: {svc.get_error()}")
except Exception as e:
    print(f"[WARN] InsightFace init failed: {e}")

if face_service is None:
    fallback = try_create_fallback_service(MODELS_DIR)
    if fallback is not None and fallback.is_ready():
        face_service      = fallback
        face_service_mode = "legacy"
        COSINE_THRESHOLD  = COSINE_THRESHOLD_ONNX
        print("[INFO] Fallback: ONNX SCRFD + ArcFace R100 loaded.")
    else:
        err = fallback.get_error() if fallback else "Không tìm thấy ONNX models"
        print(f"[ERROR] Không load được model: {err}")
        print("  1. Cài InsightFace: pip install insightface (cần C++ Build Tools)")
        print("  2. Hoặc đặt scrfd_10g_bnkps.onnx + arcface_r100.onnx vào thư mục models/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(contents: bytes) -> np.ndarray | None:
    """Load image từ bytes, áp dụng EXIF orientation. Trả về BGR."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(contents))
        img = img.convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    npimg = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


def _check_service():
    """Trả về error dict nếu service chưa ready, None nếu OK."""
    if face_service is None or not face_service.is_ready():
        err = face_service.get_error() if face_service else "No model loaded"
        return {
            "error": "Model nhận diện chưa sẵn sàng.",
            "detail": err,
            "hint": "Cài insightface: pip install insightface  |  hoặc đặt ONNX models vào thư mục models/"
        }
    return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vector L2-normalized."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    for err in errors:
        if "file" in str(err.get("loc", [])) and err.get("type") == "missing":
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Thiếu file ảnh.",
                    "hint": "POST form-data, field 'file' chứa file ảnh (jpg/png)."
                }
            )
    return JSONResponse(status_code=422, content={"detail": errors})


# ---------------------------------------------------------------------------
# API: Status
# ---------------------------------------------------------------------------
@app.get("/status", summary="Kiểm tra trạng thái model")
async def status():
    """Kiểm tra service đang dùng model nào."""
    if face_service is None:
        return {
            "status": "error",
            "mode": "none",
            "message": "pip install insightface  hoặc đặt ONNX models vào thư mục models/"
        }
    return {
        "status": "ok",
        "mode": face_service_mode,
        "cosine_threshold": COSINE_THRESHOLD,
        "max_embeddings_per_person": MAX_EMBEDDINGS_PER_PERSON,
    }


# ---------------------------------------------------------------------------
# API: So sánh 2 ảnh trực tiếp
# ---------------------------------------------------------------------------
@app.post("/compare", summary="So sánh 2 ảnh (không qua DB)")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """So sánh cosine similarity giữa 2 ảnh. Dùng để test pipeline."""
    err = _check_service()
    if err:
        return err

    img1 = load_image(await file1.read())
    img2 = load_image(await file2.read())
    if img1 is None:
        return {"error": "Không đọc được ảnh 1"}
    if img2 is None:
        return {"error": "Không đọc được ảnh 2"}

    r1 = face_service.detect_and_embed(img1)
    r2 = face_service.detect_and_embed(img2)
    if not r1:
        return {"error": "Không detect được mặt trong ảnh 1"}
    if not r2:
        return {"error": "Không detect được mặt trong ảnh 2"}

    e1 = np.array(r1[0]["embedding"], dtype=np.float32)
    e2 = np.array(r2[0]["embedding"], dtype=np.float32)
    cos_sim = _cosine(e1, e2)
    return {
        "cosine_similarity": round(cos_sim, 4),
        "same_person": cos_sim >= COSINE_THRESHOLD,
        "threshold_used": COSINE_THRESHOLD,
        "mode": face_service_mode,
    }


# ---------------------------------------------------------------------------
# API: Đăng ký khuôn mặt
# ---------------------------------------------------------------------------
@app.post("/register/{customer_id}", summary="Đăng ký khuôn mặt")
async def register_face(
    customer_id: int,
    file: UploadFile = File(...),
    replace: bool = Query(False, description="True = xóa hết ảnh cũ rồi đăng ký lại từ đầu"),
):
    """
    Đăng ký khuôn mặt cho customer_id.
    - Mỗi lần gọi thêm 1 embedding (tối đa 5).
    - Gửi `?replace=true` để xóa toàn bộ và đăng ký lại.
    - Nên đăng ký 3-5 ảnh từ nhiều góc để đạt độ chính xác cao nhất.
    """
    err = _check_service()
    if err:
        return err

    contents = await file.read()
    image = load_image(contents)
    if image is None:
        return {"error": "Không đọc được ảnh. Kiểm tra định dạng file (jpg, png)."}

    # Kiểm tra chất lượng ảnh tối thiểu
    h, w = image.shape[:2]
    if h < 80 or w < 80:
        return {"error": f"Ảnh quá nhỏ ({w}x{h}). Cần tối thiểu 80x80 px."}

    results = face_service.detect_and_embed(image)
    if not results:
        return {"error": "Không detect được khuôn mặt trong ảnh."}

    # Lấy mặt có det_score cao nhất (đã sort trong service)
    best = results[0]
    det_score = best.get("det_score", 1.0)
    if det_score < 0.3:
        return {"error": f"Khuôn mặt không rõ (det_score={det_score:.2f}). Chụp lại ảnh rõ hơn."}

    embedding = best["embedding"]

    try:
        db = SessionLocal()

        if replace:
            db.query(Face).filter(Face.customer_id == customer_id).delete()
            db.commit()

        existing_count = db.query(Face).filter(Face.customer_id == customer_id).count()

        if existing_count >= MAX_EMBEDDINGS_PER_PERSON:
            db.close()
            return {
                "error": f"Đã đạt tối đa {MAX_EMBEDDINGS_PER_PERSON} ảnh cho customer {customer_id}.",
                "hint": f"Gửi ?replace=true để đăng ký lại từ đầu.",
                "current_count": existing_count,
            }

        face = Face(customer_id=customer_id, embedding=embedding)
        db.add(face)
        db.commit()
        new_count = existing_count + 1
        db.close()
    except Exception as e:
        return {"error": f"Lỗi database: {str(e)}"}

    return {
        "message": "Face registered",
        "customer_id": customer_id,
        "total_embeddings": new_count,
        "max_embeddings": MAX_EMBEDDINGS_PER_PERSON,
        "det_score": round(det_score, 3),
        "tip": f"Đăng ký thêm {MAX_EMBEDDINGS_PER_PERSON - new_count} ảnh từ góc khác để tăng độ chính xác." if new_count < MAX_EMBEDDINGS_PER_PERSON else "Đã đủ số ảnh tối đa.",
    }


# ---------------------------------------------------------------------------
# API: Xóa đăng ký
# ---------------------------------------------------------------------------
@app.delete("/register/{customer_id}", summary="Xóa đăng ký khuôn mặt")
async def delete_faces(customer_id: int):
    """Xóa toàn bộ khuôn mặt đã đăng ký của customer."""
    try:
        db = SessionLocal()
        deleted = db.query(Face).filter(Face.customer_id == customer_id).delete()
        db.commit()
        db.close()
        return {"message": f"Đã xóa {deleted} embedding(s)", "customer_id": customer_id}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# API: Xem danh sách đăng ký
# ---------------------------------------------------------------------------
@app.get("/faces", summary="Danh sách customer đã đăng ký")
async def list_registered_faces():
    """Xem danh sách customer đã đăng ký và số lượng embedding."""
    try:
        db = SessionLocal()
        faces = db.query(Face).all()
        db.close()

        from collections import defaultdict
        counts: dict[int, int] = defaultdict(int)
        for f in faces:
            counts[f.customer_id] += 1

        return {
            "total_customers": len(counts),
            "customers": [
                {"customer_id": cid, "embedding_count": cnt}
                for cid, cnt in sorted(counts.items())
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# API: Xác thực khuôn mặt
# ---------------------------------------------------------------------------
@app.post("/verify", summary="Nhận diện khuôn mặt")
async def verify_face(
    file: UploadFile = File(...),
    camera_id: str = Query("", description="ID camera (ví dụ: cam_001)"),
    threshold: float = Query(None, description=f"Cosine threshold (mặc định theo mode)")
):
    """
    Nhận diện khuôn mặt trong ảnh so với tất cả khuôn mặt đã đăng ký.
    - So sánh với **tất cả** embedding của mỗi người, lấy điểm cao nhất.
    - Trả về customer_id nếu vượt ngưỡng threshold.
    """
    err = _check_service()
    if err:
        return err

    contents = await file.read()
    image = load_image(contents)
    if image is None:
        return {"error": "Không đọc được ảnh. Kiểm tra định dạng file (jpg, png)."}

    results = face_service.detect_and_embed(image)
    if not results:
        return {"error": "Không detect được khuôn mặt trong ảnh."}

    query_embedding = np.array(results[0]["embedding"], dtype=np.float32)
    thresh = threshold if threshold is not None else COSINE_THRESHOLD

    try:
        db = SessionLocal()
        faces = db.query(Face).all()

        # Per-customer: lấy max similarity trong tất cả embeddings của người đó
        best_per_customer: dict[int, float] = {}
        for face in faces:
            stored_emb = np.array(face.embedding, dtype=np.float32)
            cos_sim = _cosine(query_embedding, stored_emb)
            cid = face.customer_id
            if cid not in best_per_customer or cos_sim > best_per_customer[cid]:
                best_per_customer[cid] = cos_sim

        # Tìm customer có điểm cao nhất
        best_customer_id = None
        best_cos_sim = -1.0
        for cid, sim in best_per_customer.items():
            if sim > best_cos_sim:
                best_cos_sim = sim
                best_customer_id = cid

        similarity_01 = float(max(0.0, min(1.0, (best_cos_sim + 1) / 2)))
        matched = best_cos_sim >= thresh and best_customer_id is not None
        rec_status = 1 if matched else 0

        log = RecognitionLog(
            customer_id=best_customer_id if matched else None,
            similarity=similarity_01,
            camera_id=camera_id or None,
            status=rec_status
        )
        db.add(log)
        db.commit()
        db.close()
    except Exception as e:
        return {"error": f"Lỗi database: {str(e)}"}

    if matched:
        return {
            "status": "Success",
            "matched_customer_id": best_customer_id,
            "cosine_similarity": round(best_cos_sim, 4),
            "similarity": round(similarity_01, 4),
            "threshold_used": thresh,
            "mode": face_service_mode,
        }
    else:
        return {
            "status": "Fail",
            "message": "Không tìm thấy khuôn mặt khớp",
            "cosine_similarity": round(best_cos_sim, 4),
            "similarity": round(similarity_01, 4),
            "threshold_used": thresh,
            "mode": face_service_mode,
        }
