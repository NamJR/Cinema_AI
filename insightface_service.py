"""
Face recognition services - 2 modes:
1. InsightFaceService: Dùng thư viện insightface (buffalo_l = SCRFD + ArcFace R100) - state-of-the-art
2. FallbackONNXService: Dùng ONNX models thủ công (scrfd + arcface_r100.onnx) - không cần insightface
Cả 2 implement cùng interface: is_ready(), get_error(), detect_and_embed(img)
"""
import os
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# InsightFaceService - dùng thư viện insightface (buffalo_l)
# ---------------------------------------------------------------------------
class InsightFaceService:
    """
    Dùng insightface FaceAnalysis với model buffalo_l.
    buffalo_l = SCRFD-10G (detector) + ArcFace R100 (embedder)
    Embedding 512-d, L2-normalized, cosine similarity.
    """

    def __init__(self, model_name: str = "buffalo_l"):
        self.ready = False
        self.error = None
        self._app = None
        self._model_name = model_name
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name=model_name,
                providers=["CPUExecutionProvider"]
            )
            # det_size=(640,640): chuẩn, cân bằng tốc độ/chính xác
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
            self.ready = True
        except ImportError:
            self.error = (
                "insightface chưa cài. Chạy: pip install insightface "
                "(yêu cầu Microsoft C++ Build Tools trên Windows)"
            )
        except Exception as e:
            self.error = str(e)

    def is_ready(self) -> bool:
        return self.ready

    def get_error(self) -> str | None:
        return self.error

    def detect_and_embed(self, img: np.ndarray) -> list[dict]:
        """
        Detect + embed tất cả khuôn mặt trong ảnh.
        Returns: list of {bbox, embedding (512-d L2-norm), det_score}
        """
        if not self.ready or self._app is None:
            return []
        try:
            faces = self._app.get(img)
        except Exception:
            return []

        results = []
        for face in faces:
            emb = face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": emb.tolist(),
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            })
        # Sắp xếp theo det_score giảm dần (mặt rõ nhất lên đầu)
        results.sort(key=lambda x: x["det_score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# FallbackONNXService - dùng ONNX models thủ công
# ---------------------------------------------------------------------------
class FallbackONNXService:
    """
    Fallback khi không có insightface: dùng ONNX SCRFD detector + ArcFace R100 embedder.
    Cần 2 file trong models_dir:
      - scrfd_10g_bnkps.onnx  (hoặc scrfd_2.5g_bnkps.onnx)
      - arcface_r100.onnx
    """

    def __init__(self, models_dir: str):
        self.ready = False
        self.error = None
        self.detector = None
        self.embedder = None
        self.det_input_name = None
        self.emb_input_name = None

        try:
            import onnxruntime as ort
            from face_aligner import align_face_from_bbox  # noqa: F401 - verify import

            # Tìm detector ONNX (scrfd)
            det_candidates = [
                os.path.join(models_dir, "scrfd.onnx"),
                os.path.join(models_dir, "scrfd_10g_bnkps.onnx"),
                os.path.join(models_dir, "scrfd_2.5g_bnkps.onnx"),
                os.path.join(models_dir, "scrfd_500m_bnkps.onnx"),
            ]
            det_model = next((p for p in det_candidates if os.path.exists(p)), None)

            # Tìm embedder ONNX (arcface)
            emb_candidates = [
                os.path.join(models_dir, "arcface.onnx"),
                os.path.join(models_dir, "arcface_r100.onnx"),
                os.path.join(models_dir, "w600k_r50.onnx"),
            ]
            emb_model = next((p for p in emb_candidates if os.path.exists(p)), None)

            if det_model is None:
                raise FileNotFoundError(
                    f"Không tìm thấy SCRFD ONNX trong {models_dir}. "
                    "Chạy script download hoặc cài insightface."
                )
            if emb_model is None:
                raise FileNotFoundError(
                    f"Không tìm thấy arcface.onnx trong {models_dir}."
                )

            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 4
            sess_opts.intra_op_num_threads = 4

            self.detector = ort.InferenceSession(det_model, sess_opts, providers=["CPUExecutionProvider"])
            self.embedder = ort.InferenceSession(emb_model, sess_opts, providers=["CPUExecutionProvider"])
            self.det_input_name = self.detector.get_inputs()[0].name
            self.emb_input_name = self.embedder.get_inputs()[0].name
            self.ready = True
        except ImportError as e:
            self.error = f"Thiếu thư viện: {e}"
        except FileNotFoundError as e:
            self.error = str(e)
        except Exception as e:
            self.error = str(e)

    def is_ready(self) -> bool:
        return self.ready

    def get_error(self) -> str | None:
        return self.error

    # ------------------------------------------------------------------
    # Internal: SCRFD detector
    # ------------------------------------------------------------------
    def _preprocess_detector(self, img: np.ndarray):
        h, w = img.shape[:2]
        size = max(h, w)
        img_pad = cv2.copyMakeBorder(img, 0, size - h, 0, size - w, cv2.BORDER_CONSTANT, value=0)
        img_in = cv2.resize(img_pad, (640, 640))
        img_in = img_in[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, /255
        img_in = np.transpose(img_in, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return img_in, (w, h)

    def _detect_onnx(self, img: np.ndarray) -> list[dict]:
        img_in, (orig_w, orig_h) = self._preprocess_detector(img)
        try:
            outputs = self.detector.run(None, {self.det_input_name: img_in})
        except Exception:
            return [{"bbox": (0, 0, orig_w, orig_h), "det_score": 0.5}]

        results = []
        if outputs and len(outputs[0].shape) == 3:
            for p in outputs[0][0]:
                score = float(p[4])
                if score < 0.4:
                    continue
                x1 = max(0, int(p[0] / 640 * orig_w))
                y1 = max(0, int(p[1] / 640 * orig_h))
                x2 = min(orig_w, int(p[2] / 640 * orig_w))
                y2 = min(orig_h, int(p[3] / 640 * orig_h))
                results.append({"bbox": (x1, y1, x2, y2), "det_score": score})

        if not results:
            results = [{"bbox": (0, 0, orig_w, orig_h), "det_score": 0.3}]

        results.sort(key=lambda x: x["det_score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal: ArcFace embedder
    # ------------------------------------------------------------------
    def _preprocess_embedder(self, face_img: np.ndarray) -> np.ndarray:
        face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) if face_img.shape[2] == 3 else face_img
        face = cv2.resize(face, (112, 112)).astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return face

    def _embed_face(self, face_img: np.ndarray) -> np.ndarray | None:
        try:
            img_in = self._preprocess_embedder(face_img)
            emb = self.embedder.run(None, {self.emb_input_name: img_in})[0].flatten()
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            return emb.astype(np.float32)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_and_embed(self, img: np.ndarray) -> list[dict]:
        if not self.ready:
            return []
        from face_aligner import align_face_from_bbox

        faces = self._detect_onnx(img)
        results = []
        for face_info in faces:
            bbox = face_info["bbox"]
            aligned = align_face_from_bbox(img, bbox)
            emb = self._embed_face(aligned)
            if emb is None:
                continue
            results.append({
                "bbox": list(bbox),
                "embedding": emb.tolist(),
                "det_score": face_info["det_score"],
            })
        return results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def try_create_fallback_service(models_dir: str) -> "FallbackONNXService | None":
    svc = FallbackONNXService(models_dir)
    return svc if svc.is_ready() else None