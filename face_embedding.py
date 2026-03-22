import cv2
import numpy as np
import onnxruntime as ort

# Chuẩn ArcFace/InsightFace: (pixel - 127.5) / 128.0
# Input BGR 112x112, output embedding L2-normalized


class FaceEmbedder:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa ảnh face theo ArcFace (w600k_r50):
        - RGB (model được train với RGB, OpenCV mặc định BGR nên cần convert)
        - Resize 112x112
        - (pixel - 127.5) / 128.0
        - HWC -> NCHW
        """
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)

        face = face_img.astype(np.float32)
        face = (face - 127.5) / 127.5  # Chuẩn InsightFace ArcFace
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face

    def get_embedding(self, face_img: np.ndarray) -> list:
        """
        Trích embedding từ ảnh face đã align.
        Return: list embedding đã L2-normalize.
        """
        face = self._preprocess(face_img)

        embedding = self.session.run(None, {self.input_name: face})[0]
        embedding = embedding.flatten().astype(np.float32)

        # L2 normalize - bắt buộc cho so sánh cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding.tolist()