from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Column, Integer, BigInteger, Float, String, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# MySQL - cấu hình kết nối
MYSQL_USER = "root"
MYSQL_PASSWORD = "NgoHoangNam2004@"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "cinema_ai"

# Quote password để xử lý ký tự đặc biệt (@, #, ...)
_mysql_url = f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
_sqlite_url = "sqlite:///./cinema_faces.db"

try:
    engine = create_engine(_mysql_url, echo=False)
    engine.connect().close()
except Exception:
    engine = create_engine(_sqlite_url, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, nullable=False)
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(BigInteger, nullable=True)  # NULL khi status=0 (Fail)
    similarity = Column(Float, nullable=False)
    camera_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Integer, nullable=False)  # 1=Success, 0=Fail

Base.metadata.create_all(engine)