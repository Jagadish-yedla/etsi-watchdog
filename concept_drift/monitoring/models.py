# concept_drift/monitoring/models.py

from sqlalchemy import Column, Integer, String, DateTime, JSON
from .db import Base
from sqlalchemy import Column, Integer, String, DateTime, JSON
from .db import Base
from datetime import datetime

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True, nullable=False)
    info = Column(JSON)
    registered_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"
    id                 = Column(Integer, primary_key=True, index=True)
    timestamp          = Column(DateTime, nullable=False)
    model_id           = Column(String, index=True, nullable=False)
    drift_type         = Column(String, nullable=False)
    severity           = Column(String, nullable=False)
    detection_method   = Column(String, nullable=False)
    confidence_score   = Column(String, nullable=False)
    description        = Column(String, nullable=False)
    metrics            = Column(JSON, nullable=False)
    recommended_actions= Column(JSON, nullable=False)

class Evaluation(Base):
    __tablename__ = "evaluations"
    id        = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    model_id  = Column(String, index=True, nullable=False)
    accuracy  = Column(String, nullable=False)
    metrics   = Column(JSON, nullable=False)
