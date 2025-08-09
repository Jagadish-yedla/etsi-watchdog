# concept_drift/monitoring/migrate.py

from concept_drift.monitoring.db import engine, Base
from concept_drift.monitoring.models import Alert, Evaluation

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created")

