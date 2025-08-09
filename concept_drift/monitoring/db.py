'''# concept_drift/monitoring/db.py

from pathlib import Path
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load YAML config
cfg_path = Path(__file__).parents[2] / "config" / "drift_monitor_config.yaml"
cfg = yaml.safe_load(open(cfg_path, "r"))

DATABASE_URL = cfg["database"]["url"]

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()'''
# concept_drift/monitoring/db.py

from pathlib import Path
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load YAML config
cfg_path = Path(__file__).parents[2] / "config" / "drift_monitor_config.yaml"
cfg = yaml.safe_load(open(cfg_path, "r"))

# Choose the DB URL
db_cfg = cfg.get("database", {})
if "url" in db_cfg:
    # Use a full URL if provided:
    DATABASE_URL = db_cfg["url"]
else:
    # Otherwise, build it from the nested postgresql settings:
    pg = db_cfg.get("postgresql", {})
    host     = pg.get("host", "localhost")
    port     = pg.get("port", 5432)
    dbname   = pg.get("database", "drift_monitor")
    user     = pg.get("username", "postgres")
    password = pg.get("password", "password")

    DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


# 3) Build the SQLAlchemy URL
DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# 4) Set up SQLAlchemy
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
