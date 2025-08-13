-- init.sql (placed in project root or referenced by docker-compose)
CREATE USER etsi_user WITH PASSWORD 'password';
CREATE DATABASE drift_monitor;
GRANT ALL PRIVILEGES ON DATABASE drift_monitor TO etsi_user;
