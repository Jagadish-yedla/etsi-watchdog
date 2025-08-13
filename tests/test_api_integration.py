# tests/test_api_integration.py
import requests
import time

BASE = "http://127.0.0.1:5000"

def test_api_register_baseline_evaluate():
    # 1) Register
    r = requests.post(f"{BASE}/api/models", json={"model_id": "it_model", "model_info": {"name":"it_model"}})
    assert r.status_code in (200,201)
    # short pause so server processes DB writes
    time.sleep(0.2)

    # 2) baseline
    r = requests.post(f"{BASE}/api/models/it_model/baseline", json={"baseline_metrics": {"accuracy": 0.99}})
    assert r.status_code == 200

    # 3) evaluate -> give poor predictions to trigger alerts
    r = requests.post(f"{BASE}/api/models/it_model/evaluate", json={
        "y_true": [0,1,0,1,0,1,0,1],
        "y_pred": [0,0,0,0,0,0,0,0]
    })
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("status") == "success"
    ev = payload.get("evaluation_results")
    assert ev and "model_id" in ev
