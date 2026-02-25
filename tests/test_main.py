from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_predict():
    with TestClient(app) as client:
        payload = {"features": [0.0] * 30}
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        body = response.json()
        assert "prediction" in body
        assert "probability" in body
