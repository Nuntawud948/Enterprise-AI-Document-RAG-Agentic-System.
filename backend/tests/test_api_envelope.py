import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_healthz_endpoint_returns_standardized_envelope():
    """
    Verify that /healthz adheres strictly to the Enterprise APIResponse contract:
    {
      "code": 200,
      "status": "success",
      "data": { ... },
      "errors": null,
      "meta": { "request_id": ..., "timestamp": ... }
    }
    """
    response = client.get("/healthz", headers={"X-Request-ID": "test-req-12345"})
    assert response.status_code == 200
    
    body = response.json()
    assert body["code"] == 200
    assert body["status"] == "success"
    assert body["errors"] is None
    assert body["data"]["status"] == "healthy"
    assert body["meta"]["request_id"] == "test-req-12345"
    assert "timestamp" in body["meta"]
    assert response.headers.get("X-Request-ID") == "test-req-12345"
    assert "X-Process-Time" in response.headers


def test_validation_error_returns_standardized_envelope():
    """
    Verify that input validation failures return HTTP 422 with the standardized error envelope.
    """
    # Send empty body to /api/v1/auth/register
    response = client.post("/api/v1/auth/register", json={})
    assert response.status_code == 422
    
    body = response.json()
    assert body["code"] == 422
    assert body["status"] == "error"
    assert body["data"] is None
    assert isinstance(body["errors"], list)
    assert len(body["errors"]) > 0
    assert "meta" in body
