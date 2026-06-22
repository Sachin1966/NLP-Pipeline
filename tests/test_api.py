import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.connection import init_db, SessionLocal
from src.database.models import User

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    
    # Pre-test cleanup
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "testapi").first()
        if user:
            db.delete(user)
            db.commit()
    finally:
        db.close()

    yield

    # Post-test cleanup
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "testapi").first()
        if user:
            db.delete(user)
            db.commit()
    finally:
        db.close()

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_auth_register_and_login():
    # Register
    reg_response = client.post("/auth/register?username=testapi&password=apipassword&role=admin")
    assert reg_response.status_code == 200
    assert reg_response.json()["username"] == "testapi"
    
    # Login
    login_response = client.post("/auth/login?username=testapi&password=apipassword")
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()
    token = login_response.json()["access_token"]
    
    # Secured call
    headers = {"Authorization": f"Bearer {token}"}
    search_response = client.get("/search?query=battery&limit=2", headers=headers)
    assert search_response.status_code == 200
    assert "results" in search_response.json()
