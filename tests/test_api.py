import os
import sys
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

TEST_DB_PATH = os.path.abspath("test_clipmind.db")
if os.path.exists(TEST_DB_PATH):
    try:
        os.remove(TEST_DB_PATH)
    except Exception:
        pass

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["SECRET_KEY"] = "test-secret-key-12345-with-at-least-32-bytes-length"

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app, Base, engine, Session_

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "ClipMind AI"

def test_registration_and_first_admin(client):
    # First user should become admin
    res = client.post("/auth/register", json={
        "email": "admin@clipmind.ai",
        "password": "adminpassword",
        "name": "Super Admin"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == "admin@clipmind.ai"
    assert data["role"] == "admin"

    # Duplicate registration should return 409
    res_dup = client.post("/auth/register", json={
        "email": "admin@clipmind.ai",
        "password": "adminpassword"
    })
    assert res_dup.status_code == 409

    # Second user as creator
    res_creator = client.post("/auth/register", json={
        "email": "creator@clipmind.ai",
        "password": "creatorpass",
        "name": "Course Creator",
        "role": "creator"
    })
    assert res_creator.status_code == 200
    assert res_creator.json()["role"] == "creator"

    # Third user as learner
    res_learner = client.post("/auth/register", json={
        "email": "learner@clipmind.ai",
        "password": "learnerpass",
        "name": "Active Learner",
        "role": "learner"
    })
    assert res_learner.status_code == 200
    assert res_learner.json()["role"] == "learner"

def test_login_and_auth_tokens(client):
    # Bad credentials
    bad_res = client.post("/auth/login", data={"username": "admin@clipmind.ai", "password": "wrongpassword"})
    assert bad_res.status_code == 401

    # Good login
    login_res = client.post("/auth/login", data={"username": "admin@clipmind.ai", "password": "adminpassword"})
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # Test /me
    me_res = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "admin@clipmind.ai"

def test_video_upload_and_pipeline(client):
    # Login as creator
    creator_login = client.post("/auth/login", data={"username": "creator@clipmind.ai", "password": "creatorpass"})
    creator_token = creator_login.json()["access_token"]
    creator_headers = {"Authorization": f"Bearer {creator_token}"}

    # Learner should not be allowed to upload
    learner_login = client.post("/auth/login", data={"username": "learner@clipmind.ai", "password": "learnerpass"})
    learner_token = learner_login.json()["access_token"]
    learner_headers = {"Authorization": f"Bearer {learner_token}"}

    dummy_content = b"\x00\x00\x00\x20ftypmp42" + b"demo video content for clipmind ai testing" * 20
    file_payload = {"file": ("test_presentation.mp4", io.BytesIO(dummy_content), "video/mp4")}

    res_forbidden = client.post("/videos", files=file_payload, headers=learner_headers)
    assert res_forbidden.status_code == 403

    # Creator upload
    file_payload = {"file": ("test_lecture.mp4", io.BytesIO(dummy_content), "video/mp4")}
    res_upload = client.post("/videos", files=file_payload, headers=creator_headers)
    assert res_upload.status_code == 200
    video_data = res_upload.json()
    vid = video_data["id"]
    assert video_data["filename"] == "test_lecture.mp4"

    # Fetch detail
    res_detail = client.get(f"/videos/{vid}", headers=creator_headers)
    assert res_detail.status_code == 200
    det = res_detail.json()
    assert det["views"] >= 1
    assert "transcript" in det
    assert "summary_short" in det
    assert "moments" in det

    # Search in video
    res_search = client.get(f"/videos/{vid}/search?q=ClipMind", headers=creator_headers)
    assert res_search.status_code == 200
    assert isinstance(res_search.json(), list)

    # Downloads / exports
    for kind in ["transcript", "summary", "vtt", "srt", "json"]:
        res_dl = client.get(f"/videos/{vid}/download?kind={kind}", headers=creator_headers)
        assert res_dl.status_code == 200

    # Edit transcript
    new_text = "This is an updated lecture about artificial intelligence, neural networks, machine learning algorithms, and deep learning architectures."
    res_edit = client.put(f"/videos/{vid}/transcript", json={"transcript": new_text}, headers=creator_headers)
    assert res_edit.status_code == 200

    # Verify updated keywords
    res_updated = client.get(f"/videos/{vid}", headers=creator_headers)
    kw_words = [k["word"] for k in res_updated.json()["keywords"]]
    assert any("learning" in w or "neural" in w or "intelligence" in w for w in kw_words)

def test_bookmarks(client):
    learner_login = client.post("/auth/login", data={"username": "learner@clipmind.ai", "password": "learnerpass"})
    learner_token = learner_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {learner_token}"}

    # Get available video
    v_res = client.get("/videos", headers=headers)
    videos = v_res.json()
    assert len(videos) > 0
    vid = videos[0]["id"]

    # Add bookmark
    bm_res = client.post("/bookmarks", json={"video_id": vid, "at_sec": 42, "note": "Key takeaway on deep learning"}, headers=headers)
    assert bm_res.status_code == 200
    bm_id = bm_res.json()["id"]

    # List bookmarks
    list_res = client.get("/bookmarks", headers=headers)
    assert list_res.status_code == 200
    bms = list_res.json()
    assert any(b["id"] == bm_id for b in bms)

    # Delete bookmark
    del_res = client.delete(f"/bookmarks/{bm_id}", headers=headers)
    assert del_res.status_code == 200

def test_analytics(client):
    creator_login = client.post("/auth/login", data={"username": "creator@clipmind.ai", "password": "creatorpass"})
    headers = {"Authorization": f"Bearer {creator_login.json()['access_token']}"}

    res = client.get("/analytics", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "videos" in data
    assert "ready" in data
    assert "total_views" in data
    assert "top_keywords" in data

def test_admin_features(client):
    # Non-admin access denied
    learner_login = client.post("/auth/login", data={"username": "learner@clipmind.ai", "password": "learnerpass"})
    res_deny = client.get("/admin/users", headers={"Authorization": f"Bearer {learner_login.json()['access_token']}"})
    assert res_deny.status_code == 403

    # Admin access allowed
    admin_login = client.post("/auth/login", data={"username": "admin@clipmind.ai", "password": "adminpassword"})
    admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}

    users_res = client.get("/admin/users", headers=admin_headers)
    assert users_res.status_code == 200
    user_list = users_res.json()
    assert len(user_list) >= 3

    # Change role of learner to educator
    learner_user = next(u for u in user_list if u["email"] == "learner@clipmind.ai")
    patch_res = client.patch(f"/admin/users/{learner_user['id']}?role=educator", headers=admin_headers)
    assert patch_res.status_code == 200
    assert patch_res.json()["role"] == "educator"

    # Audit log
    audit_res = client.get("/admin/audit", headers=admin_headers)
    assert audit_res.status_code == 200
    assert len(audit_res.json()) > 0
