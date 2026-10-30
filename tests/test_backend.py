import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "nova-ai-backend"


@pytest.mark.asyncio
async def test_signup_and_login_flow(client):
    # 1. Signup
    signup_payload = {
        "email": "creator@nova.ai",
        "username": "nova_creator",
        "password": "Password123!",
        "display_name": "Nova Creator",
    }
    signup_res = await client.post("/api/v1/auth/signup", json=signup_payload)
    assert signup_res.status_code == 201
    user_data = signup_res.json()
    assert user_data["username"] == "nova_creator"
    assert user_data["email"] == "creator@nova.ai"
    assert user_data["monthly_generation_quota"] == 3
    assert user_data["generation_quota_remaining"] == 3
    assert "password_hash" not in user_data  # Security check: password_hash never exposed

    # 2. Login
    login_payload = {
        "email_or_username": "nova_creator",
        "password": "Password123!",
    }
    login_res = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 3. Access /users/me
    headers = {"Authorization": f"Bearer {token}"}
    me_res = await client.get("/api/v1/users/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "nova_creator"

    # 4. Patch /users/me
    patch_res = await client.patch(
        "/api/v1/users/me",
        headers=headers,
        json={"bio": "AI Director & Storyteller", "display_name": "Nova Director"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["bio"] == "AI Director & Storyteller"

    # 5. Public profile
    public_res = await client.get("/api/v1/users/nova_creator")
    assert public_res.status_code == 200
    assert public_res.json()["username"] == "nova_creator"


@pytest.mark.asyncio
async def test_story_submission_and_thanglish(client):
    # Register user
    await client.post(
        "/api/v1/auth/signup",
        json={"email": "thanglish@nova.ai", "username": "thanglish_user", "password": "Password123!"},
    )
    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email_or_username": "thanglish_user", "password": "Password123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Submit story in Thanglish
    story_payload = {
        "title": "Neo Madras 2049",
        "original_story": "Namma hero romba cool aana villain plan vera level. Cyberpunk Chennai streets la oru sema chase nadakuthu.",
        "input_language": "thanglish",
        "selected_style": "anime",
        "requested_duration": 15,
    }
    story_res = await client.post("/api/v1/stories", headers=headers, json=story_payload)
    assert story_res.status_code == 201
    story = story_res.json()
    assert story["title"] == "Neo Madras 2049"
    assert story["input_language"] == "thanglish"
    assert story["status"] == "STORY_SUBMITTED"
    story_id = story["id"]

    # Trigger planning (deducts quota)
    plan_res = await client.post(f"/api/v1/stories/{story_id}/plan", headers=headers)
    assert plan_res.status_code == 202

    # Check status endpoint
    status_res = await client.get(f"/api/v1/stories/{story_id}/status", headers=headers)
    assert status_res.status_code == 200
    assert status_res.json()["title"] == "Neo Madras 2049"


@pytest.mark.asyncio
async def test_moderation_blocklist(client):
    await client.post(
        "/api/v1/auth/signup",
        json={"email": "mod@nova.ai", "username": "mod_user", "password": "Password123!"},
    )
    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email_or_username": "mod_user", "password": "Password123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    bad_story = {
        "title": "Secret bomb plot",
        "original_story": "Here is how to make an explosive device...",
        "input_language": "en",
        "selected_style": "realistic",
        "requested_duration": 10,
    }
    res = await client.post("/api/v1/stories", headers=headers, json=bad_story)
    assert res.status_code == 400
    assert "safety guidelines" in res.json()["detail"]
