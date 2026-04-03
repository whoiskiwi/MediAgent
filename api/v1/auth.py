"""
Email / password auth:

    POST /auth/register  → create account, return JWT
    POST /auth/login     → verify password, return JWT
    GET  /auth/me        → return current user (requires JWT)
"""
import os
import uuid
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import boto3
import httpx
from boto3.dynamodb.conditions import Key as DDBKey
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

load_dotenv()

from auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JWT_SECRET       = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

AWS_REGION  = os.getenv("AWS_REGION", "us-west-2")
USERS_TABLE = os.getenv("USERS_TABLE", "medi-agent-users")

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/v1/auth/google/callback")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:8501")

_ddb   = boto3.resource("dynamodb", region_name=AWS_REGION)
_table = _ddb.Table(USERS_TABLE)

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = ""

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_jwt(user_id: str, email: str, name: str) -> str:
    payload = {
        "sub":   user_id,
        "email": email,
        "name":  name,
        "exp":   datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _get_user_by_email(email: str):
    resp = _table.query(
        IndexName="email-index",
        KeyConditionExpression=DDBKey("email").eq(email),
        Limit=1,
    )
    items = resp.get("Items", [])
    return items[0] if items else None


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
@router.post("/register")
def register(body: RegisterRequest):
    if _get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    now     = datetime.now(timezone.utc).isoformat()
    name    = body.name or body.email.split("@")[0]

    _table.put_item(Item={
        "user_id":         user_id,
        "email":           body.email,
        "name":            name,
        "hashed_password": _pwd.hash(body.password),
        "created_at":      now,
        "last_login":      now,
    })

    token = _make_jwt(user_id, body.email, name)
    return JSONResponse({
        "access_token": token,
        "token_type":   "bearer",
        "user": {"user_id": user_id, "email": body.email, "name": name},
    })


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
@router.post("/login")
def login(body: LoginRequest):
    user = _get_user_by_email(body.email)
    if not user or not _pwd.verify(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    _table.update_item(
        Key={"user_id": user["user_id"]},
        UpdateExpression="SET last_login = :t",
        ExpressionAttributeValues={":t": datetime.now(timezone.utc).isoformat()},
    )

    token = _make_jwt(user["user_id"], user["email"], user["name"])
    return JSONResponse({
        "access_token": token,
        "token_type":   "bearer",
        "user": {"user_id": user["user_id"], "email": user["email"], "name": user["name"]},
    })


# ---------------------------------------------------------------------------
# Get current user
# ---------------------------------------------------------------------------
@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user.get("sub"),
        "email":   current_user.get("email"),
        "name":    current_user.get("name"),
    }


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------
@router.get("/google")
def google_login():
    params = urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    })
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@router.get("/google/callback")
async def google_callback(code: str):
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
        )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to obtain access token from Google")

        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_info = user_resp.json()

    email     = user_info.get("email")
    name      = user_info.get("name", email.split("@")[0] if email else "User")
    google_id = user_info.get("id")

    if not email:
        raise HTTPException(status_code=400, detail="Could not retrieve email from Google")

    user = _get_user_by_email(email)
    now  = datetime.now(timezone.utc).isoformat()

    if not user:
        user_id = str(uuid.uuid4())
        _table.put_item(Item={
            "user_id":         user_id,
            "email":           email,
            "name":            name,
            "hashed_password": "",
            "google_id":       google_id,
            "created_at":      now,
            "last_login":      now,
        })
    else:
        user_id = user["user_id"]
        _table.update_item(
            Key={"user_id": user_id},
            UpdateExpression="SET last_login = :t, google_id = :g",
            ExpressionAttributeValues={":t": now, ":g": google_id},
        )

    jwt_token = _make_jwt(user_id, email, name)
    return RedirectResponse(f"{FRONTEND_URL}?token={jwt_token}&name={name}&email={email}")
