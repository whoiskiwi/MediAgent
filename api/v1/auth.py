"""
Email / password auth:

    POST /auth/register  → create account, return JWT
    POST /auth/login     → verify password, return JWT
    GET  /auth/me        → return current user (requires JWT)
"""
import os
import uuid
from datetime import datetime, timezone, timedelta

import boto3
from boto3.dynamodb.conditions import Key as DDBKey
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
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
        raise HTTPException(status_code=409, detail="该邮箱已注册")

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
        raise HTTPException(status_code=401, detail="邮箱或密码错误")

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
