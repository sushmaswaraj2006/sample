"""
ASTRA — JWT Authentication Module
Provides token creation, verification, and dependency injection for FastAPI.
Features graceful HMAC-SHA256 fallback when python-jose is not installed.
"""

import os
import json
import base64
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("ASTRA_JWT_SECRET", "astra-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# Check if passlib / bcrypt is available; fallback to standard salted sha256
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)
    def hash_password(plain: str) -> str:
        return pwd_context.hash(plain)
except Exception:
    def hash_password(plain: str) -> str:
        return hashlib.sha256((plain + SECRET_KEY).encode()).hexdigest()
    def verify_password(plain: str, hashed: str) -> bool:
        return hash_password(plain) == hashed

# ─────────────────────────────────────────────────────────────────────────────
# DEMO USER STORE
# ─────────────────────────────────────────────────────────────────────────────
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "full_name": "ASTRA Mission Commander",
        "role": "admin",
        "hashed_password": hash_password("astra2024"),
        "disabled": False,
    },
    "analyst": {
        "username": "analyst",
        "full_name": "Orbital Conjunction Analyst",
        "role": "analyst",
        "hashed_password": hash_password("analyst123"),
        "disabled": False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    username: str
    role: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class User(BaseModel):
    username: str
    full_name: str
    role: str
    disabled: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# JWT IMPLEMENTATION (Native + Jose compatible)
# ─────────────────────────────────────────────────────────────────────────────
def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

def _b64decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": int(expire.timestamp())})
    header = {"alg": "HS256", "typ": "JWT"}

    header_b64 = _b64encode(json.dumps(header).encode())
    payload_b64 = _b64encode(json.dumps(to_encode).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64encode(signature)}"

def decode_access_token(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, _b64decode(sig_b64)):
        raise ValueError("Invalid signature")
    payload = json.loads(_b64decode(payload_b64).decode())
    exp = payload.get("exp")
    if exp and datetime.now(timezone.utc).timestamp() > exp:
        raise ValueError("Token expired")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION HELPERS & DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
def get_user(username: str) -> Optional[dict]:
    return DEMO_USERS.get(username)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    if token is None:
        return None
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        role: str = payload.get("role", "analyst")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    user_dict = get_user(username)
    if user_dict is None or user_dict.get("disabled"):
        raise credentials_exception

    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

async def require_auth(current_user: Optional[User] = Depends(get_current_user)) -> User:
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

async def require_admin(current_user: User = Depends(require_auth)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current_user
