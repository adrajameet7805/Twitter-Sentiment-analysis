import jwt
import time
from datetime import datetime, timedelta
import logging

# In production this should be in an environment variable securely handled.
JWT_SECRET = "AI_DASHBOARD_SUPER_SECRET_KEY_2026_SAAS"
JWT_ALGORITHM = "HS256"

def generate_jwt_token(email, role, remember_me=False):
    """Generates a secure JWT token for the session."""
    try:
        # Default expiration is 12 hours. If remember me, 7 days.
        expiration_hours = 24 * 7 if remember_me else 12
        expiration_time = datetime.utcnow() + timedelta(hours=expiration_hours)
        
        payload = {
            "email": email,
            "role": role,
            "exp": expiration_time,
            "iat": datetime.utcnow()
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return token, expiration_time
    except Exception as e:
        logging.error(f"Error generating JWT: {e}")
        return None, None

def verify_jwt_token(token):
    """Verifies the JWT token and returns the payload if valid."""
    if not token:
        return None
        
    try:
        decoded_payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded_payload
    except jwt.ExpiredSignatureError:
        logging.warning("JWT Token has expired.")
        return None
    except jwt.InvalidTokenError:
        logging.warning("Invalid JWT Token provided.")
        return None
    except Exception as e:
        logging.error(f"General JWT verification error: {e}")
        return None
