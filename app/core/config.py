from pydantic import BaseSettings
import secrets


class Settings(BaseSettings):
    SQLALCHEMY_DATABASE_URL = "sqlite:///./app_db.db"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SECRET_KEY: str = secrets.token_urlsafe(32)
    USERS_OPEN_REGISTRATION: bool = True
    API_V1_STR: str = "/api/v1"


settings = Settings()
