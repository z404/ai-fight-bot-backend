from core.config import settings
from db.base import Base
from sqlalchemy.orm import Session

from .session import engine


def init_db(db: Session) -> None:
    Base.metadata.create_all(bind=engine)
