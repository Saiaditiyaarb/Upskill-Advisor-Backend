from sqlalchemy import Column, Integer, String, JSON
from .database import Base

class Advice(Base):
    __tablename__ = "advice"

    id = Column(Integer, primary_key=True, index=True)
    request_data = Column(JSON)
    result_data = Column(JSON)