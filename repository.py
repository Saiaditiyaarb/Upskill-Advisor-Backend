from sqlalchemy.orm import Session
from . import models, schemas

class AdviceRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_advice(self, request: schemas.AdviseRequest, result: schemas.AdviseResult):
        db_advice = models.Advice(request_data=request.dict(), result_data=result.dict())
        self.db.add(db_advice)
        self.db.commit()
        self.db.refresh(db_advice)
        return db_advice

    def get_advice(self, advice_id: int):
        return self.db.query(models.Advice).filter(models.Advice.id == advice_id).first()

    def get_all_advice(self, skip: int = 0, limit: int = 100):
        return self.db.query(models.Advice).offset(skip).limit(limit).all()