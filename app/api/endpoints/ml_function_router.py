from .ml_functions import taskmapper

from typing import Any, List

import crud
import models
import schemas
from api import deps
from core.config import settings
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

router = APIRouter()

@router.post("/runtask")
def run_ml_task(
    task_id: int,
    parameters: dict,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Run ML task.
    """
    identifier =  dict(jsonable_encoder(current_user))["id"]
    parameters["identifier"] = identifier
    task_output = taskmapper.get_task_function(task_id, parameters)
    #__________________________________________________________________________
    # TODO: Handle task output to database and other applicable areas
    #__________________________________________________________________________
    return task_output

@router.get("/gettaskdetail")
def get_task_detail(
    task_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Gets task requirements of a given task.
    """
    return taskmapper.get_task_detail(task_id)
    