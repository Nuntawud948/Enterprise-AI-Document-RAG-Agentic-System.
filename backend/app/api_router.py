from fastapi import APIRouter

from app.modules.auth.controllers.auth_controller import router as auth_router
from app.modules.documents.controllers.documents_controller import router as documents_router
from app.modules.hr.controllers.hr_controller import router as hr_router

api_router = APIRouter()

# Include Domain & Department Module Routers
api_router.include_router(auth_router)
api_router.include_router(documents_router)
api_router.include_router(hr_router)
