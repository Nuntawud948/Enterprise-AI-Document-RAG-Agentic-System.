import uuid
import time
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api_router import api_router
from app.core.config import settings
from app.core.exceptions import (
    BaseAppException,
    app_exception_handler,
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.shared.dtos.api_response import APIResponse

# Initialize FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 1. Request ID Correlation Middleware
@app.middleware("http")
async def request_correlation_middleware(request: Request, call_next):
    # Extract existing Request ID or generate a new UUID4
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    
    # Attach tracking headers to HTTP response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# 2. CORS Middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 3. Register Global Exception Handlers
app.add_exception_handler(BaseAppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# 4. Mount Central API Router
app.include_router(api_router, prefix=settings.API_V1_STR)


# 5. Production Health Check Probes
@app.get("/healthz", response_model=APIResponse[dict], tags=["System Health Probes"])
async def health_check(request: Request) -> APIResponse[dict]:
    """Base health check returning system status and version."""
    request_id = getattr(request.state, "request_id", None)
    return APIResponse.success_response(
        data={
            "app": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
            "status": "healthy",
            "version": "1.0.0"
        },
        code=status.HTTP_200_OK,
        request_id=request_id
    )


@app.get("/healthz/liveness", tags=["System Health Probes"])
async def liveness_probe() -> dict:
    """Kubernetes / Cloud Run Liveness probe."""
    return {"status": "alive"}


@app.get("/healthz/readiness", tags=["System Health Probes"])
async def readiness_probe() -> dict:
    """Kubernetes / Cloud Run Readiness probe."""
    return {"status": "ready"}
