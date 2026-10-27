# 🏢 Enterprise AI Document RAG & Agentic System - Backend

Enterprise-grade Modular Monolith (Department Slice) architecture built with **FastAPI**, **SQLAlchemy 2.0 (Async)**, **Pydantic v2**, and **PostgreSQL (pgvector)**.

---

## 🏛️ Architecture Overview

The system employs **Enterprise Modular Monolith (Department Slice)**:
- **High Cohesion:** Each business domain / department is encapsulated under `app/modules/<module_name>/`.
- **Decoupled OOP Layers:** Every module follows strict OOP layering with Interfaces (`abc.ABC`), Services, Repositories, and separated Request/Response DTOs.
- **Multi-Tenancy Strategy:** Shared Database with Department Column for row-level security and high performance.
- **Standardized API Response Envelope:** Every HTTP response follows the exact same JSON structure.

```text
backend/
├── app/
│   ├── core/                          # [Infrastructure Core]
│   │   ├── config.py                  # Settings (pydantic-settings)
│   │   ├── database.py                # Async Engine, SessionLocal, Base
│   │   ├── security.py                # JWT & Modern Direct Bcrypt
│   │   └── exceptions.py              # Custom Exceptions & Global Handlers
│   │
│   ├── shared/                        # [Shared Kernel & Cross-Cutting Concerns]
│   │   ├── clients/                   # External API Clients (GeminiClient)
│   │   ├── dtos/                      # Shared DTOs (APIResponse, Pagination)
│   │   ├── models/                    # Base Audit & Department Scoped Entities
│   │   ├── rag_engine/                # Chunker & Embedder
│   │   ├── repositories/              # Generic BaseRepository & IBaseRepository
│   │   └── tasks/                     # Background Ingestion Worker Tasks
│   │
│   ├── modules/                       # [Department & Domain Slices]
│   │   ├── auth/                      # Authentication & Identity Module
│   │   │   ├── controllers/           # auth_controller.py
│   │   │   ├── dtos/                  # requests/ & responses/ DTOs
│   │   │   ├── interfaces/            # IAuthService, IUserRepository
│   │   │   ├── services/              # AuthService (Pure Business Logic)
│   │   │   ├── repositories/          # UserRepository (Data Access)
│   │   │   └── models/                # User Entity
│   │   │
│   │   ├── documents/                 # Enterprise Documents Module
│   │   │   ├── controllers/           # documents_controller.py
│   │   │   ├── dtos/                  # requests/ & responses/ DTOs
│   │   │   ├── interfaces/            # IDocumentService, IDocumentRepository
│   │   │   ├── services/              # DocumentService
│   │   │   ├── repositories/          # DocumentRepository
│   │   │   └── models/                # Document, DocumentChunk
│   │   │
│   │   ├── hr/                        # Human Resources Department Slice
│   │   │   ├── controllers/           # hr_controller.py
│   │   │   ├── dtos/                  # requests/ & responses/ DTOs
│   │   │   ├── interfaces/            # IHRService, IHRRepository
│   │   │   ├── services/              # HRService
│   │   │   ├── repositories/          # HRRepository
│   │   │   └── models/                # HRPolicy
│   │   │
│   │   └── [future_departments]/      # Add new departments without touching other modules
│   │
│   ├── api_router.py                  # Aggregates module controllers under /api/v1
│   └── main.py                        # FastAPI entrypoint with Correlation & Error Middleware
├── tests/                             # Pytest Suite (Unit & Integration)
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## 📦 Standardized API Contract

All endpoints return the following standard JSON envelope:

```json
{
  "code": 200,
  "status": "success",
  "data": { ... },
  "errors": null,
  "meta": {
    "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "timestamp": "2026-10-27T08:20:00.000000+00:00"
  }
}
```

---

## 🚀 Getting Started

### 1. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. Run Development Server
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation available at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

### 4. Run Test Suite
```bash
pytest
```
