# 📖 Alembic Database Migrations & Runbook

## Overview
This directory contains Alembic database migration scripts for the Enterprise AI Document RAG System.
Database migrations are executed asynchronously using `asyncpg` and include full support for the PostgreSQL `pgvector` extension and HNSW indexes.

## Quick CLI Cheat Sheet

### 1. View Status & History
```bash
alembic current
alembic history --verbose
```

### 2. Preview SQL without applying (Offline / Dry-run)
```bash
alembic upgrade head --sql
alembic downgrade -1 --sql
```

### 3. Apply Migrations
```bash
alembic upgrade head
```

### 4. Rollback
```bash
alembic downgrade -1
alembic downgrade ef19bb04737f
```

### 5. Generate New Migration
```bash
alembic revision -m "description_of_change"
alembic revision --autogenerate -m "auto_detect_change"
```

## Zero-Downtime Migration Pattern (Expand & Contract)
1. **Expand**: Add columns as nullable or with default values (`server_default`). Never add breaking constraints on a live database.
2. **Backfill**: Run asynchronous scripts to update historical records.
3. **Contract**: Once all application pods are updated to read/write the new column, enforce `NOT NULL` or drop legacy columns in a subsequent migration.
