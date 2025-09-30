
## 1. Introduction

These coding guidelines define standards and best practices for all backend Python code in Recall. Follow these rules for clarity, safety, maintainability, and production readiness.

---

## 2. Project Structure

* Organize code by feature: `app/`, `models/`, `routes/`, `services/`, `llm_clients/`.
* Place tests in `tests/` and migrations in `migrations/`.
* Use `static/` for static assets.

---

## 3. General Principles

* Write clean, readable, and well-documented code.
* Prefer explicitness over cleverness.
* Optimize for maintainability and safety.

---

## 4. Logging & Error Handling

* Use the `logging` module, not `print()`.
* Use `logger = logging.getLogger(__name__)` in every service and route.
* Log at appropriate levels: `debug`, `info`, `warning`, `error`.
* Always catch and re-raise exceptions as `HTTPException` in API routes.
* Never use bare `except:`; catch specific exceptions.

---

## 5. Type Safety

* Use type hints for all function signatures and class attributes.
* Use `Optional`, `List`, `Dict`, etc. from `typing`.
* Validate types at runtime for API inputs.

---

## 6. Async & Concurrency

* Prefer `async def` for all I/O-bound service and route methods.
* Use `AsyncSession` for DB operations.
* Avoid blocking calls in async code.

---

## 7. Dependency Management

* Pin all dependencies in `backend/requirements.txt`.
---

## 8. API Design

* Use FastAPI for all HTTP APIs.
* Validate and sanitize all user inputs.
* Return clear error messages and status codes.
* Use Pydantic models for request/response schemas.
* Any API Changes, update backend\tests\resources\api_collection.postman_collection.json and run `make test-api` to verify.

---

## 9. Database & Migrations

* Use SQLAlchemy for all DB access.
* Encapsulate DB logic in service classes.
* Use migration scripts for schema changes (`migrations/`).

---

## 10. Security

* Never hardcode credentials or secrets.
* Validate all user inputs for file and workspace operations.
* Use least privilege for file and DB access.

---

## 11. Code Quality & CI

* Run `black` for formatting, `flake8`/`pylint` for linting.
* Enforce pre-commit hooks for formatting, linting, and tests.
* Use GitHub Actions for automated testing.

---

## 12. Testing & Coverage

* All code must have **unit tests** using `pytest`.
* Use fixtures for DB, app, and client setup (`conftest.py`).
* Mock external dependencies (DB, LLM, file system) in tests.
* Target >85% test coverage. All new code must maintain or improve coverage.
* Follow AAA (Arrange-Act-Assert) pattern.

**How to run tests and coverage:**

* Run all tests (backend and frontend):

	```sh
	make test
	```

* Run backend tests only (with coverage):

	```sh
	make test-backend
	```
	- This runs pytest with coverage for the backend (`backend/app`).
	- Coverage threshold is set to 100% by default (see `--cov-fail-under=100`).

* Run backend tests only (without coverage):

	```sh
	set BYPASS_COVERAGE=1
	make test-backend
	```

* Run frontend tests only (with coverage):

	```sh
	make test-frontend
	```

* Run frontend tests only (without coverage):

	```sh
	set BYPASS_COVERAGE=1
	make test-frontend
	```

* View HTML coverage report:
	- After running backend tests with coverage, open `backend/htmlcov/index.html` in your browser.

---



✅ Treat every code change as production-ready.
✅ Optimize for clarity, safety, and maintainability.
✅ Use these rules for all backend Python code in Recall.
