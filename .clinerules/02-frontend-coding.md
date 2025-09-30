
# Recall Frontend Coding Guidelines

These guidelines define standards and best practices for all frontend code in Recall. Follow these rules for clarity, maintainability, and production readiness.

---

## 1. Project Structure

- Organize code by feature: `components/`, `core/`, `shared/`, `src/`.
- Place each major UI feature in its own subfolder under `components/` (e.g., `file-explorer/`, `progress/`, `quiz/`, `workspaces/`).
- Use `core/` for shared logic (API, types, theme).
- Use `shared/` for utilities and helpers.
- Entry point: `src/App.tsx`, `src/index.tsx`.
- Static assets: `index.html`, `style.css`.

---

## 2. General Principles

- Write clean, readable, and well-documented code.
- Prefer explicitness over cleverness.
- Optimize for maintainability and safety.
- Use functional React components and hooks.
- Use TypeScript for type safety.

---

## 3. Styling

- Use CSS modules or global styles in `style.css`.
- Prefer theme variables for colors, spacing, and fonts.
- Support both light and dark themes via `ThemeContext`.
- Keep styles consistent and accessible.

---

## 4. State Management

- Use React hooks (`useState`, `useEffect`, `useContext`) for local and shared state.
- Use context providers for global state (e.g., theme, workspace selection).
- Avoid prop drilling; use context or custom hooks.

---

## 5. API & Data Fetching

- Centralize API calls in `core/api.ts`.
- Use async/await for all fetch operations.
- Handle loading, error, and empty states in UI.
- Validate and sanitize all user inputs before sending to backend.

---

## 6. Component Design

- Keep components small and focused.
- Use props and TypeScript interfaces for data contracts.
- Prefer composition over inheritance.
- Use presentational and container components where appropriate.
- Document component props and usage.

---

## 7. Testing

- All code must have **unit tests** using `vitest` and `@testing-library/react`.
- Mock API calls and external dependencies in tests.
- Target >70% test coverage. All new code must maintain or improve coverage.
- Follow AAA (Arrange-Act-Assert) pattern.

**How to run frontend tests and coverage:**

- Run all frontend tests:

	```sh
	make test-frontend
	# or
	npm run test:frontend
	```

- Run frontend tests with coverage:

	```sh
	npm run test:frontend:coverage
	# or
	make test-frontend (without BYPASS_COVERAGE)
	```

- Run all tests (backend and frontend):

	```sh
	make test
	```

---

## 8. Build & Development

- Build frontend using Vite:

	```sh
	make build-frontend
	# or
	npm run build
	```
- Development mode:

	```sh
	make dev
	# or
	npm run dev
	```
- Output is placed in `dist/frontend`.

---

## 9. Code Quality & Linting

- Use TypeScript for all code.
- Use ESLint and Prettier for linting and formatting.
- Enforce pre-commit hooks for formatting, linting, and tests.

---

✅ Treat every code change as production-ready.
✅ Optimize for clarity, safety, and maintainability.
✅ Use these rules for all frontend code in Recall.
