# 📌 Project Improvements and TODOs

This document categorizes the planned features, improvements, and ideas for the app into **Core Features**, **Building Blocks**, **Tech Quality**, and **Future Enhancements**.  
Each section includes short explanations for clarity.

---

## 🟢 Core Features (User-facing functionality)

These are the features that directly add value to users and shape the main experience of the app.

- [ ] **Improve Quiz Generation** – Refine the algorithm to produce diverse and challenging quizzes.  
- [ ] **Advanced Quiz Generation (LLM-powered)** – Use LLMs to create dynamic Q&A, spaced repetition, weak-area analysis, and voice/flashcard modes.  
- [ ] **Knowledge Graph in Progress View** – Build a graph (RAG-based) to map relationships between studied topics and suggest related material.  
- [ ] **File Tree & Tabbed Views** – Add a file explorer with tab support for recently opened documents.  
- [ ] **Rich Content Display** – Embed URLs, YouTube, images, and GIFs (similar to Notion/Jupyter).  
- [ ] **Basic File Operations** – Enable create, rename, move, copy, and delete for files/folders.  
- [ ] **Expanded File Format Support** – Add support for DOCX, PDFs, and other popular study materials.  
- [ ] **Convert Text to Rich View** – Let users upgrade plain text into formatted, interactive content.  
- [ ] **Advanced Search** – Provide powerful search across files, tags, and workspaces.  
- [ ] **Collaboration (Future)** – Real-time multi-user editing and workspace sharing.  
- [ ] **Version Control Integration (Future)** – Pull content directly from Git repositories.  
- [ ] **Customizable Quiz Settings (Future)** – Allow control over quiz difficulty, types, and formats.  
- [ ] **LLM Research Chat Mode (Future)** – Summarize, condense, and document material via LLM chat.  
- [ ] **Curated Content Marketplace (Future)** – Users can import curated subjects or curricula.  

---

## 🟡 Building Blocks / Overhead (Infrastructure)

These are necessary for a production-grade desktop application but don’t directly add user-facing value.

- [ ] **User Authentication** – Enable personalized workspaces and progress tracking.  
- [ ] **Automatic App Updates** – Add seamless update and download mechanisms.  
- [ ] **Automatic Database Migrations** – Handle schema changes safely without data loss.  
- [ ] **UI/UX Enhancements** – Improve polish with splash screens, onboarding flows, and better defaults.  

---

## 🔵 Tech Quality & Maintenance

Improvements to stability, maintainability, and performance.

- [ ] **Robust Error Handling** – Implement consistent error capture across frontend and backend.  
- [ ] **Code Refactoring** – Improve readability, maintainability, and modularity of the codebase.  
- [ ] **Performance Optimization** – Scale to handle large workspaces and heavy datasets smoothly.  
- [ ] **Automated Testing** – Expand test coverage (unit, integration, end-to-end).  

---

## 💡 Future Enhancements & Ideas

These are forward-looking ideas that can elevate the app significantly over time.

### 🔐 Security & Privacy
- [ ] **Encryption at Rest & In Transit** – Secure local database and synced data.  
- [ ] **Role-Based Access Control (RBAC)** – Permissions for collaborative workspaces.  
- [ ] **Audit Logging** – Track activity for debugging and security compliance.  
- [ ] **End-to-End Encryption (Optional)** – Full privacy for sensitive study data.  
- [ ] **Local-Only Mode** – Allow offline usage without cloud sync.  
- [ ] **Anonymized Analytics** – Opt-in telemetry with strong privacy safeguards.  

### ⚙️ Developer Productivity
- [ ] **CI/CD Pipeline Setup** – Automated builds, tests, and releases.  
- [ ] **Code Quality Gates** – Enforce linting, typing, and static analysis in PRs.  
- [ ] **Developer Documentation** – Add contributor guides, API docs, and architecture diagrams.  
- [ ] **Modular Architecture** – Prepare the codebase for plugins and extensions.  
- [ ] **Internationalization (i18n)** – Multi-language support for UI and quizzes.  
- [ ] **Theming/Customization** – Allow users and developers to style and extend UI.  

### 📊 Analytics & Observability
- [ ] **In-App Telemetry (Opt-in)** – Collect usage patterns to guide improvements.  
- [ ] **Error/Crash Reporting** – Integrate tools like Sentry or Bugsnag.  
- [ ] **Performance Metrics** – Track latency, memory use, and database health.  

### 🌐 Extensibility
- [ ] **Plugin/Extension Framework** – Let the community build new quiz types and content integrations.  
- [ ] **API/SDK for Developers** – Reuse the core quiz engine in other apps or scripts.  

### ☁️ Cross-Platform & Sync
- [ ] **Sync Across Devices** – Cloud sync for user data.  
- [ ] **Offline-First Support** – Ensure the app works fully offline with later sync.  
- [ ] **Backup & Restore** – Local export/import or cloud backups.  
- [ ] **Mobile/Desktop Sync** – Seamless handoff between devices.  
- [ ] **Progressive Web App (PWA)** – Lightweight browser-based version.  
- [ ] **Export/Import Workspaces** – Share study content easily.  

### 🧠 Knowledge & Learning Enhancements
- [ ] **Adaptive Learning Paths** – Adjust quizzes/content automatically based on progress.  
- [ ] **Progress Insights Dashboard** – Visualize weak areas, time spent, and learning curves.  
- [ ] **Gamification** – Add badges, streaks, or XP to motivate learners.  
- [ ] **Recommendation Engine** – Suggest next topics using ML models.  

### 🗂 Data Management
- [ ] **Tagging & Metadata** – Add flexible categorization for notes and media.  
- [ ] **Archive/Trash System** – Allow safe recovery of deleted items.  

### 📝 Versioning & History
- [ ] **Versioning for Notes** – Track all changes with rollback support.  
- [ ] **Versioning for Media** – Keep history of attached files like images and PDFs.  
- [ ] **Change History / Diff View** – Visualize differences between versions.  
- [ ] **Version Metadata** – Record author, timestamp, and reason for changes.  
- [ ] **Auto-Save & Drafts** – Save user work continuously with revert-to-last option.  
- [ ] **Conflict Resolution** – Handle version conflicts in multi-device or collaborative setups.  

### 🌍 Ecosystem & Community
- [ ] **Community-Shared Workspaces** – Enable sharing of flashcards or curated content.  
- [ ] **Import from External Tools** – Support importing from Notion, Obsidian, Google Docs, etc.  
- [ ] **Marketplace for Add-ons** – Distribute extensions, new quiz formats, and study packs.  

---

# ✅ Summary

- **Core Features** → What makes the app useful.  
- **Building Blocks** → Necessary infrastructure for production.  
- **Tech Quality** → Stability, testing, and performance.  
- **Future Enhancements** → Security, extensibility, collaboration, analytics, sync, and community ecosystem.  

This structure keeps the TODO list actionable, while also showing a **visionary path forward**.
