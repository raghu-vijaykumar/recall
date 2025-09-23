# Knowledge Graph Component Design

## 1. Overview
This document outlines the design for the Knowledge Graph component, which aims to analyze user workspaces, create a graph of knowledge based on user interests, and suggest relevant topics for study and exploration.

## 2. Data Sources and Extraction
*   **Primary Data Source:** All readable text files within user workspaces.
*   **Future Extension:** The system is designed to be extensible to include analysis of images and videos.
*   **Extraction Method:** Concepts and relationships will be extracted from the text content using natural language processing (NLP) techniques.

## 3. Knowledge Graph Schema (SQLite Implementation)

The knowledge graph will be stored in an SQLite database. Below is the proposed schema:

### Table: `concepts`
Represents individual topics or ideas extracted from the workspace.
| Column Name   | Data Type | Constraints               | Description                                    |
| ------------- | --------- | ------------------------- | ---------------------------------------------- |
| `concept_id`  | TEXT      | PRIMARY KEY               | Unique identifier for the concept (UUID).      |
| `name`        | TEXT      | NOT NULL                  | The name or title of the concept.              |
| `description` | TEXT      |                           | A brief description of the concept (optional). |
| `created_at`  | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Timestamp of concept creation.                 |
| `updated_at`  | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Timestamp of last update.                      |

### Table: `relationships`
Represents the connections between concepts.
| Column Name         | Data Type | Constraints                                   | Description                                                                            |
| ------------------- | --------- | --------------------------------------------- | -------------------------------------------------------------------------------------- |
| `relationship_id`   | TEXT      | PRIMARY KEY                                   | Unique identifier for the relationship (UUID).                                         |
| `source_concept_id` | TEXT      | NOT NULL, FOREIGN KEY (`concepts.concept_id`) | The `concept_id` of the source concept.                                                |
| `target_concept_id` | TEXT      | NOT NULL, FOREIGN KEY (`concepts.concept_id`) | The `concept_id` of the target concept.                                                |
| `type`              | TEXT      | NOT NULL                                      | Type of relationship (e.g., "relates_to", "dives_deep_to").                            |
| `strength`          | REAL      |                                               | A numerical value indicating the strength or relevance of the relationship (optional). |
| `created_at`        | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP                     | Timestamp of relationship creation.                                                    |

### Table: `concept_files`
Links concepts to the specific files where they are mentioned or discussed.
| Column Name        | Data Type | Constraints                                       | Description                                                                   |
| ------------------ | --------- | ------------------------------------------------- | ----------------------------------------------------------------------------- |
| `concept_file_id`  | TEXT      | PRIMARY KEY                                       | Unique identifier for the link (UUID).                                        |
| `concept_id`       | TEXT      | NOT NULL, FOREIGN KEY (`concepts.concept_id`)     | The `concept_id` linked to the file.                                          |
| `file_id`          | TEXT      | NOT NULL, FOREIGN KEY (`files.file_id`)           | The `file_id` from the existing `files` table.                                |
| `workspace_id`     | TEXT      | NOT NULL, FOREIGN KEY (`workspaces.workspace_id`) | The `workspace_id` from the existing `workspaces` table.                      |
| `snippet`          | TEXT      |                                                   | A relevant text snippet from the file where the concept appears.              |
| `relevance_score`  | REAL      |                                                   | A score indicating how relevant the concept is to this specific file snippet. |
| `last_accessed_at` | TIMESTAMP |                                                   | Timestamp when the file was last accessed in relation to this concept.        |

### Existing Tables (for reference and integration):
*   `files`: Contains metadata about files in the workspace (e.g., `file_id`, `path`, `content_hash`).
*   `workspaces`: Contains metadata about user workspaces (e.g., `workspace_id`, `name`).

## 4. Topic Suggestion Algorithms
The system will employ a "T-shaped understanding" approach for topic suggestions, combining broad and deep exploration. Suggestions will be based on:
*   **Frequency:** Most frequently occurring concepts within a workspace or across all workspaces.
*   **Recency:** Concepts related to recently accessed or modified files.
*   **Knowledge Gaps:** Identifying concepts that are related to existing concepts in the user's graph but have not yet been deeply explored or are underrepresented in their workspace content.
*   **Hierarchical Exploration:** Utilizing "dives_deep_to" relationships to suggest more specific sub-topics, and "relates_to" for broader, related areas.
*   **File Linkage:** An "appendix-like" feature will link suggested topics back to relevant files in the workspace, providing context and resources for further study.

## 5. Backend API Endpoints

### 5.1. Trigger Workspace Analysis
*   **Endpoint:** `POST /api/workspaces/{workspace_id}/analyze`
*   **Purpose:** Initiates the process of analyzing a specific user's workspace to extract concepts and build/update its knowledge graph. Supports incremental updates.
*   **Request Body:**
    ```json
    {
        "force_reanalysis": false, // Optional: boolean, if true, re-analyzes all files
        "file_paths": ["/path/to/new_file.txt", "/path/to/modified_file.md"] // Optional: array of file paths for incremental analysis
    }
    ```
*   **Response Body:**
    ```json
    {
        "status": "analysis_started",
        "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
    }
    ```

### 5.2. Retrieve Workspace Knowledge Graph
*   **Endpoint:** `GET /api/workspaces/{workspace_id}/knowledge-graph`
*   **Purpose:** Fetches the knowledge graph for a specific workspace.
*   **Request Parameters:**
    *   `depth`: (Optional) Integer, limits the graph traversal depth.
    *   `concept_id`: (Optional) String (UUID), focuses the graph on a subgraph around a specific concept.
*   **Response Body:**
    ```json
    {
        "nodes": [
            {"id": "concept_id_1", "name": "Concept A", "description": "..." },
            {"id": "concept_id_2", "name": "Concept B", "description": "..." }
        ],
        "edges": [
            {"id": "rel_id_1", "source": "concept_id_1", "target": "concept_id_2", "type": "relates_to", "strength": 0.75 }
        ]
    }
    ```

### 5.3. Get Workspace Suggested Topics
*   **Endpoint:** `GET /api/workspaces/{workspace_id}/suggested-topics`
*   **Purpose:** Provides a list of suggested topics relevant to a specific workspace.
*   **Request Parameters:**
    *   `limit`: (Optional) Integer, maximum number of topics to return.
    *   `context_concept_id`: (Optional) String (UUID), provides suggestions related to a specific concept.
*   **Response Body:**
    ```json
    {
        "topics": [
            {"concept_id": "concept_id_3", "name": "Suggested Topic X", "relevance_score": 0.92},
            {"concept_id": "concept_id_4", "name": "Suggested Topic Y", "relevance_score": 0.88}
        ]
    }
    ```

### 5.4. Retrieve Files by Concept
*   **Endpoint:** `GET /api/concepts/{concept_id}/files`
*   **Purpose:** Returns a list of file metadata associated with a given concept, across all workspaces.
*   **Request Parameters:**
    *   `concept_id`: String (UUID), the ID of the concept.
*   **Response Body:**
    ```json
    {
        "files": [
            {"file_id": "file_id_1", "path": "/path/to/file1.txt", "workspace_id": "ws_id_1", "snippet": "Relevant text snippet from file 1..."},
            {"file_id": "file_id_2", "path": "/path/to/file2.md", "workspace_id": "ws_id_2", "snippet": "Another relevant snippet from file 2..."}
        ]
    }
    ```

### 5.5. Retrieve Global Knowledge Graph
*   **Endpoint:** `GET /api/knowledge-graph/global`
*   **Purpose:** Fetches a consolidated knowledge graph across all user workspaces, representing overall interests.
*   **Request Parameters:**
    *   `depth`: (Optional) Integer, limits the graph traversal depth.
    *   `concept_id`: (Optional) String (UUID), focuses the graph on a subgraph around a specific concept.
*   **Response Body:** (Same structure as Retrieve Workspace Knowledge Graph, but aggregated globally)

### 5.6. Get Global Suggested Topics
*   **Endpoint:** `GET /api/suggested-topics/global`
*   **Purpose:** Provides a list of suggested topics based on the user's overall interests across all workspaces.
*   **Request Parameters:**
    *   `limit`: (Optional) Integer, maximum number of topics to return.
    *   `context_concept_id`: (Optional) String (UUID), provides suggestions related to a specific concept.
*   **Response Body:** (Same structure as Get Workspace Suggested Topics, but aggregated globally)

## 6. Frontend Component Design

### 6.1. Knowledge Graph Visualization
*   **Type:** Interactive force-directed graph (e.g., using D3.js or similar library).
*   **Nodes:** Represent concepts, visually distinct (e.g., by color, size) based on relevance or type.
*   **Edges:** Represent relationships, with labels for `relates_to` and `dives_deep_to`.
*   **Interactivity:**
    *   **Clicking a node:** Expands its immediate connections (if not already visible) and populates the "Associated Files Sidebar" with relevant files.
    *   **Hovering over a node/edge:** Displays a tooltip with more details (e.g., concept description, relationship strength).
    *   **Dragging nodes:** Allows users to rearrange the graph for better readability.
    *   **Zoom/Pan:** Standard graph navigation.

### 6.2. Suggested Topics UI
*   **Location:** A separate, scrollable list component, likely positioned adjacent to the knowledge graph.
*   **Content:** Displays a prioritized list of suggested topics (from `GET /api/workspaces/{workspace_id}/suggested-topics` or `GET /api/suggested-topics/global`).
*   **Interactivity:**
    *   **Clicking a suggested topic:**
        *   Highlights the corresponding concept in the knowledge graph (if present).
        *   Triggers the display of associated files in the "Associated Files Sidebar."
        *   Potentially re-centers the graph on the selected topic.

### 6.3. Associated Files Sidebar
*   **Location:** A dedicated sidebar area, appearing when a concept node or suggested topic is selected.
*   **Content:** Lists files (with their paths and relevant snippets) that are linked to the currently selected concept (from `GET /api/concepts/{concept_id}/files`).
*   **Interactivity:**
    *   **Clicking a file entry:** Opens the file in the editor or navigates to the relevant section.
    *   **Scrolling:** Allows viewing all associated files.

## 7. System Architecture Diagram

```mermaid
graph TD
    User[User] -->|Interacts with| Frontend[Frontend Application]
    Frontend -->|Calls API| Backend[Backend Service]

    subgraph Backend Service
        API_GW[API Gateway]
        KG_Service[Knowledge Graph Service]
        Analysis_Service[Workspace Analysis Service]
        DB[Database / Graph Database]
    end

    subgraph Frontend Application
        KG_Viz[Knowledge Graph Visualization]
        Suggested_Topics_UI[Suggested Topics UI]
        File_Sidebar[Associated Files Sidebar]
    end

    API_GW -->|POST /api/workspaces/{workspace_id}/analyze| Analysis_Service
    API_GW -->|GET /api/workspaces/{workspace_id}/knowledge-graph| KG_Service
    API_GW -->|GET /api/workspaces/{workspace_id}/suggested-topics| KG_Service
    API_GW -->|GET /api/concepts/{concept_id}/files| KG_Service
    API_GW -->|GET /api/knowledge-graph/global| KG_Service
    API_GW -->|GET /api/suggested-topics/global| KG_Service

    Analysis_Service -->|Extracts Concepts & Relationships| DB
    KG_Service -->|Retrieves Graph Data| DB

    Frontend --> KG_Viz
    Frontend --> Suggested_Topics_UI
    KG_Viz --> File_Sidebar

    click KG_Viz "Knowledge Graph Visualization: Interactive force-directed graph, node expansion, sidebar for associated files."
    click Suggested_Topics_UI "Suggested Topics UI: Separate, scrollable list of topics."
    click File_Sidebar "Associated Files Sidebar: Displays files linked to selected concepts."
```

I’ve formatted it like the rest of your doc, with headings and tables, so it integrates smoothly and can be extended later.

# GPU-Accelerated Local Implementation Design

This section outlines how the knowledge graph pipeline can leverage local GPU resources for **concept extraction**, **relationship building**, and **topic suggestions**, while remaining SQLite-based and offline-first.

---

## 1. Workspace Analysis (GPU-based)

### Design Options

| Design                                 | Method                                                                                                             | Pros                                                                 | Cons                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **A. Embedding-based (Sentence-BERT)** | Extract concepts using spaCy/KeyBERT → encode with `sentence-transformers` → compute similarity for relationships. | Semantic similarity, efficient with GPU, scalable for many concepts. | Needs embedding infra, relationships are similarity-based (not hierarchical). |
| **B. LLM-driven extraction**           | Use a local LLM (e.g., LLaMA/Mistral) to extract concepts + label relationships (`relates_to`, `dives_deep_to`).   | Richer, hierarchical links, domain-flexible.                         | Slower, more GPU-hungry, may hallucinate relationships.                       |
| **C. Hybrid (Recommended)**            | Use embeddings for base similarity + optional LLM pass for refinement/hierarchies.                                 | Balanced accuracy + performance, explainable, extensible to cloud.   | Requires caching to avoid re-processing unchanged files.                      |

### Example (Python, Sentence-BERT)

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

concepts = ["Kafka Streams", "Spark Structured Streaming", "Message Queue"]
embeddings = model.encode(concepts, convert_to_tensor=True)

# Compute relationships by similarity
for i in range(len(concepts)):
    for j in range(i+1, len(concepts)):
        score = util.cos_sim(embeddings[i], embeddings[j])
        if score > 0.5:
            print(concepts[i], "relates_to", concepts[j], ":", float(score))

## 2. Knowledge Graph Storage (SQLite + Vectors)

The system remains SQLite-based but with optional vector storage.

### Design Options

| Design             | Method                                                                                 | Pros                                     | Cons                                         |
| ------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| **A. SQLite only** | Store concepts/relationships as before, add embedding BLOB column (with `sqlite-vec`). | Simple, portable, single DB file.        | Embedding search less performant than FAISS. |
| **B. Hybrid**      | SQLite for metadata + FAISS (or Qdrant) for vector similarity.                         | Efficient large-scale similarity search. | Two systems to maintain.                     |

👉 Recommended: Start with SQLite only; upgrade to hybrid if datasets grow large.

## 3. Topic Suggestion (GPU-enhanced)

### Design Options

| Design                                  | Method                                                                                    | Pros                                          | Cons                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------- |
| **A. Frequency + Recency (baseline)**   | Rank concepts by occurrence + last accessed time.                                         | Simple, explainable, fast.                    | Ignores semantic relationships. |
| **B. Embedding Neighbors**              | Suggest concepts with highest embedding similarity to recently accessed ones.             | Surfaces semantically related topics.         | May return noisy neighbors.     |
| **C. Graph + Embeddings (Recommended)** | Combine graph traversal (`relates_to`, `dives_deep_to`) with semantic similarity ranking. | Balanced: structure + meaning, works offline. | More complex scoring logic.     |
| **D. LLM Reasoning**                    | Ask local LLM to generate personalized suggestions based on graph context.                | Adaptive, can infer unexplored gaps.          | Heavy GPU use, slower.          |
## 4. Local GPU-Optimized Pipeline (MVP)

### Electron Backend → Python Worker

*   Extract concepts with spaCy/KeyBERT.
*   Encode embeddings with Sentence-BERT (GPU).
*   Insert/update into SQLite (concepts, relationships, concept_files, embedding).

### Knowledge Graph Storage

*   SQLite schema + optional embedding column.
*   Vector similarity handled via `sqlite-vec`.

### Suggestions

*   Start with Frequency + Recency.
*   Enhance with semantic neighbors from embeddings.
*   Optional: LLM refinement for hierarchies.

### Frontend

*   D3.js graph visualization.
*   Suggested topics list (sorted by frequency + semantic relevance).
*   Sidebar showing associated files.

## 5. Future Cloud Upgrade Path

*   Concept/relationship extraction: move from local models → hosted LLMs (Claude/Gemini).
*   Graph storage: migrate SQLite → Postgres + PGVector or Neo4j.
*   Suggestions: add LLM personalization layer for deeper reasoning.

The local-first design ensures portability and offline functionality, while keeping migration paths open for cloud scale.


---

Do you want me to also add a **concrete scoring formula** for the hybrid suggestion algorithm (Frequency + Recency + Embedding similarity), so you can implement ranking consistently?

You said:
yes
ChatGPT said:

Perfect 👍 — here’s a Markdown section you can drop in after your Topic Suggestion section.
It documents a hybrid scoring formula that combines Frequency, Recency, and Embedding Similarity into one ranking system.

## 3.1 Hybrid Topic Suggestion Scoring

To balance **explainability** (Frequency + Recency) with **semantic intelligence** (Embedding Similarity), suggested topics are ranked using a weighted scoring formula.

### Formula

For each candidate concept **C** in workspace **W**:



Score(C) = w_f * F(C) + w_r * R(C) + w_s * S(C)


Where:
- **F(C): Frequency score**  
  Normalized count of occurrences of concept **C** across workspace files.  


F(C) = count(C) / max_count

- **R(C): Recency score**  
Based on last accessed/modified time of files linked to concept **C**.  
Recent concepts get higher values.  


R(C) = exp(-Δt / τ)

- Δt = time since last access (in hours/days).  
- τ = decay constant (e.g., 7 days).  
- **S(C): Semantic similarity score**  
Embedding similarity between **C** and user’s **current focus set** (recently opened concepts/files).  


S(C) = max_cosine_similarity(C, focus_set)


### Default Weights
- **w_f = 0.4** (Frequency: ensures common concepts are surfaced)  
- **w_r = 0.3** (Recency: prioritizes what the user touched recently)  
- **w_s = 0.3** (Semantic neighbors: brings in related but unexplored concepts)  

Weights can be tuned depending on user preference:
- Higher **w_f** → more stable, popular concepts.  
- Higher **w_r** → time-sensitive suggestions.  
- Higher **w_s** → more exploratory, semantic expansion.  

### Example (Python)

```python
import math
from sentence_transformers import util

def compute_score(concept, max_count, now, focus_embeddings,
                w_f=0.4, w_r=0.3, w_s=0.3, tau_days=7):
  # Frequency (normalized)
  F = concept["count"] / max_count

  # Recency (exponential decay)
  delta_days = (now - concept["last_accessed"]).days
  R = math.exp(-delta_days / tau_days)

  # Semantic similarity (max cosine to focus set)
  emb = concept["embedding"]
  S = max(util.cos_sim(emb, f).item() for f in focus_embeddings)

  return w_f*F + w_r*R + w_s*S

### Benefits

*   **Balanced:** Combines objective (frequency), contextual (recency), and semantic (embedding) signals.
*   **Explainable:** Each component can be shown to the user.
*   **Extendable:** Additional features (e.g., graph centrality, LLM reasoning) can be added as new terms in the formula.


---

Would you like me to also document a **visual explanation** (like a radar chart or stacked bar) that can be shown in the UI to explain to users *why* a topic was suggested? That would make the system more transparent and engaging.Could you explain how does workspace analysis work, how is knowledge graph created and hw get suggested topics work 