"""
Knowledge Graph API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..database import get_db
from ..services import KnowledgeGraphService, WorkspaceAnalysisService
from ..models import ConceptCreate, RelationshipCreate, ConceptFileCreate

router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"])


# Pydantic models for API requests/responses
class ConceptResponse(BaseModel):
    concept_id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class RelationshipResponse(BaseModel):
    relationship_id: str
    source_concept_id: str
    target_concept_id: str
    type: str
    strength: Optional[float]
    created_at: str

    class Config:
        from_attributes = True


class ConceptFileResponse(BaseModel):
    concept_file_id: str
    concept_id: str
    file_id: int
    workspace_id: int
    snippet: Optional[str]
    relevance_score: Optional[float]
    last_accessed_at: Optional[str]

    class Config:
        from_attributes = True


class KnowledgeGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class SuggestedTopicsResponse(BaseModel):
    topics: List[Dict[str, Any]]


@router.post("/workspaces/{workspace_id}/analyze")
async def analyze_workspace(
    workspace_id: int,
    force_reanalysis: bool = False,
    file_paths: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger workspace analysis to extract concepts and build knowledge graph
    """
    # Get workspace path from database
    workspace_query = text("SELECT path FROM workspaces WHERE id = :workspace_id")
    result = await db.execute(workspace_query, {"workspace_id": workspace_id})
    workspace_record = result.fetchone()

    if not workspace_record:
        raise HTTPException(status_code=404, detail="Workspace not found")

    workspace_path = workspace_record.path

    # Initialize analysis service
    analysis_service = WorkspaceAnalysisService(db)

    try:
        # Run analysis
        results = await analysis_service.analyze_workspace(
            workspace_id=workspace_id,
            workspace_path=workspace_path,
            force_reanalysis=force_reanalysis,
            file_paths=file_paths,
        )

        return {
            "status": "analysis_completed",
            "task_id": f"analysis_{workspace_id}_{hash(str(file_paths))}",
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/workspaces/{workspace_id}/graph", response_model=KnowledgeGraphResponse)
async def get_workspace_knowledge_graph(
    workspace_id: int,
    depth: int = 2,
    concept_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Get knowledge graph for a specific workspace
    """
    kg_service = KnowledgeGraphService(db)

    if concept_id:
        # Get subgraph around specific concept
        graph = await kg_service.get_concept_graph(concept_id, depth)
    else:
        # Get workspace concepts and build graph
        workspace_concepts = await kg_service.get_workspace_concepts(workspace_id)

        # Build simple graph from concepts
        nodes = []
        edges = []

        # Get relationships for all concepts
        concept_ids = [wc["concept"].concept_id for wc in workspace_concepts]
        for cid in concept_ids:
            relationships = await kg_service.get_relationships_for_concept(cid)
            for rel in relationships:
                if (
                    rel.target_concept_id in concept_ids
                ):  # Only include edges within workspace
                    edges.append(
                        {
                            "id": rel.relationship_id,
                            "source": rel.source_concept_id,
                            "target": rel.target_concept_id,
                            "type": rel.type,
                            "strength": rel.strength,
                        }
                    )

        nodes = [
            {
                "id": wc["concept"].concept_id,
                "name": wc["concept"].name,
                "description": wc["concept"].description,
            }
            for wc in workspace_concepts
        ]

        graph = {"nodes": nodes, "edges": edges}

    return graph


@router.get(
    "/workspaces/{workspace_id}/suggested-topics",
    response_model=SuggestedTopicsResponse,
)
async def get_workspace_suggested_topics(
    workspace_id: int,
    limit: int = 10,
    context_concept_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Get suggested topics for a workspace using hybrid scoring algorithm
    """
    from datetime import datetime, timedelta
    import math

    kg_service = KnowledgeGraphService(db)

    # Get workspace concepts with relevance scores
    workspace_concepts = await kg_service.get_workspace_concepts(workspace_id)

    if not workspace_concepts:
        return {"topics": []}

    # Calculate hybrid scores for each concept
    now = datetime.utcnow()
    tau_days = 7  # Decay constant for recency

    scored_concepts = []
    for wc in workspace_concepts:
        concept = wc["concept"]
        relevance_score = wc["relevance_score"] or 0.5
        last_accessed = wc["last_accessed_at"]

        # Frequency score (normalized relevance)
        F = relevance_score

        # Recency score (exponential decay)
        if last_accessed:
            delta_days = (now - last_accessed).days
            R = math.exp(-delta_days / tau_days)
        else:
            R = 0.1  # Low score for never accessed

        # Semantic similarity score (simplified - would use embeddings in production)
        # For now, use concept relationships as proxy
        relationships = await kg_service.get_relationships_for_concept(
            concept.concept_id
        )
        centrality = len(relationships) / max(1, len(workspace_concepts))
        S = min(0.8, centrality * 2)  # Scale centrality to similarity score

        # Hybrid score with weights (configurable)
        w_f, w_r, w_s = 0.4, 0.3, 0.3  # frequency, recency, semantic
        total_score = w_f * F + w_r * R + w_s * S

        scored_concepts.append(
            {
                "concept": concept,
                "score": total_score,
                "frequency_score": F,
                "recency_score": R,
                "semantic_score": S,
            }
        )

    # Sort by total score and return top suggestions
    scored_concepts.sort(key=lambda x: x["score"], reverse=True)

    suggestions = []
    for sc in scored_concepts[:limit]:
        suggestions.append(
            {
                "concept_id": sc["concept"].concept_id,
                "name": sc["concept"].name,
                "relevance_score": sc["score"],
                "description": sc["concept"].description,
                "score_breakdown": {
                    "frequency": sc["frequency_score"],
                    "recency": sc["recency_score"],
                    "semantic": sc["semantic_score"],
                },
            }
        )

    return {"topics": suggestions}


@router.get("/concepts/{concept_id}/files")
async def get_concept_files(concept_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get all files linked to a concept
    """
    kg_service = KnowledgeGraphService(db)
    files = await kg_service.get_files_for_concept(concept_id)

    return {
        "files": [
            {
                "concept_file_id": cf.concept_file_id,
                "file_id": cf.file_id,
                "workspace_id": cf.workspace_id,
                "snippet": cf.snippet,
                "relevance_score": cf.relevance_score,
                "last_accessed_at": (
                    cf.last_accessed_at.isoformat() if cf.last_accessed_at else None
                ),
            }
            for cf in files
        ]
    }


@router.get("/global/graph", response_model=KnowledgeGraphResponse)
async def get_global_knowledge_graph(
    depth: int = 2, concept_id: Optional[str] = None, db: AsyncSession = Depends(get_db)
):
    """
    Get global knowledge graph across all workspaces
    """
    # This would aggregate concepts across all workspaces
    # For now, return empty graph
    return {"nodes": [], "edges": []}


@router.get("/global/suggested-topics", response_model=SuggestedTopicsResponse)
async def get_global_suggested_topics(
    limit: int = 10,
    context_concept_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Get globally suggested topics
    """
    # This would aggregate suggestions across all workspaces
    # For now, return empty list
    return {"topics": []}


# CRUD operations for concepts
@router.post("/concepts", response_model=ConceptResponse)
async def create_concept(concept: ConceptCreate, db: AsyncSession = Depends(get_db)):
    """Create a new concept"""
    kg_service = KnowledgeGraphService(db)
    created_concept = await kg_service.create_concept(concept)
    return created_concept


@router.get("/concepts/{concept_id}", response_model=ConceptResponse)
async def get_concept(concept_id: str, db: AsyncSession = Depends(get_db)):
    """Get a concept by ID"""
    kg_service = KnowledgeGraphService(db)
    concept = await kg_service.get_concept(concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    return concept


@router.put("/concepts/{concept_id}", response_model=ConceptResponse)
async def update_concept(
    concept_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Update a concept"""
    kg_service = KnowledgeGraphService(db)
    updated_concept = await kg_service.update_concept(concept_id, name, description)
    if not updated_concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    return updated_concept


@router.delete("/concepts/{concept_id}")
async def delete_concept(concept_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a concept"""
    kg_service = KnowledgeGraphService(db)
    deleted = await kg_service.delete_concept(concept_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Concept not found")
    return {"message": "Concept deleted successfully"}


@router.get("/concepts/search")
async def search_concepts(q: str, limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Search concepts by name or description"""
    kg_service = KnowledgeGraphService(db)
    concepts = await kg_service.search_concepts(q, limit)

    return {
        "concepts": [
            {
                "concept_id": c.concept_id,
                "name": c.name,
                "description": c.description,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in concepts
        ]
    }


# Relationship operations
@router.post("/relationships", response_model=RelationshipResponse)
async def create_relationship(
    relationship: RelationshipCreate, db: AsyncSession = Depends(get_db)
):
    """Create a relationship between concepts"""
    kg_service = KnowledgeGraphService(db)
    created_relationship = await kg_service.create_relationship(relationship)
    return created_relationship


@router.get("/concepts/{concept_id}/relationships")
async def get_concept_relationships(
    concept_id: str, db: AsyncSession = Depends(get_db)
):
    """Get all relationships for a concept"""
    kg_service = KnowledgeGraphService(db)
    relationships = await kg_service.get_relationships_for_concept(concept_id)

    return {
        "relationships": [
            {
                "relationship_id": r.relationship_id,
                "source_concept_id": r.source_concept_id,
                "target_concept_id": r.target_concept_id,
                "type": r.type,
                "strength": r.strength,
                "created_at": r.created_at.isoformat(),
            }
            for r in relationships
        ]
    }


# Concept-file linking operations
@router.post("/concept-files", response_model=ConceptFileResponse)
async def create_concept_file_link(
    link: ConceptFileCreate, db: AsyncSession = Depends(get_db)
):
    """Create a link between a concept and a file"""
    kg_service = KnowledgeGraphService(db)
    created_link = await kg_service.create_concept_file_link(link)
    return created_link
