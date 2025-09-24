#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

try:
    from app.services.workspace_analysis_service import WorkspaceAnalysisService

    print("✅ WorkspaceAnalysisService import successful")

    from app.services.knowledge_graph_service import KnowledgeGraphService

    print("✅ KnowledgeGraphService import successful")

    from app.services.quiz_service import QuizService

    print("✅ QuizService import successful")

    print("\n🎉 All imports successful! Implementation is ready.")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
