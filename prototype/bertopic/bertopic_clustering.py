#!/usr/bin/env python3
"""
BERTopic Document Topic Modeling Prototype with Knowledge Graph Generation
Enhanced prototype that creates reader-friendly knowledge graphs from topic clusters
Features: context window extraction, hierarchical topic relationships, incremental indexing
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import time
from dataclasses import dataclass
from pathlib import Path
import mimetypes

# Add backend to path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Import our new modules
from knowledge_graph_builder import TopicKnowledgeGraphBuilder
from incremental_index_manager import IncrementalIndexManager, KnowledgeGraphNavigator

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    import umap
    import hdbscan

    # Try to import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from wordcloud import WordCloud

        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        print("Installing visualization packages...")
        os.system("pip install matplotlib seaborn wordcloud")

    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    VISUALIZATION_AVAILABLE = False
    print("Installing required packages...")
    os.system(
        "pip install bertopic umap-learn hdbscan scikit-learn sentence-transformers matplotlib seaborn wordcloud"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for different embedding models"""

    name: str
    model_name: str
    description: str
    min_topic_size: int = 3
    max_topics: Optional[int] = None

    def __str__(self):
        return f"{self.name} ({self.model_name})"


@dataclass
class ModelComparisonResult:
    """Results from comparing multiple models"""

    model_name: str
    num_topics: int
    coherence_score: float
    outlier_count: int
    processing_time: float
    model_config: ModelConfig

    def __str__(self):
        return f"{self.model_name}: {self.num_topics} topics, coherence: {self.coherence_score:.3f}, outliers: {self.outlier_count}, time: {self.processing_time:.2f}s"


class MultiModelTopicModeler:
    """Run topic modeling with multiple embedding models and compare results"""

    def __init__(
        self, base_output_dir: str = "prototype/bertopic", verbose: bool = True
    ):
        """
        Initialize the multi-model topic modeler.

        Args:
            base_output_dir: Base directory to store results for all models
            verbose: Whether to print progress information
        """
        self.base_output_dir = Path(base_output_dir)
        self.verbose = verbose
        self.comparison_results = []

        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Define models to test with updated configuration
        self.model_configs = [
            ModelConfig(
                name="BERT_Base",
                model_name="sentence-transformers/bert-base-nli-mean-tokens",
                description="BERT base model with mean pooling",
                min_topic_size=3,
            ),
            ModelConfig(
                name="E5_Multilingual_Base",
                model_name="intfloat/multilingual-e5-base",
                description="Multilingual E5 base model",
                min_topic_size=3,
            ),
            ModelConfig(
                name="E5_Multilingual_Large",
                model_name="intfloat/multilingual-e5-large",
                description="Large multilingual E5 model with instruction tuning",
                min_topic_size=3,
            ),
            ModelConfig(
                name="Qwen_Embedding_0.6B",
                model_name="Qwen/Qwen3-Embedding-0.6B",
                description="Qwen embedding model with 0.6B parameters",
                min_topic_size=3,
            ),
            ModelConfig(
                name="paraphrase_multilingual_mpnet_base_v2",
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                description="Paraphrase model for multilingual text",
                min_topic_size=3,
            ),
            ModelConfig(
                name="bge_base_en_v1.5",
                model_name="BAAI/bge-base-en-v1.5",
                description="BAAI large English embedding model",
                min_topic_size=3,
            ),
            ModelConfig(
                name="nomic_embed_text_v1.5",
                model_name="nomic-ai/nomic-embed-text-v1.5",
                description="Nomic's mixture of experts text embedding model",
                min_topic_size=3,
            ),
        ]

        if verbose:
            logger.info(
                f"Initialized MultiModelTopicModeler with {len(self.model_configs)} models"
            )
            logger.info(f"Results will be stored in: {self.base_output_dir}")

    def _get_large_models(self) -> List[ModelConfig]:
        """Get configuration for large 8GB-capable models"""
        return [
            ModelConfig(
                name="E5_Large",
                model_name="intfloat/e5-large-v2",
                description="Large E5 model with excellent performance (requires ~8GB)",
                min_topic_size=3,
            ),
            ModelConfig(
                name="MPNet_Large",
                model_name="sentence-transformers/all-mpnet-base-v2",
                description="High-quality MPNet model with strong semantic understanding",
                min_topic_size=3,
            ),
            ModelConfig(
                name="MiniLM_Large",
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                description="Larger MiniLM model with improved performance",
                min_topic_size=3,
            ),
            ModelConfig(
                name="BERT_Large",
                model_name="sentence-transformers/bert-large-nli-mean-tokens",
                description="Large BERT model with mean pooling (requires significant memory)",
                min_topic_size=3,
            ),
            ModelConfig(
                name="RoBERTa_Large",
                model_name="sentence-transformers/all-distilroberta-v1",
                description="DistilRoBERTa model with good balance of speed and quality",
                min_topic_size=3,
            ),
            ModelConfig(
                name="SPECTER_Large",
                model_name="sentence-transformers/paraphrase-MiniLM-L12-v2",
                description="Large paraphrase model optimized for sentence similarity",
                min_topic_size=3,
            ),
        ]

    def _get_standard_models(self) -> List[ModelConfig]:
        """Get configuration for standard smaller models"""
        return [
            ModelConfig(
                name="MiniLM",
                model_name="all-MiniLM-L6-v2",
                description="Lightweight model, fast but less accurate",
                min_topic_size=3,
            ),
            ModelConfig(
                name="MPNet",
                model_name="all-mpnet-base-v2",
                description="Higher quality embeddings, better semantic understanding",
                min_topic_size=3,
            ),
            ModelConfig(
                name="DistilBERT",
                model_name="sentence-transformers/all-distilroberta-v1",
                description="Distilled model with good balance of speed and quality",
                min_topic_size=3,
            ),
        ]

    def run_all_models(
        self,
        documents: List[str],
        create_visualizations: bool = True,
        generate_summaries: bool = True,
    ) -> List[ModelComparisonResult]:
        """
        Run topic modeling with all configured models.

        Args:
            documents: List of document texts
            create_visualizations: Whether to create visualizations for each model
            generate_summaries: Whether to generate summaries for each model

        Returns:
            List of comparison results for all models
        """
        if not documents:
            raise ValueError("No documents provided")

        print("🚀 Starting Multi-Model Topic Modeling Comparison")
        print("=" * 60)
        print(f"📄 Processing {len(documents)} documents")
        print(f"🤖 Testing {len(self.model_configs)} embedding models")
        print(f"💾 Results will be saved to: {self.base_output_dir}")
        print("=" * 60)

        results = []

        for i, model_config in enumerate(self.model_configs, 1):
            print(f"\n🔄 Model {i}/{len(self.model_configs)}: {model_config.name}")
            print(f"   Description: {model_config.description}")
            print(f"   Model: {model_config.model_name}")

            try:
                # Create model-specific output directory
                model_output_dir = self.base_output_dir / model_config.name.lower()
                model_output_dir.mkdir(exist_ok=True)

                # Run topic modeling
                start_time = time.time()

                modeler = DocumentTopicModeler(
                    model_name=model_config.model_name,
                    min_topic_size=model_config.min_topic_size,
                    max_topics=model_config.max_topics,
                    verbose=self.verbose,
                )

                topics, probs = modeler.fit_transform(documents)
                processing_time = time.time() - start_time

                # Calculate metrics
                all_topics = modeler.get_all_topics()
                num_topics = len(all_topics)

                # Calculate simple coherence score (based on topic word scores)
                coherence_score = self._calculate_coherence_score(all_topics)

                # Count outliers
                import pandas as pd

                topic_info_df = pd.DataFrame(modeler.topic_info)
                outlier_count = 0
                if not topic_info_df.empty:
                    outlier_row = topic_info_df[topic_info_df["Topic"] == -1]
                    if not outlier_row.empty:
                        outlier_count = outlier_row["Count"].iloc[0]

                # Store comparison result
                result = ModelComparisonResult(
                    model_name=model_config.name,
                    num_topics=num_topics,
                    coherence_score=coherence_score,
                    outlier_count=outlier_count,
                    processing_time=processing_time,
                    model_config=model_config,
                )
                results.append(result)

                # Save model results
                modeler.save_results(str(model_output_dir))

                if create_visualizations:
                    print(f"   📊 Creating visualizations...")
                    modeler.create_visualizations(str(model_output_dir))

                if generate_summaries:
                    print(f"   📝 Generating summary...")
                    modeler.generate_cluster_summary(str(model_output_dir))

                # Print model summary
                print(f"   ✅ Completed: {result}")

            except Exception as e:
                logger.error(f"Error with model {model_config.name}: {e}")
                print(f"   ❌ Error: {e}")

                # Add failed result
                failed_result = ModelComparisonResult(
                    model_name=model_config.name,
                    num_topics=0,
                    coherence_score=0.0,
                    outlier_count=len(documents),
                    processing_time=0.0,
                    model_config=model_config,
                )
                results.append(failed_result)

        # Create comparison summary
        self._create_comparison_summary(results, documents)

        return results

    def _calculate_coherence_score(
        self, all_topics: Dict[int, Dict[str, Any]]
    ) -> float:
        """Calculate a simple coherence score based on topic word scores"""
        if not all_topics:
            return 0.0

        total_score = 0.0
        total_words = 0

        for topic_info in all_topics.values():
            words = topic_info.get("words", [])
            if words:
                # Average the scores of top words as a simple coherence measure
                scores = [score for word, score in words[:10]]  # Top 10 words
                if scores:
                    total_score += sum(scores)
                    total_words += len(scores)

        return total_score / total_words if total_words > 0 else 0.0

    def _create_comparison_summary(
        self, results: List[ModelComparisonResult], documents: List[str]
    ):
        """Create a comprehensive comparison summary"""
        print("\n" + "=" * 80)
        print("🏆 MULTI-MODEL COMPARISON RESULTS")
        print("=" * 80)

        # Sort by coherence score (descending)
        sorted_results = sorted(results, key=lambda x: x.coherence_score, reverse=True)

        print("\n📊 RANKED BY COHERENCE SCORE:")
        print("-" * 50)

        for i, result in enumerate(sorted_results, 1):
            status = "✅" if result.num_topics > 0 else "❌"
            print(f"{i}. {status} {result}")

        # Find best model
        best_result = sorted_results[0] if sorted_results else None
        if best_result and best_result.num_topics > 0:
            print("\n🎯 RECOMMENDED MODEL:")
            print(
                f"   {best_result.model_name}: {best_result.model_config.description}"
            )
            print(f"   Model: {best_result.model_config.model_name}")
            print(f"   Topics: {best_result.num_topics}")
            print(f"   Coherence: {best_result.coherence_score:.3f}")
            print(f"   Processing time: {best_result.processing_time:.2f}s")

        # Save detailed comparison to file
        self._save_detailed_comparison(results, documents)

    def _save_detailed_comparison(
        self, results: List[ModelComparisonResult], documents: List[str]
    ):
        """Save detailed comparison results to file"""
        comparison_file = self.base_output_dir / "model_comparison.txt"

        with open(comparison_file, "w", encoding="utf-8") as f:
            f.write("BERTopic Multi-Model Comparison Results\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset: {len(documents)} documents\n")
            f.write(f"Models tested: {len(results)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Sort by coherence score
            sorted_results = sorted(
                results, key=lambda x: x.coherence_score, reverse=True
            )

            f.write("RANKED RESULTS (by coherence score):\n")
            f.write("-" * 40 + "\n\n")

            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. {result.model_name}\n")
                f.write(f"   Model: {result.model_config.model_name}\n")
                f.write(f"   Description: {result.model_config.description}\n")
                f.write(f"   Topics: {result.num_topics}\n")
                f.write(f"   Coherence Score: {result.coherence_score:.3f}\n")
                f.write(f"   Outliers: {result.outlier_count}\n")
                f.write(f"   Processing Time: {result.processing_time:.2f}s\n")
                f.write(
                    f"   Status: {'✅ Success' if result.num_topics > 0 else '❌ Failed'}\n\n"
                )

            # Recommendations
            best_result = sorted_results[0] if sorted_results else None
            if best_result and best_result.num_topics > 0:
                f.write("RECOMMENDATION:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best Model: {best_result.model_name}\n")
                f.write(
                    f"Reason: Highest coherence score ({best_result.coherence_score:.3f})\n"
                )
                f.write(f"Topics: {best_result.num_topics}\n")
                f.write(f"Processing Time: {best_result.processing_time:.2f}s\n")

        print(f"\n💾 Detailed comparison saved to: {comparison_file}")

        # Save results as JSON for programmatic access
        json_file = self.base_output_dir / "model_comparison.json"
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(documents),
            "results": [
                {
                    "model_name": r.model_name,
                    "model": r.model_config.model_name,
                    "description": r.model_config.description,
                    "num_topics": r.num_topics,
                    "coherence_score": r.coherence_score,
                    "outlier_count": r.outlier_count,
                    "processing_time": r.processing_time,
                    "status": "success" if r.num_topics > 0 else "failed",
                }
                for r in results
            ],
        }

        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_types(obj):
            if hasattr(obj, "item"):  # numpy types
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        json_data = convert_types(json_data)

        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"💾 JSON results saved to: {json_file}")


class DocumentTopicModeler:
    """
    A simplified BERTopic-based topic modeler for document analysis.
    This is a prototype focused on ease of use and quick experimentation.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 15,
        max_topics: Optional[int] = None,
        diversity: Optional[float] = None,
        verbose: bool = True,
        cluster_epsilon: float = 0.4,
        umap_neighbors: int = 8,
        umap_components: int = 3,
    ):
        """
        Initialize the topic modeler.

        Args:
            model_name: The sentence transformer model to use for embeddings
            min_topic_size: Minimum number of documents per topic
            max_topics: Maximum number of topics to extract
            diversity: Diversity parameter for topic creation (0-1)
            verbose: Whether to print progress information
            cluster_epsilon: HDBSCAN cluster merging threshold (higher = coarser clustering)
            umap_neighbors: UMAP neighborhood size (lower = coarser embeddings)
            umap_components: UMAP output dimensions (lower = more aggressive reduction)
        """
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic and required dependencies are not installed")

        self.model_name = model_name
        self.min_topic_size = min_topic_size
        self.max_topics = max_topics
        self.diversity = diversity
        self.verbose = verbose
        self.cluster_epsilon = cluster_epsilon
        self.umap_neighbors = umap_neighbors
        self.umap_components = umap_components

        # Initialize models (lazy loading)
        self._bertopic_model = None
        self._fitted = False

        # Storage for results
        self.topics = []
        self.topic_info = []
        self.document_topics = {}

        if verbose:
            logger.info(f"Initialized DocumentTopicModeler with model: {model_name}")

    def _get_bertopic_model(self, documents: List[str]) -> BERTopic:
        """Get or create BERTopic model with dynamic parameters based on dataset size"""
        if self._bertopic_model is None:
            if self.verbose:
                logger.info("Creating BERTopic model...")

            # Dynamic parameter adjustment based on dataset size
            n_docs = len(documents)

            # Adjust min_topic_size if dataset is very small
            # HDBSCAN requires min_cluster_size > 1, so ensure at least 2
            effective_min_topic_size = min(self.min_topic_size, max(2, n_docs // 3))

            if self.verbose:
                logger.info(f"Dataset size: {n_docs} documents")
                logger.info(f"Using min_topic_size={effective_min_topic_size}")

            # Choose dimensionality reduction based on dataset size
            if n_docs < 10:
                # Use PCA for very small datasets
                if self.verbose:
                    logger.info(
                        "Using PCA for dimensionality reduction (small dataset)"
                    )

                # For very small datasets, use PCA instead of UMAP
                n_components = min(3, max(1, n_docs - 1))
                umap_model = PCA(n_components=n_components, random_state=42)

                # Adjust HDBSCAN for small datasets
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=effective_min_topic_size,
                    metric="euclidean",
                    cluster_selection_epsilon=self.cluster_epsilon,
                    prediction_data=True,
                )
            else:
                # Use UMAP for larger datasets
                if self.verbose:
                    logger.info("Using UMAP for dimensionality reduction")

                # Use custom UMAP parameters or adjust based on dataset size
                n_neighbors = (
                    min(self.umap_neighbors, max(2, n_docs - 1))
                    if self.umap_neighbors
                    else min(15, max(2, n_docs - 1))
                )
                n_components = (
                    min(self.umap_components, max(2, n_docs - 1))
                    if self.umap_components
                    else min(5, max(2, n_docs - 1))
                )

                if self.verbose:
                    logger.info(
                        f"Using n_neighbors={n_neighbors}, n_components={n_components}"
                    )

                umap_model = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=42,
                )

                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_topic_size,
                    metric="euclidean",
                    cluster_selection_epsilon=self.cluster_epsilon,
                    prediction_data=True,
                )

            # Adjust CountVectorizer parameters for small datasets
            if n_docs < 10:
                # For very small datasets, use absolute counts
                min_df = 1
                max_df = n_docs  # Use all documents as max_df
            else:
                # Use fractional values for better flexibility with large datasets
                min_df = max(1, n_docs // 1000)  # At least 1, or 0.1% of documents
                max_df = 0.95  # Allow terms in up to 95% of documents

            vectorizer_model = CountVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words="english",
                ngram_range=(1, 2),
            )

            representation_model = KeyBERTInspired()

            # Create BERTopic model
            self._bertopic_model = BERTopic(
                embedding_model=self.model_name,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                nr_topics=(
                    self.max_topics
                    if self.max_topics is not None and self.max_topics > 0
                    else None
                ),
                min_topic_size=effective_min_topic_size,
                calculate_probabilities=True,
                verbose=self.verbose,
            )

        return self._bertopic_model

    def fit_transform(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fit the topic model and transform documents to topic assignments.

        Args:
            documents: List of document texts

        Returns:
            Tuple of (topic_assignments, topic_probabilities)
        """
        if not documents:
            raise ValueError("No documents provided")

        if len(documents) < self.min_topic_size:
            logger.warning(
                f"Only {len(documents)} documents provided, minimum is {self.min_topic_size}"
            )

        if self.verbose:
            logger.info(f"Fitting topic model on {len(documents)} documents...")

        # Get BERTopic model with dynamic parameters based on dataset size
        model = self._get_bertopic_model(documents)

        # Fit and transform
        topics, probs = model.fit_transform(documents)

        # Store results
        self.topics = topics
        self.topic_info = model.get_topic_info().to_dict("records")
        self._fitted = True

        # Store document-topic assignments
        for i, (doc, topic_id) in enumerate(zip(documents, topics)):
            # Handle different probability array shapes
            if probs is not None:
                try:
                    # Try to get probability for the specific topic
                    if (
                        hasattr(probs[i], "__getitem__")
                        and len(probs[i]) > topic_id >= 0
                    ):
                        probability = float(probs[i][topic_id])
                    else:
                        probability = 0.5  # Default probability
                except (IndexError, TypeError):
                    probability = 0.5  # Default probability
            else:
                probability = 0.5  # Default probability

            self.document_topics[i] = {
                "document": doc[:100] + "..." if len(doc) > 100 else doc,
                "topic_id": int(topic_id),
                "topic_name": self.get_topic_name(topic_id),
                "probability": probability,
            }

        if self.verbose:
            logger.info(f"Found {len([t for t in set(topics) if t != -1])} topics")

        return topics, probs

    def get_topic_name(self, topic_id: int) -> str:
        """Get human-readable name for a topic"""
        if not self._fitted or not self.topic_info:
            return f"Topic {topic_id}"

        # Find topic info
        for topic_info in self.topic_info:
            if topic_info["Topic"] == topic_id:
                # Use the Name field if available, otherwise create from words
                if (
                    "Name" in topic_info
                    and topic_info["Name"]
                    and topic_info["Name"] != "None"
                ):
                    return topic_info["Name"]
                elif "Words" in topic_info and topic_info["Words"]:
                    words = topic_info["Words"].split(",")[:3]
                    return " ".join(words).title()
                else:
                    return f"Topic {topic_id}"

        return f"Topic {topic_id}"

    def get_topic_words(
        self, topic_id: int, n_words: int = 10
    ) -> List[Tuple[str, float]]:
        """Get the top words for a specific topic"""
        if not self._fitted or not self._bertopic_model:
            return []

        try:
            topic_words = self._bertopic_model.get_topic(topic_id)
            # Convert numpy types to native Python types for JSON serialization
            if topic_words:
                return [(word, float(score)) for word, score in topic_words]
            return []
        except:
            return []

    def get_document_info(self, doc_index: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific document"""
        return self.document_topics.get(doc_index)

    def get_all_topics(self) -> Dict[int, Dict[str, Any]]:
        """Get information about all discovered topics"""
        if not self._fitted:
            return {}

        topics_info = {}
        for topic_info in self.topic_info:
            topic_id = topic_info["Topic"]
            if topic_id != -1:  # Skip outliers
                topics_info[topic_id] = {
                    "name": self.get_topic_name(topic_id),
                    "count": topic_info.get("Count", 0),
                    "words": self.get_topic_words(topic_id),
                    "representative_docs": self._get_representative_docs(topic_id),
                }

        return topics_info

    def _get_representative_docs(self, topic_id: int, n_docs: int = 3) -> List[str]:
        """Get representative documents for a topic"""
        docs = []
        for idx, doc_info in self.document_topics.items():
            if doc_info["topic_id"] == topic_id:
                docs.append(doc_info["document"])

        return docs[:n_docs]

    def save_results(self, output_dir: str = "bertopic_results"):
        """Save topic modeling results to files"""
        if not self._fitted:
            logger.error("Model not fitted yet")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Save topic information
        with open(f"{output_dir}/topics.json", "w") as f:
            json.dump(self.get_all_topics(), f, indent=2)

        # Save document assignments
        with open(f"{output_dir}/document_assignments.json", "w") as f:
            json.dump(self.document_topics, f, indent=2)

        # Save topic info DataFrame as CSV
        import pandas as pd

        df = pd.DataFrame(self.topic_info)
        df.to_csv(f"{output_dir}/topic_info.csv", index=False)

        if self.verbose:
            logger.info(f"Results saved to {output_dir}/")

    def print_summary(self):
        """Print a summary of the topic modeling results"""
        if not self._fitted:
            logger.error("Model not fitted yet")
            return

        print("\n" + "=" * 60)
        print("BERTOPIC DOCUMENT TOPIC MODELING RESULTS")
        print("=" * 60)

        all_topics = self.get_all_topics()
        print(f"\nTotal Topics Found: {len(all_topics)}")
        print(f"Total Documents: {len(self.document_topics)}")

        print("\nTOPICS SUMMARY:")
        print("-" * 40)

        for topic_id, topic_info in sorted(all_topics.items()):
            print(f"\nTopic {topic_id}: {topic_info['name']}")
            print(f"  Documents: {topic_info['count']}")
            print(
                f"  Top words: {', '.join([word for word, score in topic_info['words'][:5]])}"
            )

            # Show sample documents
            if topic_info["representative_docs"]:
                print("  Sample: " + topic_info["representative_docs"][0])

        print("\n" + "=" * 60)

    def create_visualizations(self, output_dir: str = "bertopic_results"):
        """Create visualizations of the topic modeling results"""
        if not self._fitted:
            logger.error("Model not fitted yet")
            return

        if not VISUALIZATION_AVAILABLE:
            print(
                "❌ Visualization libraries not available. Install matplotlib, seaborn, and wordcloud."
            )
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Set up the plotting style
            plt.style.use("default")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("BERTopic Analysis Results", fontsize=16, fontweight="bold")

            # 1. Topic Distribution (Top 15 topics)
            all_topics = self.get_all_topics()
            sorted_topics = sorted(
                all_topics.items(), key=lambda x: x[1]["count"], reverse=True
            )

            if len(sorted_topics) > 15:
                sorted_topics = sorted_topics[:15]

            topic_names = [
                (
                    topic[1]["name"][:50] + "..."
                    if len(topic[1]["name"]) > 50
                    else topic[1]["name"]
                )
                for topic in sorted_topics
            ]
            topic_counts = [topic[1]["count"] for topic in sorted_topics]

            axes[0, 0].bar(range(len(topic_names)), topic_counts)
            axes[0, 0].set_xticks(range(len(topic_names)))
            axes[0, 0].set_xticklabels(topic_names, rotation=45, ha="right")
            axes[0, 0].set_title("Topic Distribution (Top 15)")
            axes[0, 0].set_ylabel("Number of Documents")

            # 2. Word Cloud for most popular topic
            if sorted_topics:
                top_topic_id = sorted_topics[0][0]
                top_words = dict(self.get_topic_words(top_topic_id, n_words=20))

                if top_words:
                    wordcloud = WordCloud(
                        width=400,
                        height=400,
                        background_color="white",
                        colormap="viridis",
                    ).generate_from_frequencies(top_words)

                    axes[0, 1].imshow(wordcloud, interpolation="bilinear")
                    axes[0, 1].set_title(
                        f'Word Cloud - {sorted_topics[0][1]["name"][:30]}...'
                    )
                    axes[0, 1].axis("off")

            # 3. Topic Word Scores (Top 10 topics)
            axes[1, 0].axis("off")
            if len(sorted_topics) >= 10:
                sorted_topics_subset = sorted_topics[:10]
            else:
                sorted_topics_subset = sorted_topics

            cell_text = []
            for topic_id, topic_info in sorted_topics_subset:
                words = [word for word, score in topic_info["words"][:5]]
                cell_text.append(
                    [topic_info["name"][:40], topic_info["count"], ", ".join(words)]
                )

            table = axes[1, 0].table(
                cellText=cell_text,
                colLabels=["Topic", "Documents", "Top Words"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            axes[1, 0].set_title("Top Topics Summary")

            # 4. Document Length Distribution
            doc_lengths = [
                len(doc["document"].split()) for doc in self.document_topics.values()
            ]
            axes[1, 1].hist(doc_lengths, bins=30, alpha=0.7, edgecolor="black")
            axes[1, 1].set_title("Document Length Distribution")
            axes[1, 1].set_xlabel("Word Count")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].axvline(
                np.mean(doc_lengths),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(doc_lengths):.1f}",
            )
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/topic_visualizations.png", dpi=300, bbox_inches="tight"
            )
            print(f"💾 Visualizations saved to: {output_dir}/topic_visualizations.png")

            # Create individual word clouds for top 5 topics
            self._create_individual_wordclouds(output_dir, sorted_topics[:5])

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            print(f"❌ Error creating visualizations: {e}")

    def _create_individual_wordclouds(self, output_dir: str, top_topics: List):
        """Create individual word clouds for top topics"""
        try:
            for topic_id, topic_info in top_topics:
                words_dict = dict(topic_info["words"])

                if words_dict:
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color="white",
                        colormap="tab10",
                    ).generate_from_frequencies(words_dict)

                    plt.figure(figsize=(8, 6))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.title(f'Topic: {topic_info["name"][:50]}...')
                    plt.axis("off")
                    plt.savefig(
                        f"{output_dir}/wordcloud_topic_{topic_id}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

            print(
                f"💾 Individual word clouds saved to: {output_dir}/wordcloud_topic_*.png"
            )

        except Exception as e:
            logger.error(f"Error creating word clouds: {e}")

    def generate_cluster_summary(self, output_dir: str = "bertopic_results"):
        """Generate a formatted cluster summary"""
        if not self._fitted:
            logger.error("Model not fitted yet")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Load topic info for outlier count
        import pandas as pd

        topic_info_df = pd.DataFrame(self.topic_info) if self.topic_info else None

        print("🎯 BERTopic Cluster Summary")
        print("=" * 50)
        print(f"Total Topics: {len(self.get_all_topics())}")
        print()

        # Sort topics by document count
        all_topics = self.get_all_topics()
        sorted_topics = sorted(
            all_topics.items(), key=lambda x: x[1]["count"], reverse=True
        )

        print("📊 Top Topic Clusters:")
        print("-" * 30)
        print()

        for topic_id, topic_info in sorted_topics[:20]:  # Show top 20
            count = topic_info["count"]
            name = topic_info["name"]

            print(f"**{name}**")
            print(f"   📄 Documents: {count}")

            # Get top 5 words
            words = topic_info.get("words", [])
            if words:
                top_words = [word for word, score in words[:5]]
                print(f"   🔑 Top words: {', '.join(top_words)}")

            # Show a sample document
            sample_docs = topic_info.get("representative_docs", [])
            if sample_docs:
                sample = (
                    sample_docs[0][:100] + "..."
                    if len(sample_docs[0]) > 100
                    else sample_docs[0]
                )
                print(f"   💬 Sample: {sample}")

            print()

        # Show outlier information if available
        if topic_info_df is not None:
            outlier_row = topic_info_df[topic_info_df["Topic"] == -1]
            if not outlier_row.empty:
                outlier_count = outlier_row["Count"].iloc[0]
                print(f"📊 Outlier Documents: {outlier_count}")
                print("   (Documents that couldn't be assigned to any topic)")

        print("\n" + "=" * 50)
        print("✅ Cluster summary generated successfully!")

        # Save formatted summary to file
        self._save_cluster_summary_to_file(output_dir, sorted_topics, topic_info_df)

    def _save_cluster_summary_to_file(
        self, output_dir: str, sorted_topics: List, topic_info_df
    ):
        """Save formatted cluster summary to file"""
        output_file = os.path.join(output_dir, "cluster_summary.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("BERTopic Cluster Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Topics: {len(sorted_topics)}\n\n")

            f.write("Topic Clusters (sorted by document count):\n")
            f.write("-" * 50 + "\n\n")

            for topic_id, topic_info in sorted_topics:
                count = topic_info["count"]
                name = topic_info["name"]

                f.write(f"{name}\n")
                f.write(f"   Documents: {count}\n")

                # Get top 5 words
                words = topic_info.get("words", [])
                if words:
                    top_words = [word for word, score in words[:5]]
                    f.write(f"   Top words: {', '.join(top_words)}\n")

                # Show a sample document
                sample_docs = topic_info.get("representative_docs", [])
                if sample_docs:
                    sample = (
                        sample_docs[0][:80] + "..."
                        if len(sample_docs[0]) > 80
                        else sample_docs[0]
                    )
                    f.write(f"   Sample: {sample}\n")

                f.write("\n")

            # Add outlier information
            if topic_info_df is not None:
                outlier_row = topic_info_df[topic_info_df["Topic"] == -1]
                if not outlier_row.empty:
                    outlier_count = outlier_row["Count"].iloc[0]
                    f.write(f"Outlier Documents: {outlier_count}\n")
                    f.write("(Documents that couldn't be assigned to any topic)\n")

        print(f"💾 Formatted summary saved to: {output_file}")


def load_documents_from_file(file_path: str) -> List[str]:
    """
    Load documents from various file formats.

    Args:
        file_path: Path to the file containing documents

    Returns:
        List of document strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == ".txt":
            return load_txt_file(file_path)
        elif file_extension == ".csv":
            return load_csv_file(file_path)
        elif file_extension == ".json":
            return load_json_file(file_path)
        else:
            # Default to text file parsing
            return load_txt_file(file_path)

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


def load_txt_file(file_path: str) -> List[str]:
    """Load documents from a text file"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try different splitting strategies
    documents = []

    # First try splitting by double newlines (paragraphs)
    paragraphs = [para.strip() for para in content.split("\n\n") if para.strip()]
    if len(paragraphs) > 1:
        documents = paragraphs
    else:
        # Try splitting by single newlines (lines)
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if len(lines) > 1:
            documents = lines
        else:
            # Treat entire file as one document
            documents = [content.strip()]

    return documents


def load_csv_file(file_path: str) -> List[str]:
    """Load documents from a CSV file"""
    try:
        import pandas as pd

        df = pd.read_csv(file_path)

        # Look for common text columns
        text_columns = ["text", "content", "description", "body", "article", "document"]

        documents = []
        for col in text_columns:
            if col in df.columns:
                documents = df[col].dropna().astype(str).tolist()
                if documents:
                    break

        # If no standard column found, use the first text-like column
        if not documents:
            for col in df.columns:
                if df[col].dtype == "object":  # Text column
                    documents = df[col].dropna().astype(str).tolist()
                    if documents:
                        break

        if not documents:
            raise ValueError("No text columns found in CSV file")

        return documents

    except ImportError:
        raise ImportError(
            "pandas is required for CSV file loading. Install with: pip install pandas"
        )


def load_json_file(file_path: str) -> List[str]:
    """Load documents from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    # Handle different JSON structures
    if isinstance(data, list):
        # List of documents
        for item in data:
            if isinstance(item, str):
                documents.append(item)
            elif isinstance(item, dict):
                # Look for text fields in dictionaries
                text_fields = [
                    "text",
                    "content",
                    "description",
                    "body",
                    "article",
                    "document",
                ]
                for field in text_fields:
                    if field in item and isinstance(item[field], str):
                        documents.append(item[field])
                        break
                else:
                    # If no standard field, use the whole dict as string
                    documents.append(str(item))
    elif isinstance(data, dict):
        # Single document or dict with documents
        if "documents" in data and isinstance(data["documents"], list):
            documents = data["documents"]
        elif any(key in data for key in ["text", "content", "description", "body"]):
            # Single document in dict
            text_fields = [
                "text",
                "content",
                "description",
                "body",
                "article",
                "document",
            ]
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    documents = [data[field]]
                    break
            else:
                documents = [str(data)]
        else:
            documents = [str(data)]

    if not documents:
        raise ValueError("No documents found in JSON file")

    return documents


def is_text_file(file_path: str) -> bool:
    """
    Determine if a file is a text file that should be processed for topic modeling.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file should be processed, False if it's binary or should be excluded
    """
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Define binary file extensions to exclude
    binary_extensions = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".ear",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        ".deb",
        ".rpm",
        ".dmg",
        ".iso",
        ".img",
        ".log",
        ".tmp",
        ".temp",
        ".cache",
        ".lock",
    }

    # Exclude binary extensions
    if ext in binary_extensions:
        return False

    # Check MIME type
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            # Exclude binary MIME types
            if mime_type.startswith(
                ("image/", "audio/", "video/", "application/")
            ) and not mime_type.startswith(
                (
                    "text/",
                    "application/json",
                    "application/xml",
                    "application/javascript",
                )
            ):
                return False
    except:
        pass

    # Check if file can be decoded as text
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            # Check for null bytes (indicates binary file)
            if b"\x00" in chunk:
                return False
    except:
        return False

    return True


def load_documents_from_folder(
    folder_path: str, max_files: int = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Load documents from all text files in a folder and its subfolders.

    Args:
        folder_path: Path to the folder containing documents
        max_files: Maximum number of files to process (None for all files)

    Returns:
        Tuple of (documents_list, files_info_list)
        files_info_list contains metadata about each processed file
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    documents = []
    files_info = []
    processed_count = 0

    print(f"🔍 Scanning folder: {folder_path}")
    print("=" * 50)

    # Walk through all files in the folder recursively
    for root, dirs, files in os.walk(folder_path):
        # Skip common directories that shouldn't be processed
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in {"node_modules", "__pycache__", ".git", "dist", "build", "target"}
        ]

        for file in files:
            file_path = os.path.join(root, file)

            # Skip hidden files
            if file.startswith("."):
                continue

            # Check if it's a text file
            if not is_text_file(file_path):
                # print(f"⏭️  Skipping binary file: {file_path}")
                continue

            # Get file size
            try:
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)

                # Skip very large files (>50MB) to avoid memory issues
                if file_size_mb > 50:
                    print(f"⏭️  Skipping large file ({file_size_mb:.1f}MB): {file_path}")
                    continue

                # Load document from file
                try:
                    file_documents = load_documents_from_file(file_path)

                    # Add metadata for each document
                    for doc in file_documents:
                        if doc.strip():  # Only add non-empty documents
                            documents.append(doc.strip())
                            files_info.append(
                                {
                                    "file_path": file_path,
                                    "file_name": file,
                                    "file_size_bytes": file_size,
                                    "file_size_mb": round(file_size_mb, 2),
                                    "relative_path": os.path.relpath(
                                        file_path, folder_path
                                    ),
                                    "document_length": len(doc),
                                    "document_word_count": len(doc.split()),
                                }
                            )

                    processed_count += 1
                    # print(f"✅ Processed: {file} ({file_size_mb:.1f}MB, {len(file_documents)} docs)")

                    # Check if we've reached the max files limit
                    if max_files and processed_count >= max_files:
                        print(f"🛑 Reached maximum file limit: {max_files}")
                        break

                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")

            except Exception as e:
                print(f"❌ Error checking file {file_path}: {e}")

        # Break if we've reached the max files limit
        if max_files and processed_count >= max_files:
            break

    # Print summary
    print("\n📊 FOLDER PROCESSING SUMMARY")
    print("=" * 50)
    print(f"📁 Folder: {folder_path}")
    print(f"📄 Total documents loaded: {len(documents)}")
    print(f"📁 Files processed: {processed_count}")
    print(f"📏 Total size: {sum(f['file_size_mb'] for f in files_info):.1f}MB")

    if files_info:
        # Show largest files
        largest_files = sorted(
            files_info, key=lambda x: x["file_size_mb"], reverse=True
        )[:5]
        print("\n🔝 Largest files processed:")
        for file_info in largest_files:
            print(f"   • {file_info['file_name']} ({file_info['file_size_mb']:.1f}MB)")

    return documents, files_info


def load_sample_documents(file_path: str = None) -> List[str]:
    """Load sample documents for testing"""
    if file_path and os.path.exists(file_path):
        return load_documents_from_file(file_path)

    # Return sample documents if no file provided
    return [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions through trial and error and rewards.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Unsupervised learning finds hidden patterns in data without explicit labels.",
        "Transfer learning leverages pre-trained models to solve new but related problems.",
        "The stock market is influenced by various economic factors and investor sentiment.",
        "Technical analysis uses historical price patterns to predict future market movements.",
        "Fundamental analysis evaluates company financials to determine stock value.",
        "Portfolio diversification helps reduce investment risk across different assets.",
        "Cryptocurrency markets operate 24/7 and are highly volatile compared to traditional markets.",
        "Climate change is affecting weather patterns and sea levels globally.",
        "Renewable energy sources like solar and wind are becoming more cost-effective.",
        "Sustainable development aims to meet present needs without compromising future generations.",
        "Electric vehicles are gaining popularity as battery technology improves.",
        "The healthcare system faces challenges with rising costs and aging populations.",
        "Telemedicine allows remote patient care through digital communication technologies.",
        "Medical research continues to advance treatments for various diseases and conditions.",
    ]


def main():
    """Main function with command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="BERTopic Document Topic Modeling Prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bertopic_clustering.py --file documents.txt
  python bertopic_clustering.py --file data.csv --output results --max-topics 15
  python bertopic_clustering.py --file articles.json --min-topic-size 5 --verbose
  python bertopic_clustering.py --file docs.txt --max-topics 0  # Automatic topic detection
  python bertopic_clustering.py --interactive  # Use sample data
  python bertopic_clustering.py --file docs.txt --visualize --summary  # With visualizations and summary
  python bertopic_clustering.py --file docs.txt --output results --visualize --summary

Folder processing examples:
  python bertopic_clustering.py --folder /path/to/documents --output results
  python bertopic_clustering.py --folder ./docs --max-files 100 --visualize --summary
  python bertopic_clustering.py --folder /path/to/documents --max-files 50 --min-topic-size 5
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to file containing documents (.txt, .csv, .json)",
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing documents (will process all text files recursively)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process from folder (default: None for all files)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="bertopic_results",
        help="Output directory for results (default: bertopic_results)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=3,
        help="Minimum number of documents per topic (default: 3)",
    )

    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Maximum number of topics to extract (default: None for automatic detection)",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode for exploring results",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print progress information (default: True)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress information (same as --verbose False)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations of the topic modeling results",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate formatted cluster summary",
    )

    args = parser.parse_args()

    # Handle quiet mode
    verbose = args.verbose and not args.quiet

    print("BERTopic Document Topic Modeling Prototype")
    print("=" * 50)

    try:
        # Load documents
        if args.folder:
            print(f"Loading documents from folder: {args.folder}")
            documents, files_info = load_documents_from_folder(
                args.folder, args.max_files
            )

            # Save files info to output directory for reference
            if files_info:
                os.makedirs(args.output, exist_ok=True)
                files_info_file = os.path.join(args.output, "processed_files.json")
                with open(files_info_file, "w", encoding="utf-8") as f:
                    json.dump(files_info, f, indent=2, ensure_ascii=False)
                print(f"💾 Files information saved to: {files_info_file}")

        elif args.file:
            print(f"Loading documents from: {args.file}")
            documents = load_documents_from_file(args.file)
        else:
            print("Loading sample documents...")
            documents = load_sample_documents()

        print(f"Loaded {len(documents)} documents")

        if len(documents) == 0:
            print("Error: No documents found!")
            return 1

        # Initialize topic modeler
        print("Initializing BERTopic modeler...")
        modeler = DocumentTopicModeler(
            model_name=args.model,
            min_topic_size=args.min_topic_size,
            max_topics=args.max_topics,
            verbose=verbose,
        )

        # Fit model
        print("Fitting topic model...")
        topics, probs = modeler.fit_transform(documents)

        # Print results
        modeler.print_summary()

        # Save results
        print(f"\nSaving results to: {args.output}")
        modeler.save_results(args.output)

        # Generate visualizations if requested
        if args.visualize:
            print(f"\nCreating visualizations...")
            modeler.create_visualizations(args.output)

        # Generate cluster summary if requested
        if args.summary:
            print(f"\nGenerating cluster summary...")
            modeler.generate_cluster_summary(args.output)

        # Interactive exploration
        if args.interactive or not args.file:
            print("\nINTERACTIVE EXPLORATION")
            print("-" * 30)
            print(
                "Enter document index (0-{}) to see its topic assignment, or 'q' to quit:".format(
                    len(documents) - 1
                )
            )

            while True:
                try:
                    user_input = input("\nDocument index: ").strip()
                    if user_input.lower() == "q":
                        break

                    doc_idx = int(user_input)
                    if 0 <= doc_idx < len(documents):
                        doc_info = modeler.get_document_info(doc_idx)
                        if doc_info:
                            print(f"\nDocument: {doc_info['document']}")
                            print(
                                f"Topic: {doc_info['topic_name']} (ID: {doc_info['topic_id']})"
                            )
                            print(f"Confidence: {doc_info['probability']:.3f}")

                            # Show topic words
                            topic_words = modeler.get_topic_words(doc_info["topic_id"])
                            if topic_words:
                                print(
                                    f"Topic words: {', '.join([word for word, score in topic_words[:5]])}"
                                )
                        else:
                            print("No information available for this document")
                    else:
                        print(
                            f"Please enter a valid index between 0 and {len(documents)-1}"
                        )

                except ValueError:
                    print("Please enter a valid number or 'q' to quit")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break

        print(f"\n✅ Topic modeling completed! Results saved to: {args.output}")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def demonstrate_knowledge_graph_enhancement():
    """Demonstrate the enhanced knowledge graph creation from BERTopic clusters"""

    print("🎯 BERTopic Knowledge Graph Enhancement Demo")
    print("=" * 60)

    # Load sample documents
    documents = load_sample_documents()
    print(f"📄 Loaded {len(documents)} sample documents")

    # Run BERTopic clustering
    print("\n🔄 Running BERTopic clustering...")
    modeler = DocumentTopicModeler(
        model_name="all-MiniLM-L6-v2", min_topic_size=2, verbose=True
    )

    topics, probs = modeler.fit_transform(documents)
    all_topics = modeler.get_all_topics()

    print(f"✅ Found {len(all_topics)} topics")

    # Build knowledge graph from topics
    print("\n🔗 Building knowledge graph from topic clusters...")

    kg_builder = TopicKnowledgeGraphBuilder(context_window_size=2)
    knowledge_graph = kg_builder.build_from_bertopic(modeler, documents)

    # Create navigator for reader-friendly exploration
    navigator = KnowledgeGraphNavigator(knowledge_graph)

    # Demonstrate reader-friendly output
    print("\n📊 Knowledge Graph Summary:")
    print("-" * 40)
    metadata = knowledge_graph["metadata"]
    print(f"📈 Topics: {metadata['total_topics']}")
    print(f"🧠 Concepts: {metadata['total_concepts']}")
    print(f"📄 Documents: {metadata['total_documents']}")
    print(f"🔍 Context windows: {metadata['context_windows_extracted']}")

    # Show topic overview
    print("\n🏗️  Topic Hierarchy:")
    print("-" * 40)

    overview = navigator.get_topic_overview()
    for i, topic in enumerate(overview["topics"][:5], 1):
        print(f"{i}. {topic['name']}")
        print(f"   Documents: {topic['metadata']['document_count']}")
        print(f"   Top words: {', '.join(topic['metadata']['top_words'][:3])}")
        print()

    # Demonstrate topic exploration
    if overview["topics"]:
        first_topic_id = overview["topics"][0]["id"]
        print(f"\n🔍 Exploring topic: {overview['topics'][0]['name']}")
        print("-" * 40)

        exploration = navigator.explore_topic(first_topic_id)

        if "error" not in exploration:
            topic = exploration["topic"]
            print(f"📋 Description: {topic['description'][:100]}...")
            print(f"🧠 Related concepts: {len(exploration['concepts'])}")
            print(f"📄 Related documents: {len(exploration['documents'])}")

            # Show sample concepts
            for concept in exploration["concepts"][:3]:
                print(f"   • {concept['name']}")

    # Demonstrate search
    print("\n🔍 Search Demo:")
    print("-" * 40)

    search_results = navigator.search_graph("learning", limit=3)
    for result in search_results:
        node = result["node"]
        print(f"• {node['name']} ({node['type']}) - {result['match_type']}")

    # Demonstrate incremental indexing
    print("\n🔄 Incremental Index Demo:")
    print("-" * 40)

    index_manager = IncrementalIndexManager()

    # Simulate file changes
    sample_files = ["doc1.txt", "doc2.txt", "doc3.txt"]
    changes = index_manager.detect_file_changes(sample_files)

    print(f"📊 Would reindex: {index_manager.should_reindex(sample_files)}")
    print(f"📈 Total changes detected: {changes['total_new_modified']}")

    # Save results
    output_dir = "bertopic_knowledge_graph_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Save knowledge graph
    kg_file = os.path.join(output_dir, "knowledge_graph.json")
    with open(kg_file, "w") as f:
        json.dump(knowledge_graph, f, indent=2, default=str)

    # Save visualization data
    viz_file = os.path.join(output_dir, "visualization_data.json")
    with open(viz_file, "w") as f:
        json.dump(navigator.export_for_visualization(), f, indent=2, default=str)

    # Save topic overview
    overview_file = os.path.join(output_dir, "topic_overview.json")
    with open(overview_file, "w") as f:
        json.dump(navigator.get_topic_overview(), f, indent=2, default=str)

    print(f"\n💾 Knowledge graph saved to: {output_dir}/")
    print("   • knowledge_graph.json - Full graph data")
    print("   • visualization_data.json - D3.js compatible format")
    print("   • topic_overview.json - Reader-friendly summary")

    return knowledge_graph


# Enhanced main function with knowledge graph demo
def main():
    """Enhanced main function with knowledge graph demonstration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced BERTopic with Knowledge Graph Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced BERTopic Prototype with Knowledge Graph Generation
Features: Context window extraction, hierarchical topic relationships, incremental indexing

📂 FOLDER PROCESSING with MODEL SELECTION:
  python bertopic_clustering.py --folder ./documents --model all-MiniLM-L6-v2 --kg
  python bertopic_clustering.py --folder ./docs --model bge-base-en-v1.5 --kg --incremental

🔗 KNOWLEDGE GRAPH GENERATION:
  python bertopic_clustering.py --folder ./docs --kg --visualize --summary
  python bertopic_clustering.py --file documents.txt --kg --incremental

⚡ INCREMENTAL INDEXING:
  python bertopic_clustering.py --folder ./docs --incremental --force-index
  python bertopic_clustering.py --folder ./docs --incremental --skip-if-unchanged

🌐 VISUALIZATION:
  Open bertopic_knowledge_graph_demo/d3_visualization.html in your browser

🧪 DEMO & TESTING:
  python bertopic_clustering.py --demo  # Run with sample data
  python bertopic_clustering.py --folder ./test_docs --kg --visualize --summary
        """,
    )

    # Input options
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to file containing documents (.txt, .csv, .json)",
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing documents (will process all text files recursively)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process from folder (default: None for all files)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "sentence-transformers/bert-base-nli-mean-tokens",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "BAAI/bge-base-en-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
            "Qwen/Qwen3-Embedding-0.6B",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ],
        help="Sentence transformer model name for embeddings",
    )

    # Topic modeling parameters
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=15,
        help="Minimum number of documents per topic (default: 15 - increased for topic reduction)",
    )

    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Maximum number of topics to extract (default: None for automatic detection)",
    )

    # Clustering and dimensionality reduction parameters (for topic reduction)
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.4,
        help="HDBSCAN cluster merging threshold - higher = coarser clustering (default: 0.4)",
    )

    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=8,
        help="UMAP neighborhood size - lower = coarser embeddings (default: 8)",
    )

    parser.add_argument(
        "--umap-components",
        type=int,
        default=3,
        help="UMAP output dimensions - lower = more aggressive reduction (default: 3)",
    )

    # Knowledge graph options
    parser.add_argument(
        "--kg",
        action="store_true",
        help="Generate knowledge graph from topic clusters with context window extraction",
    )

    parser.add_argument(
        "--context-window-size",
        type=int,
        default=3,
        help="Number of sentences for context windows in knowledge graph (default: 3)",
    )

    # Incremental indexing
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Enable incremental indexing - only reprocess changed/added files",
    )

    parser.add_argument(
        "--force-index",
        action="store_true",
        help="Force full re-indexing regardless of file changes (requires --incremental)",
    )

    parser.add_argument(
        "--index-file",
        type=str,
        default="results/topic_index.json",
        help="Path to index metadata file for incremental processing",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="prototype/bertopic/bertopic_results",
        help="Output directory for results relative to prototype/bertopic",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations of the topic modeling results",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate formatted cluster summary",
    )

    parser.add_argument(
        "--save-d3-data",
        action="store_true",
        help="Save D3.js compatible data for web visualization",
    )

    # Demo and testing
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run knowledge graph demonstration with sample data",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print progress information (default: True)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress information",
    )

    args = parser.parse_args()

    print("🚀 Enhanced BERTopic with Knowledge Graph Generation")
    print("=" * 60)
    print(f"📊 Model: {args.model}")
    if args.folder:
        print(f"📁 Folder: {args.folder}")
    elif args.file:
        print(f"📄 File: {args.file}")
    else:
        print("📝 Using: Sample documents")

    # Handle verbosity
    verbose = args.verbose and not args.quiet

    try:
        # Run demonstration if requested
        if args.demo:
            demonstrate_knowledge_graph_enhancement()
            return 0

        # Load documents based on input type
        documents = []
        files_processed = []
        incremental_manager = None

        if args.folder:
            print(f"\n🔍 Processing folder: {args.folder}")

            # Get all file paths for incremental indexing
            import glob

            file_patterns = ["**/*.txt", "**/*.csv", "**/*.json", "**/*.md"]
            all_files = []
            for pattern in file_patterns:
                all_files.extend(
                    glob.glob(os.path.join(args.folder, pattern), recursive=True)
                )
            all_files = [f for f in all_files if is_text_file(f)]

            # Handle incremental indexing
            if args.incremental:
                incremental_manager = IncrementalIndexManager(args.index_file)
                should_reindex = incremental_manager.should_reindex(
                    all_files, args.force_index
                )

                if not should_reindex:
                    print("✅ No changes detected. Skipping re-indexing.")
                    print(f"💡 To force re-indexing, use --force-index")
                    return 0

                print("🔄 Changes detected. Proceeding with indexing...")

            documents, files_processed = load_documents_from_folder(
                args.folder, args.max_files
            )

            if incremental_manager and files_processed:
                # Update index metadata
                file_paths = [f["file_path"] for f in files_processed]
                incremental_manager.update_index_metadata(file_paths, args.model)

        elif args.file:
            print(f"\n📄 Loading file: {args.file}")
            documents = load_documents_from_file(args.file)
        else:
            print("\n📝 Loading sample documents...")
            documents = load_sample_documents()

        if len(documents) == 0:
            print("❌ Error: No documents found!")
            return 1

        print(f"✅ Loaded {len(documents)} documents")

        # Create output directory based on model name and folder/file analyzed
        # Normalize model name (remove special characters and convert to lowercase)
        model_short = (
            args.model.replace("sentence-transformers/", "")
            .replace("intfloat/", "")
            .replace("BAAI/", "")
            .replace("Qwen/", "")
            .replace("-", "")
            .replace("_", "")
            .lower()
        )

        # Get folder or file base name for directory naming
        if args.folder:
            folder_name = os.path.basename(os.path.normpath(args.folder))
            if folder_name == "." or folder_name == "":
                folder_name = "current"
        elif args.file:
            folder_name = os.path.splitext(os.path.basename(args.file))[0]
        else:
            folder_name = "sample"

        # Create result directory name under ./results/
        results_base = "results"
        output_dir = os.path.join(results_base, f"{model_short}_{folder_name}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {output_dir}/")

        # Initialize and fit topic modeler
        print(f"\n🤖 Initializing BERTopic with model: {args.model}")
        print(
            f"   Settings: min_topic_size={args.min_topic_size}, cluster_epsilon={args.cluster_epsilon}, umap_neighbors={args.umap_neighbors}, umap_components={args.umap_components}"
        )
        modeler = DocumentTopicModeler(
            model_name=args.model,
            min_topic_size=args.min_topic_size,
            max_topics=args.max_topics,
            cluster_epsilon=args.cluster_epsilon,
            umap_neighbors=args.umap_neighbors,
            umap_components=args.umap_components,
            verbose=verbose,
        )

        print("🔄 Fitting topic model...")
        topics, probs = modeler.fit_transform(documents)

        all_topics = modeler.get_all_topics()
        print(f"✅ Found {len(all_topics)} topics")

        # Print topic summary
        modeler.print_summary()

        # Generate knowledge graph if requested
        knowledge_graph = None
        navigator = None

        if args.kg:
            print("\n🔗 Generating Knowledge Graph...")
            kg_builder = TopicKnowledgeGraphBuilder(
                context_window_size=args.context_window_size
            )
            knowledge_graph = kg_builder.build_from_bertopic(modeler, documents)

            # Create navigator for exploration
            navigator = KnowledgeGraphNavigator(knowledge_graph)

            # Save knowledge graph data
            kg_file = os.path.join(output_dir, "knowledge_graph.json")
            with open(kg_file, "w") as f:
                json.dump(knowledge_graph, f, indent=2, default=str)

            print(f"💾 Knowledge graph saved: {kg_file}")

            # Save D3.js compatible data if requested
            if args.save_d3_data or args.kg:
                viz_file = os.path.join(output_dir, "visualization_data.json")
                viz_data = navigator.export_for_visualization()
                with open(viz_file, "w") as f:
                    json.dump(viz_data, f, indent=2, default=str)

                # Copy D3 visualization HTML to output directory
                import shutil

                html_src = os.path.join(
                    "prototype", "bertopic", "d3_visualization.html"
                )
                html_dest = os.path.join(output_dir, "d3_visualization.html")
                if os.path.exists(html_src):
                    shutil.copy2(html_src, html_dest)

                print("🌐 D3 visualization data saved")
                print(
                    f"🎯 Open {html_dest} in your browser to view the knowledge graph"
                )

        # Save standard results
        print(f"\n💾 Saving results to: {output_dir}")
        modeler.save_results(output_dir)

        # Create visualizations
        if args.visualize:
            print("📊 Creating visualizations...")
            modeler.create_visualizations(output_dir)

        # Generate cluster summary
        if args.summary:
            print("📝 Generating cluster summary...")
            modeler.generate_cluster_summary(output_dir)

        # Print knowledge graph summary if generated
        if knowledge_graph:
            metadata = knowledge_graph["metadata"]
            print("\n📊 KNOWLEDGE GRAPH SUMMARY")
            print("=" * 40)
            print(f"📈 Topics: {metadata['total_topics']}")
            print(f"🧠 Concepts: {metadata['total_concepts']}")
            print(f"📄 Documents: {metadata['total_documents']}")
            print(f"🔍 Context windows: {metadata['context_windows_extracted']}")

        # Handle file data (for folder processing)
        if files_processed and args.folder:
            files_info_file = os.path.join(output_dir, "processed_files.json")
            with open(files_info_file, "w", encoding="utf-8") as f:
                json.dump(files_processed, f, indent=2, ensure_ascii=False)
            print(f"📁 Files information saved: {files_info_file}")

        print("\n✅ Processing completed successfully!")
        print(f"📂 Results available in: {output_dir}")
        if knowledge_graph:
            print("\n🔗 KNOWLEDGE GRAPH FEATURES:")
            print("   • Interactive D3.js visualization")
            print("   • Context window extraction")
            print("   • Hierarchical topic relationships")
            print("   • Reader-friendly navigation")
            if args.incremental:
                print("   • Incremental index management")

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
