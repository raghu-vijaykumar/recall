"""
Core BERTopic Processing Module

Provides the main processor classes for topic modeling with proper logging,
OOP design, and error handling.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    import umap
    import hdbscan

    # Try to import visualization libraries
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from wordcloud import WordCloud

        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        plt = None
        sns = None
        WordCloud = None

    BERTOPIC_AVAILABLE = True
except ImportError as e:
    BERTopic = None
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None
    WordCloud = None
    BERTOPIC_AVAILABLE = False

# Configure package logger
logger = logging.getLogger(__name__)


@dataclass
class TopicModelingConfig:
    """
    Configuration for topic modeling parameters.

    Attributes:
        model_name: Name of the sentence transformer model to use
        min_topic_size: Minimum documents per topic
        max_topics: Maximum number of topics (None for auto)
        cluster_epsilon: HDBSCAN cluster merging threshold
        umap_neighbors: UMAP neighborhood size
        umap_components: UMAP output dimensions
        diversity: Diversity threshold for topic representation
        verbose: Enable verbose logging during processing
    """

    model_name: str = "all-MiniLM-L6-v2"
    min_topic_size: int = 15
    max_topics: Optional[int] = None
    cluster_epsilon: float = 0.4
    umap_neighbors: int = 8
    umap_components: int = 3
    diversity: Optional[float] = None
    verbose: bool = False


class ModelBuilder:
    """
    Handles creation and configuration of BERTopic models.

    This class encapsulates all model building logic, ensuring
    dynamic parameter adjustment based on dataset size and
    proper error handling.
    """

    def __init__(self, config: TopicModelingConfig):
        """
        Initialize the model builder.

        Args:
            config: Configuration object for topic modeling

        Raises:
            ImportError: If BERTopic dependencies are not available
        """
        self.config = config
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "BERTopic and required dependencies not installed. "
                "Install with: pip install bertopic scikit-learn"
            )

    def build_model(self, documents: List[str]) -> BERTopic:
        """
        Build and configure BERTopic model with optimized parameters.

        Dynamically adjusts parameters based on dataset size for optimal
        performance and topic quality.

        Args:
            documents: List of document texts

        Returns:
            Configured BERTopic model instance

        Raises:
            ValueError: If no documents provided or BERTopic unavailable
        """
        if not documents:
            raise ValueError("Cannot build model: no documents provided")

        n_docs = len(documents)

        # Dynamic parameter adjustment for better performance
        effective_min_topic_size = min(self.config.min_topic_size, max(2, n_docs // 10))
        n_neighbors = min(self.config.umap_neighbors, max(2, n_docs - 1))
        n_components = min(self.config.umap_components, max(2, n_docs - 1))

        if self.config.verbose:
            logger.info(f"Building model for {n_docs} documents")
            logger.info(f"Effective min_topic_size: {effective_min_topic_size}")

        # Choose appropriate dimensionality reduction based on dataset size
        if n_docs < 10:
            # Use PCA for very small datasets
            umap_model = PCA(n_components=max(2, n_docs - 1), random_state=42)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_topic_size,
                metric="euclidean",
                cluster_selection_epsilon=self.config.cluster_epsilon,
                prediction_data=True,
            )
        else:
            # Use UMAP for larger datasets
            umap_model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_topic_size,
                metric="euclidean",
                cluster_selection_epsilon=self.config.cluster_epsilon,
                prediction_data=True,
            )

        # Configure vectorizer based on data size
        min_df = max(1, n_docs // 1000)
        max_df = 0.95

        vectorizer_model = CountVectorizer(
            min_df=min_df, max_df=max_df, stop_words="english", ngram_range=(1, 2)
        )

        representation_model = KeyBERTInspired()

        # Create the BERTopic model
        model = BERTopic(
            embedding_model=self.config.model_name,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            nr_topics=self.config.max_topics,
            min_topic_size=effective_min_topic_size,
            calculate_probabilities=True,
            verbose=self.config.verbose,
        )

        logger.info("BERTopic model configured successfully")
        return model


class TopicAnalyzer:
    """
    Handles topic analysis and extraction operations.

    Separates analysis logic from model building, providing
    methods for topic interpretation and document assignment.
    """

    def __init__(self, bertopic_model: BERTopic, config: TopicModelingConfig):
        """
        Initialize the topic analyzer.

        Args:
            bertopic_model: Fitted BERTopic model
            config: Configuration object
        """
        self.model = bertopic_model
        self.config = config
        self._fitted = False

        # Results storage
        self.topics: List[int] = []
        self.topic_info: List[Dict[str, Any]] = []
        self.document_topics: List[Dict[str, Any]] = []
        self.probabilities: Optional[np.ndarray] = None

    def fit_transform(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fit the model and transform documents to topic assignments.

        Args:
            documents: List of document texts

        Returns:
            Tuple of (topic_assignments, topic_probabilities)

        Raises:
            ValueError: If no documents provided
        """
        if not documents:
            raise ValueError("No documents provided for topic analysis")

        if self.config.verbose:
            logger.info(f"Fitting topic model on {len(documents)} documents...")

        self.topics, self.probabilities = self.model.fit_transform(documents)

        # Store results
        self.topic_info = self.model.get_topic_info().to_dict("records")
        self._fitted = True

        # Process document-topic assignments
        self._process_document_assignments(documents, self.topics, self.probabilities)

        if self.config.verbose:
            n_topics = len([t for t in set(self.topics) if t != -1])
            logger.info(f"Successfully extracted {n_topics} topics")

        return self.topics, self.probabilities

    def _process_document_assignments(
        self, documents: List[str], topics: List[int], probabilities: np.ndarray
    ):
        """Process and store document-topic assignment information."""
        self.document_topics = []

        for i, (doc, topic_id) in enumerate(zip(documents, topics)):
            # Handle probability extraction safely
            probability = self._extract_probability(probabilities, i, topic_id)

            doc_info = {
                "document_index": i,
                "document_preview": (doc[:100] + "..." if len(doc) > 100 else doc),
                "topic_id": int(topic_id),
                "topic_name": self.get_topic_name(topic_id),
                "probability": probability,
            }
            self.document_topics.append(doc_info)

    def _extract_probability(
        self, probabilities: Optional[np.ndarray], doc_idx: int, topic_id: int
    ) -> float:
        """
        Safely extract topic probability for a document.

        Args:
            probabilities: Probability matrix from BERTopic
            doc_idx: Document index
            topic_id: Topic ID

        Returns:
            Topic assignment probability (default 0.5 if unavailable)
        """
        if probabilities is None:
            return 0.5

        try:
            if (
                hasattr(probabilities, "__getitem__")
                and len(probabilities) > doc_idx
                and len(probabilities[doc_idx]) > topic_id >= 0
            ):
                return float(probabilities[doc_idx][topic_id])
        except (IndexError, TypeError, AttributeError):
            pass

        return 0.5

    def get_topic_name(self, topic_id: int) -> str:
        """
        Get human-readable name for a topic.

        Args:
            topic_id: Numeric topic identifier

        Returns:
            Topic name string
        """
        if not self._fitted or not self.topic_info:
            return f"Topic {topic_id}"

        for topic_info in self.topic_info:
            if topic_info.get("Topic") == topic_id:
                # Use Name field if available
                name = topic_info.get("Name", "").strip()
                if name and name.lower() not in ["none", ""]:
                    return name

                # Fall back to first few words
                words = topic_info.get("Words", "").split(",")[:3]
                if words:
                    return " ".join(words).title()

        return f"Topic {topic_id}"

    def get_topic_words(
        self, topic_id: int, n_words: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top words for a topic.

        Args:
            topic_id: Topic identifier
            n_words: Number of words to return

        Returns:
            List of (word, score) tuples
        """
        if not self._fitted or not self.model:
            return []

        try:
            topic_words = self.model.get_topic(topic_id)
            if topic_words:
                return [(word, float(score)) for word, score in topic_words[:n_words]]
        except Exception as e:
            logger.warning(f"Failed to get topic words for topic {topic_id}: {e}")

        return []

    def get_all_topics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all discovered topics.

        Returns:
            Dictionary mapping topic_id to topic information
        """
        if not self._fitted:
            return {}

        topics_info = {}
        for topic_info in self.topic_info:
            topic_id = topic_info.get("Topic")
            if topic_id is not None and topic_id != -1:  # Skip outliers
                topics_info[topic_id] = {
                    "name": self.get_topic_name(topic_id),
                    "count": topic_info.get("Count", 0),
                    "words": self.get_topic_words(topic_id),
                    "representative_docs": self._get_representative_docs(topic_id),
                }

        return topics_info

    def _get_representative_docs(self, topic_id: int, n_docs: int = 3) -> List[str]:
        """
        Get representative documents for a topic.

        Args:
            topic_id: Topic identifier
            n_docs: Number of documents to return

        Returns:
            List of document previews
        """
        docs = [
            doc_info["document_preview"]
            for doc_info in self.document_topics
            if doc_info["topic_id"] == topic_id
        ]
        return docs[:n_docs]

    def get_document_info(self, doc_index: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific document.

        Args:
            doc_index: Document index

        Returns:
            Document information dictionary or None
        """
        if doc_index < len(self.document_topics):
            return self.document_topics[doc_index]
        return None

    def get_topic_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the topic modeling results.

        Returns:
            Dictionary with various statistics
        """
        if not self._fitted:
            return {}

        total_docs = len(self.document_topics)
        outlier_count = sum(1 for doc in self.document_topics if doc["topic_id"] == -1)
        topic_count = len(self.get_all_topics())

        # Calculate topic distribution
        topic_sizes = {}
        for doc in self.document_topics:
            topic_id = doc["topic_id"]
            topic_sizes[topic_id] = topic_sizes.get(topic_id, 0) + 1

        return {
            "total_topics": topic_count,
            "total_documents": total_docs,
            "outlier_documents": outlier_count,
            "documents_per_topic": topic_sizes,
            "average_docs_per_topic": (total_docs / max(topic_count, 1)),
        }


class ResultExporter:
    """
    Handles exporting and saving of topic modeling results.

    Separates persistence logic from analysis logic, supporting
    multiple output formats.
    """

    def __init__(self, analyzer: TopicAnalyzer, config: TopicModelingConfig):
        """
        Initialize the result exporter.

        Args:
            analyzer: Fitted TopicAnalyzer instance
            config: Configuration object
        """
        self.analyzer = analyzer
        self.config = config

    def save_topics_json(self, output_dir: Path) -> None:
        """
        Save topic information as JSON.

        Args:
            output_dir: Directory to save results
        """
        if not self.analyzer._fitted:
            logger.warning("Analyzer not fitted, skipping topic JSON export")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        topics_file = output_dir / "topics.json"

        topics_data = self.analyzer.get_all_topics()

        with open(topics_file, "w", encoding="utf-8") as f:
            import json

            json.dump(topics_data, f, indent=2)

        if self.config.verbose:
            logger.info(f"Topics saved to: {topics_file}")

    def save_document_assignments(self, output_dir: Path) -> None:
        """
        Save document-topic assignments as JSON.

        Args:
            output_dir: Directory to save results
        """
        if not self.analyzer._fitted:
            logger.warning("Analyzer not fitted, skipping document assignments export")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        assignments_file = output_dir / "document_assignments.json"

        with open(assignments_file, "w", encoding="utf-8") as f:
            import json

            json.dump(self.analyzer.document_topics, f, indent=2)

    def save_topic_info_csv(self, output_dir: Path) -> None:
        """
        Save topic info as CSV.

        Args:
            output_dir: Directory to save results
        """
        if not self.analyzer._fitted:
            logger.warning("Analyzer not fitted, skipping CSV export")
            return

        try:
            import pandas as pd

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_file = output_dir / "topic_info.csv"

            df = pd.DataFrame(self.analyzer.topic_info)
            df.to_csv(csv_file, index=False)

            if self.config.verbose:
                logger.info(f"Topic info CSV saved to: {csv_file}")

        except ImportError:
            logger.warning("pandas not available for CSV export")

    def save_all_results(self, output_dir: Path) -> None:
        """
        Save all standard results.

        Args:
            output_dir: Directory to save results
        """
        if self.config.verbose:
            logger.info(f"Saving all results to: {output_dir}")

        self.save_topics_json(output_dir)
        self.save_document_assignments(output_dir)
        self.save_topic_info_csv(output_dir)

        if self.config.verbose:
            logger.info("All results saved successfully")


class VisualizationManager:
    """
    Handles creation and management of visualizations.

    Separates visualization logic from analysis logic, supporting
    various chart types and export formats.
    """

    def __init__(self, analyzer: TopicAnalyzer, config: TopicModelingConfig):
        """
        Initialize the visualization manager.

        Args:
            analyzer: Fitted TopicAnalyzer instance
            config: Configuration object
        """
        self.analyzer = analyzer
        self.config = config

    def _check_visualization_available(self) -> bool:
        """
        Check if visualization libraries are available.

        Returns:
            True if libraries available, False otherwise
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning(
                "Visualization libraries not available. "
                "Install: pip install matplotlib seaborn wordcloud"
            )
            return False
        return True

    def create_topic_distribution_plot(self, output_dir: Path) -> None:
        """
        Create topic distribution bar chart.

        Args:
            output_dir: Directory to save visualization
        """
        if not self._check_visualization_available() or not self.analyzer._fitted:
            return

        all_topics = self.analyzer.get_all_topics()
        if not all_topics:
            return

        # Get top topics by document count
        sorted_topics = sorted(
            all_topics.items(), key=lambda x: x[1]["count"], reverse=True
        )[
            :15
        ]  # Top 15 topics

        if not sorted_topics:
            return

        topic_names = [
            (
                topic_data["name"][:50] + "..."
                if len(topic_data["name"]) > 50
                else topic_data["name"]
            )
            for _, topic_data in sorted_topics
        ]
        topic_counts = [topic_data["count"] for _, topic_data in sorted_topics]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(topic_names)), topic_counts, alpha=0.8)
        plt.xticks(range(len(topic_names)), topic_names, rotation=45, ha="right")
        plt.title("Topic Distribution (Top 15)")
        plt.ylabel("Number of Documents")
        plt.xlabel("Topic")
        plt.tight_layout()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "topic_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        if self.config.verbose:
            logger.info(f"Topic distribution plot saved to: {plot_file}")

    def create_word_clouds(self, output_dir: Path, max_topics: int = 5) -> None:
        """
        Create word clouds for top topics.

        Args:
            output_dir: Directory to save visualizations
            max_topics: Maximum number of word clouds to create
        """
        if not self._check_visualization_available() or not self.analyzer._fitted:
            return

        all_topics = self.analyzer.get_all_topics()
        if not all_topics:
            return

        # Get top topics by document count
        sorted_topics = sorted(
            all_topics.items(), key=lambda x: x[1]["count"], reverse=True
        )[:max_topics]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for topic_id, topic_data in sorted_topics:
            words_dict = dict(topic_data["words"])

            if not words_dict:
                continue

            plt.figure(figsize=(8, 6))
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color="white",
                colormap="tab10",
                prefer_horizontal=0.9,
            ).generate_from_frequencies(words_dict)

            plt.imshow(wordcloud, interpolation="bilinear")
            plt.title(f'Topic: {topic_data["name"][:50]}...')
            plt.axis("off")

            cloud_file = output_dir / f"wordcloud_topic_{topic_id}.png"
            plt.savefig(cloud_file, dpi=150, bbox_inches="tight")
            plt.close()

        if self.config.verbose:
            logger.info(f"Word clouds saved to: {output_dir}")

    def create_document_length_histogram(self, output_dir: Path) -> None:
        """
        Create document length distribution histogram.

        Args:
            output_dir: Directory to save visualization
        """
        if not self._check_visualization_available() or not self.analyzer._fitted:
            return

        doc_lengths = [
            len(doc_info["document_preview"].split())
            for doc_info in self.analyzer.document_topics
        ]

        if not doc_lengths:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(doc_lengths, bins=30, alpha=0.7, edgecolor="black", color="skyblue")
        plt.title("Document Length Distribution")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add mean line
        mean_length = np.mean(doc_lengths)
        plt.axvline(
            mean_length,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_length:.1f} words",
        )
        plt.legend()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hist_file = output_dir / "document_lengths.png"
        plt.savefig(hist_file, dpi=300, bbox_inches="tight")
        plt.close()

        if self.config.verbose:
            logger.info(f"Document length histogram saved to: {hist_file}")

    def create_all_visualizations(self, output_dir: Path) -> None:
        """
        Create all available visualizations.

        Args:
            output_dir: Directory to save visualizations
        """
        if not self.analyzer._fitted:
            logger.warning("Analyzer not fitted, skipping visualizations")
            return

        if self.config.verbose:
            logger.info("Creating visualizations...")

        self.create_topic_distribution_plot(output_dir)
        self.create_word_clouds(output_dir)
        self.create_document_length_histogram(output_dir)

        if self.config.verbose:
            logger.info(f"All visualizations saved to: {output_dir}")


class SummaryGenerator:
    """
    Generates human-readable summaries and reports.

    Separates reporting logic from analysis logic, providing
    comprehensive summaries for users and stakeholders.
    """

    def __init__(self, analyzer: TopicAnalyzer, config: TopicModelingConfig):
        """
        Initialize the summary generator.

        Args:
            analyzer: Fitted TopicAnalyzer instance
            config: Configuration object
        """
        self.analyzer = analyzer
        self.config = config

    def print_console_summary(self) -> None:
        """Print topic modeling summary to console."""
        if not self.analyzer._fitted:
            logger.warning("Analyzer not fitted, no summary available")
            return

        all_topics = self.analyzer.get_all_topics()
        stats = self.analyzer.get_topic_statistics()

        print("\n" + "=" * 80)
        print("BERTOPIC DOCUMENT TOPIC MODELING RESULTS")
        print("=" * 80)

        print(f"\n📊 OVERVIEW:")
        print(f"   • Total Topics Found: {len(all_topics)}")
        print(f"   • Total Documents: {stats.get('total_documents', 0)}")
        print(f"   • Outlier Documents: {stats.get('outlier_documents', 0)}")
        print(f".2f")
        print("\n🎯 TOP TOPICS:")

        # Show top 10 topics sorted by document count
        sorted_topics = sorted(
            all_topics.items(), key=lambda x: x[1]["count"], reverse=True
        )[:10]

        for i, (topic_id, topic_data) in enumerate(sorted_topics, 1):
            print(f"\n   {i}. {topic_data['name']}")
            print(f"      📄 Documents: {topic_data['count']}")
            top_words = [word for word, _ in topic_data["words"][:5]]
            print(f"      🔑 Top words: {', '.join(top_words)}")

            # Show sample document if available
            sample_docs = topic_data.get("representative_docs", [])
            if sample_docs:
                sample = sample_docs[0][:80] + "..."
                print(f"      💬 Sample: {sample}")

        print("\n" + "=" * 80)

    def generate_cluster_summary(self, output_dir: Path) -> None:
        """
        Generate formatted cluster summary file.

        Args:
            output_dir: Directory to save summary
        """
        if not self.analyzer._fitted:
            logger.warning("Analyzer not fitted, skipping cluster summary")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "cluster_summary.txt"

        all_topics = self.analyzer.get_all_topics()
        stats = self.analyzer.get_topic_statistics()

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("BERTopic Cluster Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Topics: {len(all_topics)}\n")
            f.write(f"Total Documents: {stats.get('total_documents', 0)}\n")
            f.write(f"Outlier Documents: {stats.get('outlier_documents', 0)}\n\n")

            f.write("Topic Clusters (sorted by document count):\n")
            f.write("-" * 50 + "\n\n")

            # Sort topics by document count
            sorted_topics = sorted(
                all_topics.items(), key=lambda x: x[1]["count"], reverse=True
            )

            for topic_id, topic_data in sorted_topics:
                name = topic_data["name"]
                count = topic_data["count"]
                words = topic_data.get("words", [])

                f.write(f"{name}\n")
                f.write(f"   Documents: {count}\n")

                if words:
                    top_words = [word for word, score in words[:5]]
                    f.write(f"   Top words: {', '.join(top_words)}\n")

                sample_docs = topic_data.get("representative_docs", [])
                if sample_docs:
                    sample = sample_docs[0][:80] + "..."
                    f.write(f"   Sample: {sample}\n")

                f.write("\n")

        if self.config.verbose:
            logger.info(f"Cluster summary saved to: {summary_file}")


class BERTopicProcessor:
    """
    Main processor class coordinating the entire topic modeling pipeline.

    Implements the facade pattern to provide a simple interface while
    managing specialized components internally.
    """

    def __init__(self, config: Optional[TopicModelingConfig] = None):
        """
        Initialize the BERTopic processor.

        Args:
            config: Optional configuration object (uses defaults if None)
        """
        self.config = config or TopicModelingConfig()

        # Initialize components (created when first used)
        self.model_builder = None
        self.analyzer = None
        self.exporter = None
        self.visualizer = None
        self.summarizer = None

        logger.info("BERTopic processor initialized")

    def process_documents(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Complete topic modeling pipeline: build model → fit → analyze.

        Args:
            documents: List of document texts to process

        Returns:
            Tuple of (topic_assignments, probabilities)

        Raises:
            ValueError: If no documents provided
            RuntimeError: If processing fails
        """
        if not documents:
            raise ValueError("Cannot process empty document list")

        try:
            # Initialize components on first use
            if self.model_builder is None:
                self.model_builder = ModelBuilder(self.config)

            # Build and configure the model
            model = self.model_builder.build_model(documents)

            # Create analyzer and initialize dependent components
            self.analyzer = TopicAnalyzer(model, self.config)
            self.exporter = ResultExporter(self.analyzer, self.config)
            self.visualizer = VisualizationManager(self.analyzer, self.config)
            self.summarizer = SummaryGenerator(self.analyzer, self.config)

            # Fit the model and analyze results
            topics, probabilities = self.analyzer.fit_transform(documents)

            logger.info("Document processing completed successfully")
            return topics, probabilities

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise RuntimeError(f"BERTopic processing failed: {e}") from e

    def save_results(self, output_dir: Path) -> None:
        """
        Save all results using the exporter.

        Args:
            output_dir: Directory to save results
        """
        if self.exporter is None:
            raise RuntimeError("No exporter available. Call process_documents() first.")
        self.exporter.save_all_results(output_dir)

    def create_visualizations(self, output_dir: Path) -> None:
        """
        Create all visualizations using the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        if self.visualizer is None:
            raise RuntimeError(
                "No visualizer available. Call process_documents() first."
            )
        self.visualizer.create_all_visualizations(output_dir)

    def generate_summary(self, output_dir: Optional[Path] = None) -> None:
        """
        Generate summary using the summarizer.

        Args:
            output_dir: Optional directory to save summary file
        """
        if self.summarizer is None:
            raise RuntimeError(
                "No summarizer available. Call process_documents() first."
            )

        self.summarizer.print_console_summary()
        if output_dir:
            self.summarizer.generate_cluster_summary(output_dir)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from the analyzer.

        Returns:
            Dictionary with various statistics
        """
        if self.analyzer is None:
            return {}
        return self.analyzer.get_topic_statistics()

    def get_topics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all discovered topics.

        Returns:
            Dictionary mapping topic_id to topic information
        """
        if self.analyzer is None:
            return {}
        return self.analyzer.get_all_topics()
