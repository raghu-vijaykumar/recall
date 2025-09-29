#!/usr/bin/env python3
"""
BERTopic Enhanced - Production Ready CLI

Refactored CLI that replicates the original functionality using the new OOP architecture.
Supports all the original command line arguments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import from the new refactored package
from src.bertopic.core import BERTopicProcessor, TopicModelingConfig
from src.bertopic.preprocessing import (
    DocumentPreprocessor,
    PreprocessingConfig,
    FolderPreprocessor,
    PreprocessingPipeline,
)
from src.bertopic.knowledge_graph import TopicKnowledgeGraphBuilder
from src.bertopic.utils import DocumentLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Command line interface with all original arguments supported."""

    parser = argparse.ArgumentParser(
        description="BERTopic Enhanced - Production Ready Topic Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bertopic_clustering.py --folder "C:\\workspace\\recall-test\\docs" --preprocess-folder --model intfloat/multilingual-e5-base --output results --summary
  python bertopic_clustering.py --file documents.txt --model all-MiniLM-L6-v2 --kg --visualize
  python bertopic_clustering.py --folder ./docs --incremental --force-index
        """,
    )

    # Input options (same as original)
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to file containing documents (.txt, .csv, .json)",
    )
    parser.add_argument(
        "--folder", type=str, help="Path to folder containing documents"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Maximum files to process"
    )

    # Preprocessing options
    parser.add_argument(
        "--preprocess-folder",
        action="store_true",
        help="Use preprocessing pipeline for folder input",
    )

    # Model selection (same as original)
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-base",
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
        help="Sentence transformer model name (default: intfloat/multilingual-e5-base)",
    )

    # Topic modeling parameters (same as original)
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=15,
        help="Minimum documents per topic (default: 15)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Maximum topics to extract (default: None = auto)",
    )
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.4,
        help="HDBSCAN cluster merging threshold (default: 0.4)",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=8,
        help="UMAP neighborhood size (default: 8)",
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=3,
        help="UMAP output dimensions (default: 3)",
    )

    # Feature flags (same as original)
    parser.add_argument("--kg", action="store_true", help="Generate knowledge graph")
    parser.add_argument(
        "--context-window-size",
        type=int,
        default=3,
        help="Context window size for knowledge graph (default: 3)",
    )
    parser.add_argument(
        "--incremental", action="store_true", help="Enable incremental indexing"
    )
    parser.add_argument(
        "--force-index", action="store_true", help="Force full re-indexing"
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default="results/topic_index.json",
        help="Index metadata file (default: results/topic_index.json)",
    )

    # Output options (same as original)
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate cluster summary"
    )
    parser.add_argument(
        "--save-d3-data", action="store_true", help="Save D3 visualization data"
    )

    # Logging options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print progress information (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress information"
    )

    args = parser.parse_args()

    # Validation
    if not args.file and not args.folder:
        parser.error("Provide --file or --folder argument")

    verbose = args.verbose and not args.quiet

    print("🚀 BERTopic Enhanced - Production Ready")
    print("=" * 50)

    try:
        # ===================================================================
        # STEP 1: LOAD DOCUMENTS
        # ===================================================================

        documents = []
        files_processed = []

        if args.folder:
            print(f"📁 Processing folder: {args.folder}")
            documents, files_processed = DocumentLoader.load_from_folder(
                args.folder, args.max_files
            )
        elif args.file:
            print(f"📄 Loading file: {args.file}")
            documents = DocumentLoader.load_from_file(args.file)

        if not documents:
            print("❌ No documents found!")
            return 1

        print(f"✅ Loaded {len(documents)} documents")

        # ===================================================================
        # STEP 2: PREPROCESSING (if requested)
        # ===================================================================

        if args.preprocess_folder and args.folder:
            print("🔄 Using preprocessing pipeline...")

            # Use FolderPreprocessor for full folder analysis with extension stats
            folder_processor = FolderPreprocessor()

            # Create temporary output directory for preprocessing
            temp_output_dir = os.path.join(args.output, "temp_preprocessing")
            os.makedirs(temp_output_dir, exist_ok=True)

            # Run folder preprocessing (this includes extension analysis)
            preprocess_results = folder_processor.preprocess_folder(
                args.folder, temp_output_dir, verbose=verbose
            )

            if "error" in preprocess_results:
                print(f"❌ Preprocessing failed: {preprocess_results['error']}")
                return 1

            # Load the processed documents
            processed_docs_file = preprocess_results["output_file"]
            import json

            with open(processed_docs_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents = data["documents"]

            print("✅ Preprocessing completed:")
            print(f"   Original: {len(documents)} documents")
            print(f"   After filtering: {len(documents)} documents")
            print(f"   Files processed: {preprocess_results.get('files_processed', 0)}")

            # Show extension analysis results
            if "extension_analysis" in preprocess_results:
                ext_stats = preprocess_results["extension_analysis"]
                print("📊 File Extension Analysis:")
                print(f"   Total files: {ext_stats['total_files']}")
                print(f"   Text files: {ext_stats['text_files_count']}")
                print(f"   Binary files: {ext_stats['binary_files_count']}")

                # Show top 5 extensions
                sorted_ext = sorted(
                    ext_stats["extensions"].items(), key=lambda x: x[1], reverse=True
                )
                print("   Top extensions:")
                for ext, count in sorted_ext[:5]:
                    print(f"     {ext}: {count} files")

        # ===================================================================
        # STEP 3: TOPIC MODELING
        # ===================================================================

        print(f"🤖 Processing with model: {args.model}")

        # Configure topic modeling
        config = TopicModelingConfig(
            model_name=args.model,
            min_topic_size=args.min_topic_size,
            max_topics=args.max_topics,
            cluster_epsilon=args.cluster_epsilon,
            umap_neighbors=args.umap_neighbors,
            umap_components=args.umap_components,
            verbose=verbose,
        )

        processor = BERTopicProcessor(config)
        topics, probs = processor.process_documents(documents)

        # ===================================================================
        # STEP 4: SAVE RESULTS
        # ===================================================================

        # Create output directory structure like original
        model_short = (
            args.model.replace("sentence-transformers/", "")
            .replace("intfloat/", "")
            .replace("BAAI/", "")
            .replace("Qwen/", "")
            .replace("-", "")
            .replace("_", "")
            .lower()
        )

        if args.folder:
            folder_name = os.path.basename(os.path.normpath(args.folder))
        elif args.file:
            folder_name = os.path.splitext(os.path.basename(args.file))[0]
        else:
            folder_name = "sample"

        output_dir = os.path.join(args.output, f"{model_short}_{folder_name}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Results: {output_dir}/")

        # Save results
        processor.save_results(output_dir)

        if args.visualize:
            processor.create_visualizations(output_dir)

        if args.summary:
            processor.generate_summary(output_dir)

        # ===================================================================
        # STEP 5: KNOWLEDGE GRAPH (if requested)
        # ===================================================================

        if args.kg:
            print("\n🔗 Generating Knowledge Graph...")
            try:
                kg_builder = TopicKnowledgeGraphBuilder(
                    context_window_size=args.context_window_size
                )
                knowledge_graph = kg_builder.build_from_bertopic(
                    processor.analyzer, documents
                )

                # Save knowledge graph
                kg_file = os.path.join(output_dir, "knowledge_graph.json")
                with open(kg_file, "w") as f:
                    import json

                    json.dump(knowledge_graph, f, indent=2, default=str)

                print("✅ Knowledge graph saved")

                # Save D3 visualization data if requested
                if args.save_d3_data or args.kg:
                    from src.bertopic.knowledge_graph import (
                        KnowledgeGraphNavigator,
                    )

                    navigator = KnowledgeGraphNavigator(knowledge_graph)

                    viz_file = os.path.join(output_dir, "visualization_data.json")
                    viz_data = navigator.export_for_visualization()
                    with open(viz_file, "w") as f:
                        json.dump(viz_data, f, indent=2, default=str)

                    print("✅ D3 visualization data saved")

            except Exception as e:
                print(f"⚠ Knowledge graph generation failed: {e}")
                print("   (This may be expected if optional dependencies are missing)")

        print("\n✅ Processing completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
