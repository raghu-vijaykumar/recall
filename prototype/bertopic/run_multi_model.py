#!/usr/bin/env python3
"""
Script to run BERTopic with multiple models and store results under prototype/bertopic/results
Enhanced version that accepts document file input and supports larger 8GB models
"""

import os
import sys
import argparse
from bertopic_clustering import (
    MultiModelTopicModeler,
    load_documents_from_file,
    load_sample_documents,
)


def main():
    """Run multi-model topic modeling with document file input support"""
    parser = argparse.ArgumentParser(
        description="BERTopic Multi-Model Analysis with Document File Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_multi_model.py --file documents.txt
  python run_multi_model.py --file data.csv --output results
  python run_multi_model.py --file articles.json --models large
  python run_multi_model.py --file docs.txt --models all --visualize --summary
  python run_multi_model.py --interactive  # Use sample data
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to file containing documents (.txt, .csv, .json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="prototype/bertopic/results",
        help="Output directory for results (default: prototype/bertopic/results)",
    )

    parser.add_argument(
        "--models",
        type=str,
        choices=["all", "large", "standard"],
        default="all",
        help="Model set to use: 'all' (default), 'large' (8GB models), 'standard' (smaller models)",
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations for each model"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate cluster summaries for each model",
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
        help="Print progress information",
    )

    args = parser.parse_args()

    print("🚀 Running Enhanced Multi-Model BERTopic Analysis")
    print("=" * 60)

    # Load documents
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1

        print(f"📄 Loading documents from: {args.file}")
        try:
            documents = load_documents_from_file(args.file)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return 1
    else:
        print("📄 Loading sample documents...")
        documents = load_sample_documents()

    print(f"✅ Loaded {len(documents)} documents for analysis")

    if len(documents) == 0:
        print("❌ No documents found!")
        return 1

    # Initialize multi-model topic modeler with specified output directory
    modeler = MultiModelTopicModeler(base_output_dir=args.output, verbose=args.verbose)

    # Configure models based on selection
    if args.models == "large":
        print("🤖 Using large 8GB-capable models...")
        modeler.model_configs = modeler._get_large_models()
    elif args.models == "standard":
        print("🤖 Using standard models...")
        modeler.model_configs = modeler._get_standard_models()
    else:
        print("🤖 Using all available models...")

    print(f"🔧 Testing {len(modeler.model_configs)} embedding models")

    # Run all models
    results = modeler.run_all_models(
        documents=documents,
        create_visualizations=args.visualize,
        generate_summaries=args.summary,
    )

    print("\n✅ Multi-model analysis completed!")
    print(f"📁 Results stored under: {args.output}")
    print(f"🤖 Tested {len(results)} models")

    # Show best performing model
    successful_results = [r for r in results if r.num_topics > 0]
    if successful_results:
        best_model = max(successful_results, key=lambda x: x.coherence_score)
        print(
            f"🏆 Best model: {best_model.model_name} (coherence: {best_model.coherence_score:.3f})"
        )

        # Show model details
        print(f"   Model: {best_model.model_config.model_name}")
        print(f"   Topics: {best_model.num_topics}")
        print(f"   Processing time: {best_model.processing_time:.2f}s")
    else:
        print("⚠️  No models completed successfully")

    return 0


if __name__ == "__main__":
    main()
