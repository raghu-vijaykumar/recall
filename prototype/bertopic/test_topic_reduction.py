#!/usr/bin/env python3
"""
Test script to demonstrate topic reduction strategies for BERTopic clustering.
Tests different parameter combinations to reduce the number of topics from 450+ to a more manageable number.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    from bertopic_clustering import DocumentTopicModeler, load_sample_documents

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("❌ BERTopic clustering module not found")
    sys.exit(1)


def test_topic_reduction_strategies():
    """Test different strategies for reducing topic count"""

    print("🎯 BERTOPIC TOPIC REDUCTION TESTING")
    print("=" * 60)
    print(
        "Testing parameter combinations to reduce topic count from 450+ to ~50-100 topics"
    )
    print()

    # Load sample documents (expand to create more documents for testing)
    base_documents = load_sample_documents()
    # Repeat documents to simulate larger dataset
    documents = []
    for i in range(5):  # 5x expansion = 100 documents
        for doc in base_documents:
            documents.append(f"Doc_{i+1}: {doc}")

    print(
        f"📄 Testing with {len(documents)} documents (expanded from {len(base_documents)})"
    )
    print()

    # Define test configurations
    test_configs = [
        {
            "name": "Baseline (High Topics)",
            "min_topic_size": 3,
            "cluster_epsilon": 0.1,
            "umap_neighbors": 15,
            "umap_components": 5,
            "description": "Original settings that likely produce 450+ topics",
        },
        {
            "name": "Increased Min Topic Size",
            "min_topic_size": 10,
            "cluster_epsilon": 0.1,
            "umap_neighbors": 15,
            "umap_components": 5,
            "description": "Increase minimum topic size from 3 to 10",
        },
        {
            "name": "Coarser Clustering",
            "min_topic_size": 10,
            "cluster_epsilon": 0.4,
            "umap_neighbors": 15,
            "umap_components": 5,
            "description": "Increase cluster_epsilon for coarser clustering",
        },
        {
            "name": "Reduced Embedding Dimensions",
            "min_topic_size": 10,
            "cluster_epsilon": 0.4,
            "umap_neighbors": 15,
            "umap_components": 3,
            "description": "Reduce UMAP components from 5 to 3 (more aggressive reduction)",
        },
        {
            "name": "Smaller Neighborhood",
            "min_topic_size": 10,
            "cluster_epsilon": 0.4,
            "umap_neighbors": 8,
            "umap_components": 3,
            "description": "Reduce UMAP neighbors from 15 to 8 (coarser embeddings)",
        },
        {
            "name": "Combined Aggressive Reduction",
            "min_topic_size": 15,
            "cluster_epsilon": 0.6,
            "umap_neighbors": 5,
            "umap_components": 2,
            "description": "Aggressive combination for maximum topic reduction",
        },
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        print(f"\n🔄 Test {i}/{len(test_configs)}: {config['name']}")
        print(f"   {config['description']}")
        print(
            f"   Params: min_topic_size={config['min_topic_size']}, epsilon={config['cluster_epsilon']}, neighbors={config['umap_neighbors']}, components={config['umap_components']}"
        )

        try:
            # Initialize modeler with test parameters
            modeler = DocumentTopicModeler(
                model_name="all-MiniLM-L6-v2",
                min_topic_size=config["min_topic_size"],
                cluster_epsilon=config["cluster_epsilon"],
                umap_neighbors=config["umap_neighbors"],
                umap_components=config["umap_components"],
                verbose=False,  # Reduce noise
            )

            # Fit model
            topics, probs = modeler.fit_transform(documents)

            # Get results
            all_topics = modeler.get_all_topics()
            num_topics = len(all_topics)

            # Calculate simple metrics
            topic_sizes = [topic["count"] for topic in all_topics.values()]
            avg_topic_size = sum(topic_sizes) / len(topic_sizes) if topic_sizes else 0
            max_topic_size = max(topic_sizes) if topic_sizes else 0
            min_topic_size_result = min(topic_sizes) if topic_sizes else 0

            result = {
                "config_name": config["name"],
                "description": config["description"],
                "params": {
                    "min_topic_size": config["min_topic_size"],
                    "cluster_epsilon": config["cluster_epsilon"],
                    "umap_neighbors": config["umap_neighbors"],
                    "umap_components": config["umap_components"],
                },
                "num_topics": num_topics,
                "avg_topic_size": round(avg_topic_size, 1),
                "max_topic_size": max_topic_size,
                "min_topic_size": min_topic_size_result,
                "status": "success",
            }

            results.append(result)
            print(
                f"   ✅ Result: {num_topics} topics (avg size: {avg_topic_size:.1f} docs)"
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")
            result = {
                "config_name": config["name"],
                "description": config["description"],
                "params": {
                    "min_topic_size": config["min_topic_size"],
                    "cluster_epsilon": config["cluster_epsilon"],
                    "umap_neighbors": config["umap_neighbors"],
                    "umap_components": config["umap_components"],
                },
                "num_topics": 0,
                "avg_topic_size": 0.0,
                "max_topic_size": 0,
                "min_topic_size": 0,
                "status": "failed",
                "error": str(e),
            }
            results.append(result)

    # Analyze and display results
    print("\n" + "=" * 80)
    print("📊 TOPIC REDUCTION RESULTS SUMMARY")
    print("=" * 80)

    # Sort by number of topics (ascending)
    success_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] != "success"]

    success_results.sort(key=lambda x: x["num_topics"])

    print("\n🎯 SUCCESSFUL CONFIGURATIONS (sorted by topic count):")
    print("-" * 60)

    for result in success_results:
        print(f"📈 {result['config_name']}: {result['num_topics']} topics")
        print(f"   Avg topic size: {result['avg_topic_size']} docs")
        print(f"   Range: {result['min_topic_size']}-{result['max_topic_size']} docs")
        print(
            f"   ε={result['params']['cluster_epsilon']}, n={result['params']['umap_neighbors']}, c={result['params']['umap_components']}"
        )
        print()

    if failed_results:
        print("❌ FAILED CONFIGURATIONS:")
        print("-" * 40)
        for result in failed_results:
            print(f"   {result['config_name']}: {result['error']}")
        print()

    # Find best result
    if success_results:
        best_result = success_results[0]  # Already sorted by topic count
        if best_result["num_topics"] <= 100:
            print("🏆 BEST CONFIGURATION (lowest topic count ≤ 100):")
        else:
            print("🏆 BEST CONFIGURATION (lowest topic count available):")
        print(f"   {best_result['config_name']}: {best_result['num_topics']} topics")
        print(
            f"   Settings: min_topic_size={best_result['params']['min_topic_size']}, epsilon={best_result['params']['cluster_epsilon']}, neighbors={best_result['params']['umap_neighbors']}, components={best_result['params']['umap_components']}"
        )
        print(f"   Performance: avg {best_result['avg_topic_size']} docs per topic")
        print()

        # Check if we achieved the goal
        if best_result["num_topics"] <= 100:
            print("🎉 SUCCESS: Achieved topic count of 100 or fewer!")
            print("   This configuration should work well for reducing 450+ topics.")
        elif best_result["num_topics"] <= 200:
            print("⚠️  PARTIAL SUCCESS: Topic count in acceptable range (100-200).")
            print("   May need further tuning for even fewer topics.")
        else:
            print("❌ GOAL NOT MET: Topic count still too high.")
            print("   Consider more aggressive parameter settings.")

    # Save results to file
    output_file = "topic_reduction_test_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "test_summary": {
                    "total_configs_tested": len(test_configs),
                    "successful_tests": len(success_results),
                    "failed_tests": len(failed_results),
                    "input_documents": len(documents),
                    "topic_range_achieved": (
                        f"{min(r['num_topics'] for r in success_results)}-{max(r['num_topics'] for r in success_results)}"
                        if success_results
                        else "N/A"
                    ),
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n🔧 RECOMMENDED USAGE:")
    print(
        "python bertopic_clustering.py --min-topic-size 15 --cluster-epsilon 0.4 --umap-neighbors 8 --umap-components 3 --folder /path/to/documents"
    )

    return results


if __name__ == "__main__":
    if not IMPORTS_AVAILABLE:
        print("❌ Required dependencies not available")
        sys.exit(1)

    try:
        test_topic_reduction_strategies()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
