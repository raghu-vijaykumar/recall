# BERTopic Enhanced - Production Ready Knowledge Graph Topic Modeling

A production-ready Python package that enhances BERTopic with document preprocessing, knowledge graph generation, incremental indexing, and comprehensive visualization capabilities.

## ✨ Features

- **🔄 Document Preprocessing**: Clean, filter, and deduplicate text documents
- **🤖 Advanced Topic Modeling**: BERTopic with optimized configurations
- **🕸️ Knowledge Graphs**: Automatic concept extraction and relationship mapping
- **📊 Rich Visualizations**: Interactive charts, word clouds, and D3.js graphs
- **🔄 Incremental Processing**: Efficient updates without full re-processing
- **📦 Production Ready**: Proper package structure, logging, and error handling

## 🚀 Installation

### From Source
```bash
git clone https://github.com/your-repo/bertopic-enhanced.git
cd bertopic-enhanced
pip install -r requirements.txt
pip install -e .
```

### Using pip (when published)
```bash
pip install bertopic-enhanced
```

### Optional Dependencies
```bash
# For enhanced preprocessing
pip install bertopic-enhanced[preprocessing]

# For development
pip install bertopic-enhanced[dev]

# For advanced visualizations
pip install bertopic-enhanced[visualization]
```

## 📖 Quick Start

### Basic Topic Modeling
```python
from bertopic import BERTopicProcessor, TopicModelingConfig

# Configure your model
config = TopicModelingConfig(
    model_name="all-MiniLM-L6-v2",  # Fast and reliable
    min_topic_size=10,
    verbose=True
)

# Create processor and process documents
processor = BERTopicProcessor(config)

documents = [
    "Machine learning is transforming industries",
    "Natural language processing helps understand text",
    "Deep learning uses neural networks",
    # ... more documents
]

topics, probabilities = processor.process_documents(documents)

# Get results
topics_info = processor.get_topics()
stats = processor.get_statistics()

# Save results
processor.save_results("./results")
processor.create_visualizations("./results")
processor.generate_summary()
```

### Document Preprocessing Pipeline
```python
from bertopic.preprocessing import DocumentPreprocessor, PreprocessingConfig

# Configure preprocessing
config = PreprocessingConfig(
    remove_urls=True,
    remove_emails=True,
    min_words_per_doc=10,
    remove_duplicates=True,
    allowed_languages=["en"]
)

preprocessor = DocumentPreprocessor(config)

# Preprocess documents
processed_docs, stats = preprocessor.preprocess_documents(raw_documents)

print(f"Processed {stats['final_count']} documents")
```

### Knowledge Graph Generation
```python
from bertopic.knowledge_graph import TopicKnowledgeGraphBuilder

# Build knowledge graph from processed topics
kg_builder = TopicKnowledgeGraphBuilder(context_window_size=3)
knowledge_graph = kg_builder.build_from_bertopic(
    modeler=processor.analyzer,  # The fitted analyzer
    documents=documents,
    min_relationship_strength=0.3
)

# Explore the graph
navigator = KnowledgeGraphNavigator(knowledge_graph)
topic_overview = navigator.get_topic_overview()
search_results = navigator.search_graph("machine learning")
```

### Incremental Processing
```python
from bertopic.incremental import IncrementalIndexManager

# Set up incremental indexing
index_manager = IncrementalIndexManager(index_file="./topic_index.json")

# Check if reprocessing is needed
file_paths = ["./documents/file1.txt", "./documents/file2.txt"]
if index_manager.should_reindex(file_paths):
    # Process documents
    processor = BERTopicProcessor()
    topics, probs = processor.process_documents(documents)

    # Update index
    index_manager.update_index_metadata(file_paths)
```

## 📁 Project Structure

```
bertopic-enhanced/
├── src/
│   └── bertopic/
│       ├── __init__.py          # Package initialization
│       ├── core.py              # Main BERTopic processor
│       ├── preprocessing.py     # Document preprocessing pipeline
│       ├── knowledge_graph.py   # Knowledge graph building & navigation
│       ├── incremental.py       # Incremental indexing system
│       └── utils.py             # Utility classes and functions
├── tests/
│   └── test_core.py             # Core functionality tests
├── setup.py                     # Package setup script
├── requirements.txt             # Dependencies
└── README.md                    # This documentation
```

## 🔧 Configuration

### Topic Modeling Configuration
```python
from bertopic.core import TopicModelingConfig

config = TopicModelingConfig(
    model_name="all-MiniLM-L6-v2",      # Embedding model
    min_topic_size=15,                  # Minimum documents per topic
    max_topics=None,                    # Maximum topics (None = auto)
    cluster_epsilon=0.4,                # HDBSCAN clustering threshold
    umap_neighbors=8,                   # UMAP neighbors
    umap_components=3,                  # UMAP components
    verbose=True                        # Enable logging
)
```

### Preprocessing Configuration
```python
from bertopic.preprocessing import PreprocessingConfig

config = PreprocessingConfig(
    remove_urls=True,                   # Remove URLs
    remove_emails=True,                 # Remove email addresses
    normalize_unicode=True,             # Unicode normalization
    min_words_per_doc=10,              # Minimum document length
    max_words_per_doc=1000,            # Maximum document length
    allowed_languages=["en"],           # Language filtering
    remove_duplicates=True,            # Duplicate removal
    duplicate_threshold=0.95           # Similarity threshold
)
```

## 📊 Output Files

The processor generates several output files:

- `topics.json` - Detailed topic information and keywords
- `document_assignments.json` - Document-to-topic assignments
- `topic_info.csv` - Topic statistics in CSV format
- `topic_distribution.png` - Topic size distribution chart
- `wordcloud_topic_X.png` - Word clouds for each topic
- `document_lengths.png` - Document length analysis
- `cluster_summary.txt` - Human-readable summary
- `knowledge_graph.json` - Knowledge graph structure (if enabled)
- `visualization_data.json` - D3.js compatible data

## 🔍 API Reference

### Core Classes

#### BERTopicProcessor
Main processing orchestrator.

**Methods:**
- `process_documents(documents)` - Process documents and extract topics
- `save_results(output_dir)` - Save all results to files
- `create_visualizations(output_dir)` - Generate charts and plots
- `generate_summary(output_dir=None)` - Print and save summary
- `get_statistics()` - Get processing statistics
- `get_topics()` - Get detailed topic information

#### DocumentPreprocessor
Document cleaning and filtering.

**Methods:**
- `preprocess_documents(documents, verbose=True)` - Clean and filter documents
- `preprocess_single_document(text)` - Process individual document

#### TopicKnowledgeGraphBuilder
Knowledge graph construction.

**Methods:**
- `build_from_bertopic(modeler, documents, min_relationship_strength=0.3)` - Build graph from BERTopic results

#### KnowledgeGraphNavigator
Knowledge graph exploration.

**Methods:**
- `get_topic_overview()` - Get overview of all topics
- `explore_topic(topic_id)` - Explore specific topic details
- `search_graph(query, limit=10)` - Search graph nodes
- `get_similar_topics(topic_id, limit=5)` - Find similar topics

#### IncrementalIndexManager
Change detection and indexing.

**Methods:**
- `should_reindex(file_paths, force=False)` - Check if reprocessing needed
- `detect_file_changes(file_paths)` - Get detailed change information
- `update_index_metadata(file_paths, model_version="1.0")` - Update index

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_core.py::TestBERTopicProcessor::test_process_documents_basic
```

## 📈 Performance Tips

### Memory Optimization
- Use `min_topic_size=10-15` for better topic quality
- Limit `max_topics` if processing large datasets
- Use lightweight embedding models for initial exploration

### Speed Optimization
- Enable incremental processing for ongoing datasets
- Use `all-MiniLM-L6-v2` for fast processing
- Set `verbose=False` for production runs

### Quality Optimization
- Use `intfloat/multilingual-e5-base` for multilingual content
- Increase `min_topic_size` for cleaner topics
- Enable preprocessing for noisy data

## 🐛 Troubleshooting

### Common Issues

**Memory Error**: Reduce `min_topic_size` or use smaller models
**No Topics Found**: Check document quality and `min_topic_size` setting
**Slow Processing**: Use lighter embedding models or enable incremental processing
**Import Errors**: Ensure all dependencies are installed with correct versions

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.INFO)

config = TopicModelingConfig(verbose=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [BERTopic](https://github.com/MaartenGr/BERTopic) - Foundation for topic modeling
- Sentence Transformers and scikit-learn communities
- All contributors and users of the original codebase

## 📞 Support

- GitHub Issues for bug reports and feature requests
- Pull requests are welcome
- Documentation improvements appreciated

---

**🚀 Happy topic modeling!** Discover insights from your documents with BERTopic Enhanced.
