# Enhanced BERTopic Knowledge Graph Prototype

This enhanced BERTopic prototype creates **reader-friendly knowledge graphs** from topic clusters, featuring context window extraction, hierarchical topic relationships, and incremental indexing.

## 🚀 Features

### 🎯 Core Capabilities
- **Context Window Extraction**: Automatically extracts relevant text windows around topic mentions
- **Hierarchical Relationships**: Builds topic-subtopic relationships and concept associations
- **Interactive D3.js Visualization**: Web-based graph exploration with search and filtering
- **Incremental Indexing**: Only reprocess changed/modified files
- **Multiple Embedding Models**: Support for 9+ sentence transformer models

### 📂 Folder Processing & Model Selection
```bash
# Process any folder with automatic model selection
python bertopic_clustering.py --folder ./your-documents --model bge-base-en-v1.5 --kg
python bertopic_clustering.py --folder ./docs --model all-MiniLM-L6-v2 --kg --incremental
```

### 🔗 Knowledge Graph Generation
- **Node Types**: Topics, Concepts, Documents
- **Edge Types**: Topic membership, hierarchical relationships, concept relationships
- **Reader-Friendly Output**: Categorized and searchable knowledge structure

### ⚡ Incremental Indexing
```bash
# Only process changes, skip unchanged files
python bertopic_clustering.py --folder ./docs --incremental --skip-if-unchanged
# Force full reprocessing if needed
python bertopic_clustering.py --folder ./docs --incremental --force-index
```

### 🌐 Interactive Visualization
**Open `d3_visualization.html` in any browser** for:
- Interactive graph exploration with zoom and pan
- Node filtering by type (Topics/Concepts/Documents)
- Edge filtering by relationship type
- Search functionality across all nodes
- Click-to-explore detailed node information

## 📁 Project Structure

```
prototype/bertopic/
├── bertopic_clustering.py          # Main CLI script with all features
├── knowledge_graph_builder.py      # KG construction and context extraction
├── incremental_index_manager.py    # Change detection and incremental processing
├── d3_visualization.html           # Interactive web visualization
├── bertopic_results/               # Standard topic modeling outputs
├── bertopic_knowledge_graph_demo/  # KG-specific outputs
└── README.md                       # This documentation
```

## 🛠️ Usage Examples

### Basic Usage
```bash
# Demo with sample data
python bertopic_clustering.py --demo

# Process a single file
python bertopic_clustering.py --file documents.txt --kg

# Process an entire folder recursively
python bertopic_clustering.py --folder ./documents --kg --visualize --summary
```

### Advanced Usage with Model Selection
```bash
# Use different embedding models
python bertopic_clustering.py --folder ./docs --model intfloat/multilingual-e5-base --kg
python bertopic_clustering.py --folder ./docs --model BAAI/bge-base-en-v1.5 --kg --incremental

# Control topic model parameters
python bertopic_clustering.py --folder ./docs --kg --min-topic-size 5 --max-topics 20

# Full pipeline with visualization
python bertopic_clustering.py --folder ./docs --kg --visualize --summary --save-d3-data
```

### Incremental Processing
```bash
# First run - creates index
python bertopic_clustering.py --folder ./docs --incremental --kg

# Subsequent runs - only processes changes
python bertopic_clustering.py --folder ./docs --incremental --kg

# Force complete reprocessing
python bertopic_clustering.py --folder ./docs --incremental --force-index --kg
```

## 📊 Output Files

### Topic Modeling Results (`bertopic_results/`)
- `topics.json` - Topic information and keywords
- `document_assignments.json` - Document-to-topic mappings
- `topic_info.csv` - Detailed topic statistics
- `topic_visualizations.png` - Visual analysis charts
- `wordcloud_topic_*.png` - Individual topic word clouds

### Knowledge Graph Results (`bertopic_knowledge_graph_demo/`)
- `knowledge_graph.json` - Complete graph structure
- `visualization_data.json` - D3.js compatible format
- `topic_overview.json` - Hierarchical topic navigation
- `d3_visualization.html` - Interactive web viewer
- `cluster_summary.txt` - Human-readable KG summary

## 🔧 Configuration Options

### Model Selection
Choose from these embedding models:
- `all-MiniLM-L6-v2` (default - fast, good quality)
- `all-mpnet-base-v2` (higher quality, slower)
- `BAAI/bge-base-en-v1.5` (specialized for retrieval)
- `intfloat/multilingual-e5-base` (multilingual support)
- `intfloat/multilingual-e5-large` (largest multilingual model)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `Qwen/Qwen3-Embedding-0.6B` (large embedding model)
- `nomic-ai/nomic-embed-text-v1.5` (mixture of experts)

### Knowledge Graph Settings
- `--context-window-size`: Sentences around topic mentions (default: 3)
- `--min-topic-size`: Minimum documents per topic (default: 3)
- `--max-topics`: Maximum number of topics (default: auto)

### Incremental Processing
- `--index-file`: Custom path for index metadata file
- `--force-index`: Ignore change detection, force full reprocessing

## 🌟 Key Innovations

### 1. **Reader-Friendly Knowledge Graph**
- **Topic Exploration**: Click any topic to see related concepts and documents
- **Concept Extraction**: Automatically identifies key concepts within context windows
- **Hierarchical Navigation**: Topics organized by document count and relationships

### 2. **Context Window Intelligence**
- **Semantic Context**: Extracts meaningful text windows around keyword mentions
- **Concept Discovery**: Groups related mentions into coherent concepts
- **Relationship Building**: Links concepts to topics with weighted relationships

### 3. **Incremental Processing**
- **Change Detection**: Hash-based file monitoring
- **Selective Processing**: Only reprocess modified/added files
- **Index Management**: Persistent metadata for efficient updates

### 4. **Multi-Model Support**
- **Model Comparison**: Built-in framework for comparing embedding models
- **Domain Adaptation**: Choose models best suited for your content
- **Performance Optimization**: Balance speed vs. quality based on needs

## 🎯 Use Cases

### Document Analysis
- **Research Paper Collections**: Extract themes and relationships
- **Legal Document Libraries**: Discover topic patterns and hierarchies
- **News Article Archives**: Track topic evolution over time

### Content Organization
- **Knowledge Base Management**: Visual topic exploration
- **Educational Content**: Hierarchical learning path discovery
- **Technical Documentation**: Concept relationship mapping

### Business Intelligence
- **Customer Feedback Analysis**: Identify pain points and themes
- **Product Documentation**: Topic-based content navigation
- **Policy Document Review**: Relationship and context extraction

## 🐛 Troubleshooting

### Common Issues
1. **Memory Errors**: Use `--max-files` to limit processing or try smaller models
2. **Empty Results**: Check file encoding (UTF-8 recommended) or use `--force-index`
3. **Visualization Not Loading**: Ensure web browser allows local file access

### Performance Tips
- **Large Datasets**: Use `--incremental` for ongoing processing
- **Model Selection**: Start with `all-MiniLM-L6-v2` for good balance
- **Memory Constraints**: Use `min-topic-size=5` or higher for cleaner topics

## 🤝 Integration Notes

This prototype can be integrated into the main Recall application by:
1. Using the `TopicKnowledgeGraphBuilder` class for KG construction
2. Utilizing the `IncrementalIndexManager` for efficient updates
3. Adapting the D3.js visualization for the frontend interface
4. Incorporating model selection into the backend services

## 📈 Future Enhancements

- **Advanced NLP**: spaCy integration for better concept extraction
- **Time-Based Analysis**: Topic evolution over document timestamps
- **Multi-Language Support**: Enhanced processing for non-English content
- **Real-time Updates**: WebSocket-based incremental visualization
- **Similarity Metrics**: Advanced algorithm-based topic relationship detection

---

**🚀 Ready to explore your documents?** Try: `python bertopic_clustering.py --demo`
