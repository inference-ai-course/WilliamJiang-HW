# arXiv CS.CL RAG System

A complete Retrieval-Augmented Generation (RAG) system for arXiv Computer Science Computation and Language (CS.CL) papers.

## Features

- **Automatic Paper Collection**: Downloads 50 latest CS.CL papers from arXiv
- **PDF Processing**: Extracts raw text from PDFs using PyMuPDF
- **Text Chunking**: Splits papers into chunks of ≤512 tokens with overlap
- **Semantic Search**: Uses sentence-transformers for embedding generation
- **FAISS Indexing**: Fast similarity search using FAISS
- **Web Interface**: Beautiful HTML interface for querying the system
- **Persistent Storage**: Saves processed data for future use

## System Architecture

```
arXiv Papers → PDF Download → Text Extraction → Text Cleaning → Chunking → Embeddings → FAISS Index → Web Interface
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python "RAG with arXiv Papers.py"
   ```

3. **Open Browser**: Navigate to `http://localhost:8000`

4. **Initialize**: Click "Initialize System" to download and process 50 papers (this may take 5-10 minutes)

5. **Search**: Once initialized, enter questions about CS.CL topics and get relevant paper chunks

## API Endpoints

- `GET /` - Main web interface
- `POST /initialize` - Initialize the RAG system
- `POST /search` - Search for relevant paper chunks
- `GET /status` - Get system status

## Example Queries

- "What are the latest advances in large language models?"
- "How do researchers evaluate machine translation systems?"
- "What are the challenges in natural language processing?"
- "Explain recent developments in speech recognition"
- "What are the applications of transformers in NLP?"

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Chunk Size**: ≤512 tokens with 50 token overlap
- **Search Results**: Top 5 most relevant chunks
- **Similarity Metric**: L2 distance in FAISS
- **Text Processing**: Automatic cleaning and normalization

## File Structure

- `RAG with arXiv Papers.py` - Main application
- `templates/index.html` - Web interface template
- `rag_system.pkl` - Saved system data (created after initialization)
- `downloads/` - Temporary PDF storage (auto-cleaned)

## Performance

- **Initialization**: ~5-10 minutes for 50 papers
- **Search**: <1 second for most queries
- **Memory Usage**: ~100-200MB for embeddings
- **Storage**: ~50-100MB for processed data

## Troubleshooting

- **Import Errors**: Ensure virtual environment is activated
- **PDF Download Issues**: Check internet connection and arXiv availability
- **Memory Issues**: Reduce `max_results` in the code if needed
- **Port Conflicts**: Change port in the code if 8000 is busy

## Future Enhancements

- Add more arXiv categories
- Implement advanced chunking strategies
- Add document summarization
- Support for different embedding models
- Export functionality for research purposes
