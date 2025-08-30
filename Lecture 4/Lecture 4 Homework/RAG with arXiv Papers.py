import os
import json
import pickle
from typing import List, Dict, Any
import numpy as np
import requests
import fitz  # PyMuPDF
import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path

class ArxivRAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG system with embedding model and FAISS index."""
        self.model = SentenceTransformer(model_name)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.papers = []
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.chunk_to_paper = []  # Maps chunk index to paper index
        
    def download_arxiv_papers(self, category: str = "cs.CL", max_results: int = 50) -> List[Dict]:
        """Download arXiv papers from specified category."""
        print(f"Searching for {max_results} papers in {category}...")
        
        # Search for papers
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            paper_info = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'published': result.published.strftime("%Y-%m-%d"),
                'arxiv_id': result.entry_id.split('/')[-1]
            }
            papers.append(paper_info)
            print(f"Found: {paper_info['title']}")
        
        self.papers = papers
        return papers
    
    def download_pdf(self, pdf_url: str, filename: str) -> str:
        """Download PDF from arXiv URL."""
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            pdf_path = f"downloads/{filename}.pdf"
            os.makedirs("downloads", exist_ok=True)
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            return pdf_path
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')
        return text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoder.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks with token limit."""
        tokens = self.encoder.encode(text)
        chunks = []
        
        if len(tokens) <= max_tokens:
            chunks.append(self.encoder.decode(tokens))
            return chunks
        
        step = max_tokens - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if i + max_tokens >= len(tokens):
                break
        
        return chunks
    
    def process_papers(self):
        """Download, extract, and chunk all papers."""
        print("Processing papers...")
        
        for i, paper in enumerate(self.papers):
            print(f"Processing paper {i+1}/{len(self.papers)}: {paper['title']}")
            
            # Download PDF
            pdf_path = self.download_pdf(paper['pdf_url'], paper['arxiv_id'])
            if not pdf_path:
                continue
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                continue
            
            # Clean text
            text = self.clean_text(text)
            
            # Chunk text
            paper_chunks = self.chunk_text(text)
            
            # Add chunks with paper reference
            for chunk in paper_chunks:
                self.chunks.append(chunk)
                self.chunk_to_paper.append(i)
            
            # Clean up downloaded PDF
            try:
                os.remove(pdf_path)
            except:
                pass
        
        print(f"Total chunks created: {len(self.chunks)}")
    
    def create_embeddings(self):
        """Generate embeddings for all chunks."""
        print("Generating embeddings...")
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def build_faiss_index(self):
        """Build FAISS index for similarity search."""
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        print("FAISS index built successfully")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks given a query."""
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_k
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                paper_idx = self.chunk_to_paper[idx]
                results.append({
                    'chunk': self.chunks[idx],
                    'paper_title': self.papers[paper_idx]['title'],
                    'paper_authors': self.papers[paper_idx]['authors'],
                    'paper_summary': self.papers[paper_idx]['summary'],
                    'paper_url': self.papers[paper_idx]['pdf_url'],
                    'arxiv_id': self.papers[paper_idx]['arxiv_id'],
                    'published': self.papers[paper_idx]['published'],
                    'similarity_score': float(1 / (1 + distance)),
                    'rank': i + 1
                })
        
        return results
    
    def save_system(self, filename: str = "rag_system.pkl"):
        """Save the RAG system to disk."""
        data = {
            'papers': self.papers,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'chunk_to_paper': self.chunk_to_paper
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"System saved to {filename}")
    
    def load_system(self, filename: str = "rag_system.pkl"):
        """Load the RAG system from disk."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.papers = data['papers']
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.chunk_to_paper = data['chunk_to_paper']
            
            # Rebuild FAISS index
            self.build_faiss_index()
            print(f"System loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False

# Initialize FastAPI app
app = FastAPI(title="arXiv CS.CL RAG System")
templates = Jinja2Templates(directory="templates")

# Create templates directory
os.makedirs("templates", exist_ok=True)

# Global RAG system instance
rag_system = ArxivRAGSystem()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_papers(query: str = Form(...)):
    """Search papers and return relevant chunks."""
    if not rag_system.faiss_index:
        return {"error": "RAG system not initialized. Please run the data collection first."}
    
    results = rag_system.search(query, top_k=5)
    return {"query": query, "results": results}

@app.post("/initialize")
async def initialize_system():
    """Initialize the RAG system by downloading and processing papers."""
    try:
        # Download papers
        rag_system.download_arxiv_papers(category="cs.CL", max_results=50)
        
        # Process papers
        rag_system.process_papers()
        
        # Create embeddings
        rag_system.create_embeddings()
        
        # Build FAISS index
        rag_system.build_faiss_index()
        
        # Save system
        rag_system.save_system()
        
        return {"message": "System initialized successfully", "papers": len(rag_system.papers), "chunks": len(rag_system.chunks)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
async def get_status():
    """Get system status."""
    return {
        "initialized": rag_system.faiss_index is not None,
        "papers": len(rag_system.papers),
        "chunks": len(rag_system.chunks) if rag_system.chunks else 0
    }

# Create HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv CS.CL RAG System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .search-section {
            margin-bottom: 30px;
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            padding: 12px 24px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .init-button {
            background-color: #e74c3c;
            margin-bottom: 20px;
        }
        .init-button:hover {
            background-color: #c0392b;
        }
        .status {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .results {
            margin-top: 20px;
        }
        .result-item {
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .paper-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            cursor: help;
            position: relative;
            border-bottom: 1px dotted #bdc3c7;
        }
        .paper-title:hover::after {
            content: "📖 " attr(data-summary);
            position: absolute;
            bottom: 100%;
            left: 0;
            background: #34495e;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: normal;
            max-width: 500px;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            line-height: 1.4;
        }
        .paper-authors {
            color: #7f8c8d;
            margin-bottom: 10px;
        }
        .chunk-text {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            border: 1px solid #ddd;
        }
        .similarity-score {
            color: #27ae60;
            font-weight: bold;
        }
        .paper-actions {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .open-paper-btn {
            background-color: #e67e22;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            display: inline-block;
        }
        .open-paper-btn:hover {
            background-color: #d35400;
        }
        .paper-meta {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .arxiv-id {
            color: #7f8c8d;
            font-size: 12px;
            font-family: monospace;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 3px;
        }
        .paper-date {
            color: #7f8c8d;
            font-size: 12px;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 3px;
        }
        .loading {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 arXiv CS.CL RAG System</h1>
        
        <div class="search-section">
            <button class="init-button" onclick="initializeSystem()">🚀 Initialize System (Download 50 Papers)</button>
            
            <div class="status" id="status">
                <strong>Status:</strong> <span id="statusText">Not initialized</span>
            </div>
            
            <div class="search-box">
                <input type="text" id="queryInput" placeholder="Enter your question about CS.CL papers..." />
                <button onclick="searchPapers()">🔍 Search</button>
            </div>
            
            <div style="text-align: center; color: #7f8c8d; font-size: 14px; margin-top: 10px;">
                💡 <strong>Tip:</strong> Hover over paper titles to see summaries, click "Open Full Paper" to read the complete article
            </div>
        </div>
        
        <div id="results" class="results"></div>
    </div>

    <script>
        // Check status on page load
        window.onload = function() {
            checkStatus();
        };

        async function checkStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                updateStatus(data);
            } catch (error) {
                console.error('Error checking status:', error);
            }
        }

        function updateStatus(data) {
            const statusText = document.getElementById('statusText');
            if (data.initialized) {
                statusText.innerHTML = `✅ Initialized - ${data.papers} papers, ${data.chunks} chunks`;
            } else {
                statusText.innerHTML = `❌ Not initialized`;
            }
        }

        async function initializeSystem() {
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = '⏳ Initializing...';
            button.disabled = true;
            
            try {
                const response = await fetch('/initialize', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    alert('System initialized successfully!');
                    checkStatus();
                }
            } catch (error) {
                alert('Error initializing system: ' + error);
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        }

        async function searchPapers() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

            try {
                const formData = new FormData();
                formData.append('query', query);

                const response = await fetch('/search', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (data.error) {
                    resultsDiv.innerHTML = `<div class="error">${data.error}</div>`;
                } else {
                    displayResults(data);
                }
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error">Error performing search</div>';
                console.error('Search error:', error);
            }
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (!data.results || data.results.length === 0) {
                resultsDiv.innerHTML = '<div class="loading">No results found</div>';
                return;
            }

            let html = `<h3>🔍 Search Results for: "${data.query}"</h3>`;
            
            data.results.forEach(result => {
                html += `
                    <div class="result-item">
                        <div class="paper-title" data-summary="${result.paper_summary}">📄 ${result.paper_title}</div>
                        <div class="paper-authors">👥 Authors: ${result.paper_authors.join(', ')}</div>
                        <div class="paper-meta">
                            <div class="arxiv-id">📚 arXiv ID: ${result.arxiv_id}</div>
                            <div class="paper-date">📅 Published: ${result.published || 'N/A'}</div>
                            <div class="similarity-score">🎯 Relevance: ${(result.similarity_score * 100).toFixed(1)}%</div>
                        </div>
                        <div class="chunk-text">${result.chunk}</div>
                        <div class="paper-actions">
                            <a href="${result.paper_url}" target="_blank" class="open-paper-btn">🔗 Open Full Paper</a>
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }

        // Allow Enter key to trigger search
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchPapers();
            }
        });
    </script>
</body>
</html>
"""

# Create the HTML template file
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

if __name__ == "__main__":
    print("Starting arXiv CS.CL RAG System...")
    print("1. First, visit http://localhost:8000")
    print("2. Click 'Initialize System' to download and process 50 CS.CL papers")
    print("3. Once initialized, you can search through the papers")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
