# 🚀 ReLitAgent — Research Literature Retrieval Agent

ReLitAgent is an agentic AI system that helps researchers recall relevant concepts, retrieve supporting passages from previously-read papers, and extract accurate citations while writing.  
It ingests PDFs, embeds and indexes them, and uses semantic search plus an LLM-based reasoning agent to surface the most relevant excerpts.

---

## 🎯 Key Features

- 📄 PDF ingestion and clean text extraction  
- 🧩 Section-aware chunking for scientific structure  
- 🔍 Semantic vector search using FAISS or NumPy fallback  
- 🧠 Agentic reasoning: LLM synthesizes retrieved context  
- 📚 Citation-aware excerpt retrieval  
- 🖥️ CLI interface + modular architecture  
- ⚙️ Extensible components suitable for research workflows  

---

## 📦 Repository Structure

```
ResearchAssistantAgent/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── agent/
│   │   ├── researcher_agent.py
│   │   └── prompts.py
│   │
│   ├── ingestion/
│   │   ├── pdf_reader.py
│   │   ├── section_extractor.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   ├── embedding_model.py
│   │   └── openai_embeddings.py
│   │
│   ├── index/
│   │   ├── index_builder.py
│   │   ├── index_loader.py
│   │   └── searcher.py
│   │
│   ├── retrieval/
│   │   ├── semantic_retriever.py
│   │   └── context_assembler.py
│   │
│   ├── cli/
│   │   └── main.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   └── logging.py
│   │
│   └── fallback/
│       └── fallback_FindRelevantPaper.py
│
├── data/
│   ├── pdfs/
│   ├── vector_store/
│   └── cache/
│
└── notebooks/
    ├── demo_pipeline.ipynb
    ├── evaluate_retrieval.ipynb
    └── inspect_embeddings.ipynb
```

---

## 📐 System Architecture

```
                         ┌────────────────────────┐
                         │        PDF Files        │
                         └────────────┬───────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │           PDF Ingestion             │
                   │     (pdf_reader + sections)         │
                   └──────────────────┬──────────────────┘
                                      │
                           Text Cleaning & Chunking
                                      │
                           ┌──────────┴─────────┐
                           │     Embeddings     │
                           │ (OpenAI / ST-T5)   │
                           └──────────┬─────────┘
                                      │
                         ┌────────────┴────────────┐
                         │     Vector Index         │
                         │   (FAISS / NumPy)        │
                         └────────────┬────────────┘
                                      │
                              Query Embedding
                                      │
                             Top-K Similar Chunks
                                      │
                           ┌──────────┴──────────┐
                           │   Agentic LLM        │
                           │ researcher_agent.py  │
                           └──────────┬──────────┘
                                      │
                                   Answer
                              + Source Citations
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ReLitAgent.git
cd ReLitAgent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

## 📥 Build the Index (first-time)

Place PDFs inside:

```
data/pdfs/
```

Then run:

```bash
python -m src.cli.main build_index data/pdfs data/vector_store
```

This will:

- load PDFs  
- extract text  
- embed chunks  
- build FAISS (or NumPy) index  
- save metadata  

---

## 🔍 Query the Papers

Example:

```bash
python -m src.cli.main ask "taxonomy with new dataset for abusive language" data/vector_store
```

---

## 🧪 Programmatic Example

```python
from index.searcher import Searcher
from agent.researcher_agent import ResearchAssistantAgent

searcher = Searcher("data/vector_store")
agent = ResearchAssistantAgent()

hits = searcher.search("hierarchical concept drift", k=5)
answer = agent.answer("hierarchical concept drift", hits, documents=None)

print(answer)
```

---

## 🧭 Roadmap

- Support local embedding models (bge, mxbai)  
- Add citation graph embeddings  
- Add agentic multi-step retrieval  
- Add PDF annotation export  
- Add Streamlit interface  

---

## 🤝 Contributing

Contributions are welcome.  
Please open an issue before submitting PRs.

---

## 📄 License

MIT License — free for academic and commercial use.
```

