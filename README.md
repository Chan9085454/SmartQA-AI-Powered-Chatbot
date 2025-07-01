SmartQA: Document-Based Question Answering Chatbot

SmartQA is an AI-powered chatbot that answers user questions by retrieving the most relevant information from uploaded documents (e.g., PDFs). It uses semantic search, NLP embeddings, and vector similarity to return highly relevant, context-aware answers.

---

Project Structure

```bash
.
├── chunk.py         # Splits raw text or PDFs into manageable text chunks
├── embedding.py     # Converts text chunks into vector embeddings
├── QA_system.py     # Handles user queries and retrieves best-matching answers
├── app.py           # Flask/FastAPI app to deploy the QA system
├── data/            # Folder containing raw or cleaned documents
├── embeddings/      # Stores generated embeddings or FAISS index
└── README.md        # Project documentation
