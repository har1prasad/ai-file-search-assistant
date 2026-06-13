import os
import logging
import hashlib
import numpy as np
from dotenv import load_dotenv
from google import genai
from app.extraction.extractor import chunk_text
from app.embeddings.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class ChatEngine:
    """LLM interaction engine supporting multi-turn chat sessions and local RAG context retrieval."""

    def __init__(self, db_manager=None, faiss_manager=None):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.embedding_manager = EmbeddingManager()
        
        # Injectable managers to retrieve pre-computed embeddings directly
        self.db_manager = db_manager
        self.faiss_manager = faiss_manager
        
        self.sessions = {}
        logger.info("Gemini initialized successfully.")

    def ask(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text

    def reset_chat(self) -> None:
        """Clear all active conversation sessions to free up memory and start fresh."""
        self.sessions.clear()
        logger.info("Conversational chat sessions cleared.")

    def ask_about_file(self, file_content: str, question: str, file_id: int | None = None) -> str:
        """Ask Gemini questions about a selected file using local RAG and multi-turn memory.

        If file_id is provided and chunks are cached in FAISS, retrieves embeddings directly
        without re-generating them, speeding up local execution.
        """
        file_content = file_content.strip()
        if not file_content:
            return "File is empty."

        file_hash = hashlib.md5(file_content.encode('utf-8', errors='ignore')).hexdigest()

        # 1. Retrieve matching chunks for context (uses pre-computed vectors if available)
        context = self._get_context(file_content, question, file_id)

        # 2. Initialize Gemini chat session if not already existing
        if file_hash not in self.sessions:
            chat = self.client.chats.create(
                model="gemini-2.5-flash",
                config={
                    "system_instruction": (
                        "You are an AI assistant helping a user understand a file.\n"
                        "Answer questions based ONLY on the provided context.\n"
                        "If the answer cannot be determined from the context, state that clearly.\n"
                        "Be concise and clear in your responses."
                    )
                }
            )
            self.sessions[file_hash] = {
                "chat": chat,
                "history": []
            }

        session = self.sessions[file_hash]
        chat = session["chat"]

        # 3. Prompt format
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        # 4. Generate response through conversational chat session
        try:
            response = chat.send_message(prompt)
            answer = response.text.strip()
        except Exception as exc:
            logger.exception("Gemini API call failed: %s", exc)
            return f"Error communicating with Gemini: {exc}"

        # 5. Append to conversation log
        session["history"].append(("User", question))
        session["history"].append(("AI", answer))

        # 6. Format dialogue history in Markdown
        formatted_history = []
        for speaker, text in session["history"]:
            formatted_history.append(f"### 💬 {speaker}\n{text}")

        return "\n\n---\n\n".join(formatted_history)

    def _get_context(self, file_content: str, question: str, file_id: int | None = None) -> str:
        """Extract top relevant chunks from text based on query vector similarity.

        Attempts to retrieve cached vector embeddings from FAISS to avoid CPU re-computation.
        """
        if len(file_content) <= 3000:
            return file_content

        # Try to pull pre-computed embeddings using the file ID
        if file_id is not None and self.db_manager is not None and self.faiss_manager is not None:
            try:
                chunks = self.db_manager.get_chunks_by_file_id(file_id)
                if chunks:
                    chunk_ids = [c["id"] for c in chunks]
                    chunk_texts = [c["content"] for c in chunks]
                    
                    # Direct FAISS index vector retrieval
                    chunk_embeddings = self.faiss_manager.get_embeddings(chunk_ids)
                    if chunk_embeddings is not None and len(chunk_embeddings) == len(chunk_ids):
                        query_embedding = self.embedding_manager.generate_embedding(question)
                        scores = np.dot(chunk_embeddings, query_embedding)
                        top_indices = np.argsort(scores)[::-1][:5]
                        logger.info("Retrieved %d chunk vectors directly from FAISS (cache hit).", len(chunk_ids))
                        return "\n\n---\n\n".join(chunk_texts[idx] for idx in top_indices)
            except Exception as exc:
                logger.warning("Failed to retrieve embeddings from FAISS cache: %s. Falling back to encoding.", exc)

        # Fallback to computing chunk embeddings on the fly
        chunks = chunk_text(file_content, size=800, overlap=150)
        if not chunks:
            return file_content[:10000]

        try:
            chunk_embeddings = self.embedding_manager.generate_embeddings(chunks)
            query_embedding = self.embedding_manager.generate_embedding(question)

            scores = np.dot(chunk_embeddings, query_embedding)
            top_indices = np.argsort(scores)[::-1][:5]
            return "\n\n---\n\n".join(chunks[idx] for idx in top_indices)
        except Exception as exc:
            logger.warning("Local RAG context retrieval failed: %s. Using fallback.", exc)
            return file_content[:10000]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chat = ChatEngine()

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        print(chat.ask(q))