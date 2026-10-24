import os
from groq import Groq
from app.config import settings
from app.schemas import SearchResultItem


class AnswerGenerator:
    """Handles grounded natural-language generation over retrieved context."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.GROQ_MODEL_NAME
        self._client: Groq | None = None

    def _get_client(self) -> Groq | None:
        """Lazily initialize the Groq client so dynamically injected keys are detected."""
        if self._client is not None:
            return self._client

        # Check settings first, fallback to os.environ
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if api_key:
            self._client = Groq(api_key=api_key)
        return self._client

    @property
    def is_configured(self) -> bool:
        return self._get_client() is not None

    def generate_answer(self, query: str, context_chunks: list[SearchResultItem]) -> str:
        """Synthesizes a response strictly grounded in the provided document chunks."""
        client = self._get_client()
        if not client:
            return (
                "LLM Generation is disabled: 'GROQ_API_KEY' not found in environment variables. "
                "Retrieved chunks are available in the 'sources' field."
            )

        if not context_chunks:
            return "No relevant context was found in the documentation to answer your question."

        # 1. Format context with verifiable source labels
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            formatted_context += (
                f"[Document {i}] (Source: {chunk.source_file})\n"
                f"{chunk.content}\n\n"
            )

        # 2. Strict system prompt to prevent hallucination
        system_instruction = (
            "You are a technical documentation assistant. Answer the user's question "
            "strictly using only the provided context. If the answer cannot be determined "
            "from the context, state that you do not know. Always cite the source file name "
            "when making factual claims."
        )

        user_content = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{formatted_context}\n"
            f"User Question: {query}\n"
            f"Answer:"
        )

        # 3. Safe inference call with error handling
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,  # Low temperature for factual consistency
            )
            return response.choices[0].message.content or "No answer returned."
        except Exception as e:
            return f"Error during generation from Groq API: {str(e)}"


# Singleton instance
generator = AnswerGenerator()