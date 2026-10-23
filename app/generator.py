import os
from groq import Groq
from app.schemas import SearchResultItem

class AnswerGenerator:
  """Handles grounded natural-language generation over retrueved context."""

  def __init__(self,model_name: str = "openai/gpt-oss-120b"):
    self.model_name = model_name
    self.api_key = os.getenv("GROQ_API_KEY")
    self.client = None

    if self.api_key:
      self.client = Groq(api_key=self.api_key)

  @property
  def is_configured(self) -> bool:
    return self.client is not None

  def generate_answer(self, query: str, context_chunks: list[SearchResultItem]) -> str:
    """Synthesizes a response strictly grounded in the provided document chunks."""
    if not self.is_configured:
      # Fallback if no API key is provided
      return(
        "LLM Generation is disabled: 'GROQ_API_KEY' not found in environment variables."
        "Retrieved chunks are available in the 'sources' field."
      )

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

    # 3. Fast inference call
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1  # Low temperature for factual consistency
    )

    return response.choices[0].message.content


# Singleton instance
generator = AnswerGenerator()  