"""Target A: an off-the-shelf RAG framework implementation.

This is the "someone else's system" side of the comparison. It uses llama-index
as published (VectorStoreIndex + retriever + compact response synthesis) with a
thin adapter so the evaluation harness can talk to it using the same interface
as the baseline.

Honest note for the README: model/embedding access is plugged in through our
provider adapter, so this measures *the framework's pipeline*, not the vendor's
model quality.
"""

from __future__ import annotations

from pathlib import Path

from evals import llm as llm_mod

SYSTEM_PROMPT = (
    "Answer ONLY from the provided context. If the context does not contain the answer, "
    "reply with exactly I_DO_NOT_KNOW. Never use outside knowledge."
)


def _import_li():
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex  # noqa: F401
        from llama_index.core.embeddings import BaseEmbedding  # noqa: F401
        from llama_index.core.llms import (  # noqa: F401
            ChatMessage, ChatResponse, CompletionResponse, CustomLLM, LLMMetadata,
        )
        from llama_index.core.response_synthesizers import get_response_synthesizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "llama-index-core is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return True


class OssRAG:
    name = "oss-rag(llama-index)"

    def __init__(self, corpus_chunks: list[tuple[str, str]], provider: llm_mod.BaseProvider | None = None,
                 top_k: int = 5):
        """corpus_chunks: list of (page_ref, text)."""
        _import_li()
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
        from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse, CustomLLM, LLMMetadata
        from llama_index.core.response_synthesizers import get_response_synthesizer

        self.provider = provider or llm_mod.get_provider()
        self.top_k = top_k

        outer = self

        class ProviderEmbedding(BaseEmbedding):
            """Adapter: our provider's vectors, wrapped as a llama-index embedding."""

            def _get_query_embedding(self, query: str):
                return outer.provider.embed([query])[0].tolist()

            async def _aget_query_embedding(self, query: str):
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text: str):
                return outer.provider.embed([text])[0].tolist()

            def _get_text_embeddings(self, texts):
                return [v.tolist() for v in outer.provider.embed(list(texts))]

        class ProviderLLM(CustomLLM):
            @property
            def metadata(self):
                return LLMMetadata(
                    context_window=8192, num_output=512, model_name="provider-adapter",
                    is_chat_model=True, system_prompt=SYSTEM_PROMPT,
                )

            def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
                out = outer.provider.chat([{"role": "user", "content": prompt}])
                return CompletionResponse(text=out["text"])

            def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
                raise NotImplementedError("streaming not used in evaluation")

            def chat(self, messages, **kwargs) -> ChatResponse:
                payload = [{"role": m.role.value if hasattr(m.role, "value") else str(m.role),
                            "content": m.content} for m in messages]
                if not any(m["role"] == "system" for m in payload):
                    payload.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                out = outer.provider.chat(payload)
                return ChatResponse(message=ChatMessage(role="assistant", content=out["text"]))

            def stream_chat(self, messages, **kwargs):
                raise NotImplementedError("streaming not used in evaluation")

        Settings.embed_model = ProviderEmbedding()
        Settings.llm = ProviderLLM()

        docs = [Document(text=text, metadata={"page": page}) for page, text in corpus_chunks]
        self.index = VectorStoreIndex.from_documents(docs)
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        self.synth = get_response_synthesizer(response_mode="compact")
        self._last_nodes = []

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        if k and k != self.top_k:
            self.retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = self.retriever.retrieve(question)
        self._last_nodes = nodes
        pages = []
        for n in nodes:
            node = getattr(n, "node", n)
            pages.append(str(node.metadata.get("page", "?")) if hasattr(node, "metadata") else "?")
        return pages

    def answer(self, question: str) -> dict:
        nodes = self.retriever.retrieve(question)
        self._last_nodes = nodes
        response = self.synth.synthesize(query=question, nodes=nodes)
        text = str(response).strip()
        pages = []
        for n in nodes:
            node = getattr(n, "node", n)
            pages.append(str(node.metadata.get("page", "?")) if hasattr(node, "metadata") else "?")
        return {
            "text": text,
            "refused": text.upper().startswith("I_DO_NOT_KNOW"),
            "contexts": [getattr(getattr(n, "node", n), "text", "") for n in nodes],
            "retrieved": pages,
        }
