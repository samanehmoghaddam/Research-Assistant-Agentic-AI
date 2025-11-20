from langchain.chat_models import ChatOpenAI
from retrieval.context_assembler import assemble_context


class ResearchAssistantAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

    def answer(self, query, hits, documents):
        context = assemble_context(hits, documents)
        prompt = (
            "Use the following retrieved context from research papers to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a concise answer and cite source file/page numbers."
        )
        return self.llm.predict(prompt)
