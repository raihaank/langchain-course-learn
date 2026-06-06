from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def create_rag_chain(vectorstore=None):
    """Creates a RAG chain with conversation memory and a strict system prompt."""

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, streaming=True)

    template = """You are a helpful and conversational assistant for question-answering tasks.
If the user's input is a simple greeting (like "hi" or "how are you"), respond politely and conversationally.

Use the following pieces of retrieved context to answer the question.
If the answer is explicitly contained in the context, extract and summarize it.
If the answer is NOT contained in the context, you MUST answer exactly: "Sorry Mate. This information is not available in the provided document."
Do not try to make up an answer. Do not use your prior knowledge for factual questions.

Format your answer in a summarized and pleasing manner. If there are multiple points or a list, provide a brief introductory sentence (e.g., "The key points are listed below:") followed by bullet points.

--- Conversation History ---
{chat_history}
----------------------------

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain
