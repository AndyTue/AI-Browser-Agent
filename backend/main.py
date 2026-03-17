"""FastAPI backend for the AI Browser Chatbot."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.embedding.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore
from backend.services.pipeline import Pipeline
from backend.services.retriever import Retriever
from backend.services.chat_memory import ChatMemory
from backend.llm.groq_client import GroqClient

# --- Initialize components ---
app = FastAPI(title="AI Browser Chatbot", version="1.0.0")

embedder = Embedder()
store = FAISSStore(dimension=embedder.dimension)
pipeline = Pipeline(embedder=embedder, store=store)
retriever = Retriever(embedder=embedder, store=store)
memory = ChatMemory(max_exchanges=5)
llm = GroqClient()

# Track the currently processed URL
current_url: str | None = None


# --- Request/Response models ---
class ProcessRequest(BaseModel):
    url: str


class ProcessResponse(BaseModel):
    status: str
    chunks_count: int
    title: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    source_url: str | None


# --- Endpoints ---
import asyncio
import sys

def run_pipeline_sync(url: str):
    """Run the pipeline in a fresh asyncio loop (solves Windows ProactorEventLoop issues)."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.run(pipeline.process_url(url))

@app.post("/process", response_model=ProcessResponse)
async def process_url(request: ProcessRequest):
    """Process a URL: crawl, parse, chunk, embed, and store in FAISS."""
    global current_url

    try:
        # Clear previous state
        store.clear()
        memory.clear()

        # Run pipeline in a dedicated thread to avoid Uvicorn event loop conflicts with Playwright
        result = await asyncio.to_thread(run_pipeline_sync, request.url)

        current_url = request.url

        return ProcessResponse(
            status=result["status"],
            chunks_count=result["num_chunks"],
            title=result["title"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using retrieved context from the processed URL."""
    global current_url

    if current_url is None or store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No URL has been processed yet. Please process a URL first.",
        )

    try:
        # Retrieve relevant chunks
        results = retriever.retrieve(request.question, k=5)

        # Build context from retrieved chunks
        context_parts = []
        for r in results:
            url = r["metadata"]["url"]
            context_parts.append(f"[Source: {url}]\n{r['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Get conversation history
        history = memory.get_history()

        # Generate answer
        answer = llm.generate(context=context, history=history, question=request.question)

        # Save to memory
        memory.add(request.question, answer)

        return ChatResponse(answer=answer, source_url=current_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "url_processed": current_url is not None,
        "current_url": current_url,
        "vectors_stored": store.total_vectors,
    }
