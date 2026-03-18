"""FastAPI backend for the AI Browser Agent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.embedding.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore
from backend.services.pipeline import Pipeline
from backend.services.retriever import Retriever
from backend.services.chat_memory import ChatMemory
from backend.llm.groq_client import GroqClient
from backend.services.cache_manager import PageCache
from backend.services.link_ranker import rank_links

# --- Initialize components ---
app = FastAPI(title="AI Browser Agent", version="1.0.0")

embedder = Embedder()
store = FAISSStore(dimension=embedder.dimension)
llm = GroqClient()
pipeline = Pipeline(embedder=embedder, store=store, llm=llm)
retriever = Retriever(embedder=embedder, store=store)
memory = ChatMemory(max_exchanges=5)
page_cache = PageCache()

# Track the currently processed URL
current_url: str | None = None
current_summary: str | None = None
current_links: list[dict] = []


# --- Request/Response models ---
class ProcessRequest(BaseModel):
    url: str


class ProcessResponse(BaseModel):
    status: str
    num_chunks: int
    title: str
    summary: str
    internal_links: list[dict]


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
    global current_url, current_summary, current_links

    try:
        # Clear previous state
        store.clear()
        memory.clear()
        page_cache.clear() # Limpiamos la caché para la nueva sesión

        # Run pipeline
        result = await asyncio.to_thread(run_pipeline_sync, request.url)

        current_url = request.url
        current_summary = result["summary"]
        current_links = result["internal_links"]

        return ProcessResponse(
            status=result["status"],
            num_chunks=result["num_chunks"],
            title=result["title"],
            summary=current_summary,
            internal_links=current_links
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


import json
from backend.crawler.playwright_crawler import crawl_url
from backend.parser.html_parser import parse_html

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using Agentic RAG with manual JSON tool calling."""
    global current_url, current_summary, current_links

    if current_url is None or store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No URL has been processed yet. Please process a URL first.",
        )

    try:
        results = retriever.retrieve(request.question, k=5)
        context_parts = [
            f"[Source: {r['metadata']['url']}]\n{r['text'][:600]}"  # cap per chunk
            for r in results
        ]
        context = "\n\n---\n\n".join(context_parts)
        links_str = "\n".join(
            [f"- {l['text']}: {l['url']}" for l in current_links[:20]]
        )
        history = memory.get_history()

        ranked_links = rank_links(
            question=request.question,
            links=current_links,
            embedder=embedder,
            top_k=5,           # Only best 5 links — saves tokens, improves focus
        )
        links_str = "\n".join(
            [f"- {l['text']}: {l['url']}" for l in ranked_links]
        )

        system_prompt = (
            "You are an AI Research Assistant exploring a website.\n"
            f"Root Page URL: {current_url}\n"
            f"Root Page Summary: {current_summary}\n\n"
            "## Most Relevant Internal Links for this question:\n"
            f"{links_str}\n\n"
            "## Instructions\n"
            "1. Read the Retrieved Context carefully.\n"
            "2. IMPORTANT: If the answer is not explicitly stated in the context, "
            "you MUST call scrape_url on the most relevant link above. "
            "Never say 'I don't have enough information' — always try to find it first.\n"
            "3. After scraping, answer using the new content.\n"
            "4. Always cite the Source URL.\n\n"
            + llm.get_tool_instructions()
        )
        messages = [{"role": "system", "content": system_prompt}]

        if history and history != "No previous conversation.":
            messages.append({
                "role": "user",
                "content": f"Previous conversation:\n{history}",
            })

        messages.append({
            "role": "user",
            "content": (
                f"Retrieved Context:\n{context}\n\n"
                f"Question: {request.question}"
            ),
        })

        max_iterations = 5
        for iteration in range(max_iterations):

            response = llm.generate(messages=messages)

            if response.wants_tool:
                target_url = response.tool_call.arguments["url"]
                print(f"[Iteration {iteration+1}] Agent scraping: {target_url}")

                # Add the model's tool-call turn to history
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                try:
                    cached = page_cache.get(target_url)
                    if cached:
                        scraped_text = cached[:8000]
                        tool_result = f"[Cached] Content of {target_url}:\n\n{scraped_text}"
                    else:
                        def run_incremental_sync(u):
                            if sys.platform == "win32":
                                asyncio.set_event_loop_policy(
                                    asyncio.WindowsProactorEventLoopPolicy()
                                )
                            return asyncio.run(pipeline.process_incremental_url(u))

                        full_text = await asyncio.to_thread(
                            run_incremental_sync, target_url
                        )
                        page_cache.set(target_url, full_text)
                        scraped_text = full_text[:8000]
                        tool_result = (
                            f"Content of {target_url}:\n\n{scraped_text}"
                        )

                except Exception as e:
                    tool_result = f"Failed to scrape {target_url}: {str(e)}"

                # Feed result back as a user turn (simple, no tool_call_id needed)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Here is the content you requested from {target_url}:\n\n"
                        f"{tool_result}\n\n"
                        "Now answer the original question using this information."
                    ),
                })

            else:
                # Final answer
                answer = response.content
                memory.add(request.question, answer)
                return ChatResponse(answer=answer, source_url=current_url)

        fallback = "I needed too many steps to find the answer. Please ask a more specific question."
        memory.add(request.question, fallback)
        return ChatResponse(answer=fallback, source_url=current_url)

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
