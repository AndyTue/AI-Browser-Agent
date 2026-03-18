"""FastAPI backend for the AI Browser Agent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.embedding.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore
from backend.services.pipeline import Pipeline
from backend.services.retriever import Retriever
from backend.services.chat_memory import ChatMemory
from backend.llm.groq_client import GroqClient

# --- Initialize components ---
app = FastAPI(title="AI Browser Agent", version="1.0.0")

embedder = Embedder()
store = FAISSStore(dimension=embedder.dimension)
llm = GroqClient()
pipeline = Pipeline(embedder=embedder, store=store, llm=llm)
retriever = Retriever(embedder=embedder, store=store)
memory = ChatMemory(max_exchanges=5)

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
    global current_url

    try:
        # Clear previous state
        store.clear()
        memory.clear()

        # Run pipeline in a dedicated thread to avoid Uvicorn event loop conflicts with Playwright
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
    """Answer a question using Agentic RAG and Tool Calling."""
    global current_url, current_summary, current_links

    if current_url is None or store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No URL has been processed yet. Please process a URL first.",
        )

    try:
        # Retrieve relevant chunks from the root page
        results = retriever.retrieve(request.question, k=5)

        # Build context from retrieved chunks
        context_parts = []
        for r in results:
            url = r["metadata"]["url"]
            context_parts.append(f"[Source: {url}]\n{r['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Build list of links as string
        links_str = "\n".join([f"- {link['text']}: {link['url']}" for link in current_links])

        # Get conversation history
        history = memory.get_history()

        system_prompt = (
            "You are an AI Assistant that helps users find information on a website.\n"
            f"Root Page URL: {current_url}\n"
            f"Root Page Summary: {current_summary}\n\n"
            "Available Internal Links:\n"
            f"{links_str}\n\n"
            "Instructions:\n"
            "1. Answer the user's question using the retrieved context below.\n"
            "2. If you need more information from the internal links, use the 'scrape_url' tool to fetch it.\n"
            "3. Always cite the Source URL when answering.\n"
            "4. Do not hallucinate.\n\n"
            "CRITICAL: Do not output raw <function> or XML tags in your response. Use the native JSON tool calling format."
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            messages.append({"role": "user", "content": f"Previous conversation history:\n{history}"})

        user_content = f"Retrieved Root Page Context:\n{context}\n\nQuestion: {request.question}"
        messages.append({"role": "user", "content": user_content})

        tools = [llm.get_scrape_tool_schema()]
        max_iterations = 5
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            response_msg = llm.generate(messages=messages, tools=tools)

            if response_msg.tool_calls:
                # Add the assistant's tool call message exactly as dict
                # Exclude None values to prevent Groq API 400 errors on the next iteration
                messages.append(response_msg.model_dump(exclude_none=True))

                for tool_call in response_msg.tool_calls:
                    if tool_call.function.name == "scrape_url":
                        args = json.loads(tool_call.function.arguments)
                        target_url = args.get("url")
                        
                        try:
                            # Perform real-time scraping
                            html = await crawl_url(target_url)
                            parsed = parse_html(html, target_url)
                            scraped_text = parsed["text"][:8000] # Limit size
                            
                            # Add tool response
                            messages.append({
                                "role": "tool",
                                "name": "scrape_url",
                                "tool_call_id": tool_call.id,
                                "content": f"Successfully scraped {target_url}:\n\n{scraped_text}"
                            })
                        except Exception as e:
                            print(f"Failed to scrape tool url: {e}")
                            messages.append({
                                "role": "tool",
                                "name": "scrape_url",
                                "tool_call_id": tool_call.id,
                                "content": f"Failed to scrape {target_url}: {str(e)}"
                            })
            else:
                # No tool calls, got final answer
                answer = response_msg.content
                memory.add(request.question, answer)
                return ChatResponse(answer=answer, source_url=current_url)

        # Output fallback if loop maxes out
        fallback_answer = "I needed too many steps to answer. Could you ask a more specific question?"
        memory.add(request.question, fallback_answer)
        return ChatResponse(answer=fallback_answer, source_url=current_url)

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
