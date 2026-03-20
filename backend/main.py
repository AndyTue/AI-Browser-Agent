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

# --- Retrieval constants ---
INITIAL_RETRIEVAL_K = 8      # chunks on first retrieval (before any scrape)
POST_SCRAPE_RETRIEVAL_K = 12  # chunks after a scrape adds more content
PROACTIVE_SCRAPE_THRESHOLD = 0.60

# --- Initialize components ---
app = FastAPI(title="AI Browser Agent", version="1.0.0")

embedder = Embedder()
store = FAISSStore(dimension=embedder.dimension)
llm = GroqClient()
pipeline = Pipeline(embedder=embedder, store=store, llm=llm)
retriever = Retriever(embedder=embedder, store=store)
memory = ChatMemory(max_exchanges=3)  # reduced from 5 to cap history tokens
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
        page_cache.clear()

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


def _chunk_id(r: dict) -> str:
    """Build a unique identifier for a retrieval chunk."""
    return f"{r['metadata']['url']}:{r['metadata']['chunk_id']}"


def _filter_new_chunks(results: list[dict], seen: set[str]) -> list[dict]:
    """Return only chunks not yet seen, and register the new ones."""
    new = []
    for r in results:
        cid = _chunk_id(r)
        if cid not in seen:
            seen.add(cid)
            new.append(r)
    return new


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
        # ── Step 1: Rank links ────────────────────────────────────────
        ranked_links = rank_links(
            question=request.question,
            links=current_links,
            embedder=embedder,
            top_k=2,
        )

        # ── Step 2: Proactive scrape if score ≥ threshold ─────────────
        did_proactive_scrape = False
        top_link = ranked_links[0] if ranked_links else None

        if top_link and top_link.get("score", 0) >= PROACTIVE_SCRAPE_THRESHOLD:
            target_url = top_link["url"]
            if target_url != current_url:
                print(f"[Proactive scrape] score={top_link['score']:.3f} → {target_url}")
                try:
                    cached = page_cache.get(target_url)
                    if not cached:
                        def run_incremental_sync(u):
                            if sys.platform == "win32":
                                asyncio.set_event_loop_policy(
                                    asyncio.WindowsProactorEventLoopPolicy()
                                )
                            return asyncio.run(pipeline.process_incremental_url(u))

                        await asyncio.to_thread(run_incremental_sync, target_url)
                        cached_text = page_cache.get(target_url) or ""
                        page_cache.set(target_url, cached_text)
                    did_proactive_scrape = True
                except Exception as e:
                    print(f"[Proactive scrape] Failed: {e}")

        # ── Step 3: Single retrieval (tiered k) ──────────────────────
        retrieval_k = POST_SCRAPE_RETRIEVAL_K if did_proactive_scrape else INITIAL_RETRIEVAL_K
        results = retriever.retrieve(request.question, k=retrieval_k)

        # Deduplication set — persists across all iterations
        seen_chunk_ids: set[str] = set()
        results = _filter_new_chunks(results, seen_chunk_ids)

        context_parts = [
            f"[Source: {r['metadata']['url']}]\n{r['text']}"
            for r in results
        ]
        context = llm.truncate_context("\n\n---\n\n".join(context_parts))

        # ── Step 4: Build links string ────────────────────────────────
        links_str = "\n".join(
            [f"- {l['text']}: {l['url']}" for l in ranked_links]
        )

        # ── Step 5: Build messages ────────────────────────────────────
        history = memory.get_history_summary()

        system_prompt = (
            "You are an AI Research Assistant exploring a website.\n"
            f"Root Page URL: {current_url}\n"
            f"Root Page Summary: {current_summary}\n\n"

            "## Source Priority & Mandatory Scraping\n"
            "1. If the user asks for a LIST of items (e.g. 'What are the AWS courses?' or 'list Microsoft courses'), "
            "the root page only contains a partial menu. You MUST output the JSON tool block for `scrape_url` on the specific category link "
            "(e.g., /aws-training) rather than answering from the root page context.\n"
            "2. If the user asks for DETAILS of a specific item (e.g. 'Tell me about AWS Security Essentials'), "
            "you MUST output the JSON tool block for `scrape_url` on that specific course link to get the exact duration, price, and modules. "
            "Do not answer these questions using just the root page.\n\n"

            "## Response Scope Rules\n"
            "Match your answer strictly to what was asked:\n"
            "- If asked for a LIST (e.g. 'what courses exist', 'list the X') return ONLY a numbered or bulleted list. "
            "Extract as many as you can find. No extra sections.\n"
            "- If asked for DETAILS about a specific item, return a structured answer with all available fields.\n"
            "- Never add sections the user did not ask for.\n\n"

            "## Most Relevant Internal Links for this question:\n"
            f"{links_str}\n\n"
            "## Instructions\n"
            "1. Read the Retrieved Context carefully.\n"
            "2. If you do not have the COMPLETE list or FULL details, you MUST output the JSON tool block for `scrape_url` on the most relevant link. NEVER apologize or say you lack information without scraping first.\n"
            "3. After scraping, you WILL receive relevant chunks. Use them to answer completely and specifically.\n"
            "4. Always cite the specific Source URL you scraped as your final source.\n\n"
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

        last_scraped_url = current_url

        # ── Step 6: Agent loop ────────────────────────────────────────
        max_iterations = 5
        for iteration in range(max_iterations):

            response = llm.generate(messages=messages)

            if response.wants_tool:
                target_url = response.tool_call.arguments["url"]
                last_scraped_url = target_url
                print(f"[Iteration {iteration+1}] Agent scraping: {target_url}")

                # --- Compact assistant message (Problem 3) ---
                messages.append({
                    "role": "assistant",
                    "content": llm.extract_tool_call_summary(response.content),
                })

                try:
                    cached = page_cache.get(target_url)
                    if cached:
                        scraped_text = cached
                    else:
                        def run_incremental_sync(u):
                            if sys.platform == "win32":
                                asyncio.set_event_loop_policy(
                                    asyncio.WindowsProactorEventLoopPolicy()
                                )
                            return asyncio.run(pipeline.process_incremental_url(u))

                        scraped_text = await asyncio.to_thread(
                            run_incremental_sync, target_url
                        )
                        page_cache.set(target_url, scraped_text)

                    # --- Deduplicated, tiered retrieval (Problem 4) ---
                    fresh_results = retriever.retrieve(
                        request.question, k=POST_SCRAPE_RETRIEVAL_K
                    )

                    # Prioritize chunks from the target URL
                    target_chunks = [
                        r for r in fresh_results
                        if r["metadata"]["url"] == target_url
                    ]
                    other_chunks = [
                        r for r in fresh_results
                        if r["metadata"]["url"] != target_url
                    ]
                    ordered_results = target_chunks + other_chunks

                    # Remove already-seen chunks
                    new_chunks = _filter_new_chunks(ordered_results, seen_chunk_ids)

                    if new_chunks:
                        fresh_context = "\n\n---\n\n".join([
                            f"[Source: {r['metadata']['url']}]\n{r['text']}"
                            for r in new_chunks
                        ])
                        tool_result = llm.truncate_context(
                            f"=== NEW CHUNKS from {target_url} ===\n\n"
                            f"{fresh_context}"
                        )
                    else:
                        tool_result = (
                            f"No new relevant chunks found for {target_url}."
                        )

                except Exception as e:
                    tool_result = f"Failed to scrape {target_url}: {str(e)}"

                # --- Compact tool result message (Problem 4) ---
                messages.append({
                    "role": "user",
                    "content": (
                        f"Chunks from {target_url}:\n\n"
                        f"{tool_result}\n\n"
                        "Answer the original question using the chunks above. "
                        "Omit fields you cannot find; never say 'not specified'."
                    ),
                })

                # --- Trim system prompt links after tool call (Problem 7) ---
                sys_content = messages[0]["content"]
                links_header = "## Most Relevant Internal Links for this question:\n"
                instr_header = "## Instructions\n"
                if links_header in sys_content and instr_header in sys_content:
                    before = sys_content.split(links_header)[0]
                    after = sys_content.split(instr_header, 1)[1]
                    messages[0]["content"] = (
                        before
                        + f"Most relevant link already scraped: {target_url}\n\n"
                        + "## Instructions\n" + after
                    )

            else:
                # Final answer
                answer = response.content
                memory.add(request.question, answer)
                return ChatResponse(answer=answer, source_url=last_scraped_url)

        fallback = "I needed too many steps to find the answer. Please ask a more specific question."
        memory.add(request.question, fallback)
        return ChatResponse(answer=fallback, source_url=last_scraped_url)

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