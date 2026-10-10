"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree
from urllib.parse import quote

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                try:
                    delay = float(exc.retry_after)
                except (TypeError, ValueError):
                    delay = 0.0
            else:
                exponential = max(0.0, float(base)) * (2 ** attempt)
                # Additive jitter avoids synchronised retries while retaining true
                # exponential backoff. The cap applies to the final delay.
                delay = exponential + random.uniform(0.0, max(0.0, float(base)))
            time.sleep(max(0.0, min(float(cap), delay)))


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_after(response):
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _request(method, url, **kwargs):
    """Make one HTTP request and translate transient failures for with_retry."""
    try:
        response = httpx.request(method, url, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in _RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
    response.raise_for_status()
    return response


def _clean(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit] if limit else text


def _local_compact():
    """Use smaller payloads for low-context local models without changing normal provider behaviour."""
    return (os.getenv("LAB_LOCAL_COMPAT") or "").strip().lower() in {"1", "true", "yes"}


def _error_text(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        # httpx exception strings can contain the complete query-string URL.
        message = message.replace(secret, "[REDACTED]").replace(quote(secret, safe=""), "[REDACTED]")
    return "ERROR: " + message[:1000]


_arxiv_lock = threading.Lock()
_arxiv_last_call = 0.0
_local_evidence_lock = threading.Lock()
_local_evidence = []


def reset_local_evidence():
    """Clear the in-process evidence cache used only by the local-model fallback."""
    with _local_evidence_lock:
        _local_evidence.clear()


def get_local_evidence():
    """Return a copy of compact records actually retrieved by source tools."""
    with _local_evidence_lock:
        return [dict(item) for item in _local_evidence]


def _record_local(records, source):
    if not _local_compact():
        return
    with _local_evidence_lock:
        known = {item.get("url") for item in _local_evidence}
        for record in records:
            url = _clean(record.get("url"))
            if not url or url in known:
                continue
            _local_evidence.append({
                "id": _clean(record.get("id")) or url,
                "url": url,
                "title": _clean(record.get("title")) or url,
                "date": _clean(record.get("published") or record.get("date"))[:10] or "n.d.",
                "source": source,
                "evidence": _clean(record.get("summary") or record.get("evidence"), 700),
            })
            known.add(url)


def _record_web_text(text):
    records = []
    pattern = re.compile(
        r"(?m)^Title:\s*(?P<title>.+?)\s*$\n^URL:\s*(?P<url>https?://\S+)\s*$"
        r"(?:\n^Published:\s*(?P<published>.*?)\s*$)?",
    )
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        records.append({
            "id": match.group("url"),
            "url": match.group("url").rstrip(".,)"),
            "title": match.group("title"),
            "published": (match.group("published") or "n.d.")[:10],
            "summary": _clean(block, 700),
        })
    _record_local(records, "web")


def _throttle_arxiv():
    global _arxiv_last_call
    with _arxiv_lock:
        remaining = 3.0 - (time.monotonic() - _arxiv_last_call)
        if remaining > 0:
            time.sleep(remaining)
        _arxiv_last_call = time.monotonic()


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    try:
        terms = [term for term in re.findall(r"[\w-]+", str(query), flags=re.UNICODE)
                 if term.lower() not in {"and", "or", "not"}]
        if not terms:
            return "NO RESULTS"
        limit = max(1, min(3 if _local_compact() else 30, int(max_results)))

        def call():
            _throttle_arxiv()
            return _request(
                "GET",
                ARXIV_URL,
                params={
                    "search_query": " AND ".join(f"all:{term}" for term in terms),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": limit,
                    "start": 0,
                },
                timeout=30,
                follow_redirects=True,
            )

        response = with_retry(call, attempts=7, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.content)
        atom = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", atom):
            raw_url = _clean(entry.findtext("atom:id", default="", namespaces=atom))
            raw_id = raw_url.rsplit("/abs/", 1)[-1].rstrip("/")
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _clean(entry.findtext("atom:published", default="", namespaces=atom))[:10],
                "title": _clean(entry.findtext("atom:title", default="", namespaces=atom)),
                "summary": _clean(entry.findtext("atom:summary", default="", namespaces=atom),
                                  360 if _local_compact() else 600),
            })
        _record_local(records, "arxiv")
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools must report failures to the agent, never raise
        return _error_text(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(100, int(limit)))}
        if str(date).strip():
            params["date"] = str(date).strip()
        response = with_retry(
            lambda: _request("GET", HF_DAILY_URL, params=params, timeout=30, follow_redirects=True)
        )
        records = _hf_records(response.json(), prefer_ai_summary=False)
        needle = _clean(keyword).casefold()
        if needle:
            records = [record for record in records
                       if needle in (record["title"] + " " + record["summary"]).casefold()]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        if _local_compact():
            records = records[:3]
        _record_local(records, "hf-daily")
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not _clean(query):
            return "NO RESULTS"
        params = {"q": str(query), "limit": max(1, min(3 if _local_compact() else 50, int(limit)))}
        response = with_retry(
            lambda: _request("GET", HF_SEARCH_URL, params=params, timeout=30, follow_redirects=True)
        )
        records = _hf_records(response.json(), prefer_ai_summary=True)
        _record_local(records, "hf-search")
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


def _as_number(value, default=0):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _hf_records(payload, *, prefer_ai_summary):
    if not isinstance(payload, list):
        raise ValueError("Hugging Face response is not a list")
    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
        paper_id = _clean(paper.get("id"))
        if not paper_id:
            continue
        summary = (paper.get("ai_summary") if prefer_ai_summary else None) or paper.get("summary") \
            or item.get("summary") or ""
        published = paper.get("publishedAt") or item.get("publishedAt") or paper.get("published") or ""
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": _clean(published)[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(summary, 360 if _local_compact() else 600),
            "upvotes": _as_number(paper.get("upvotes", item.get("upvotes"))),
            "github": _clean(paper.get("githubRepo") or item.get("githubRepo")),
            "stars": _as_number(paper.get("githubStars", item.get("githubStars"))),
        })
    return records


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        query = _clean(query)
        if not query:
            return "NO RESULTS"
        arguments = {
            "query": query,
            "objective": _clean(objective) or f"Find reliable, relevant sources about {query}",
            "numResults": max(1, min(3 if _local_compact() else 10, int(num_results))),
        }
        text = _exa_call("web_search_exa", arguments)
        if not text.strip():
            return "NO RESULTS"
        _record_web_text(text)
        return text[:5000] if _local_compact() else text
    except Exception as exc:
        return _error_text(exc, (os.getenv("EXA_API_KEY") or "").strip())


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    try:
        url = _clean(url)
        if not url:
            return "NO RESULTS"
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        return text[:6000 if _local_compact() else 12000] if text.strip() else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc, (os.getenv("EXA_API_KEY") or "").strip())


def _parse_mcp_response(response):
    """Return the JSON-RPC object from either JSON or an SSE response."""
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        return response.json()
    objects = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data and data != "[DONE]":
            objects.append(json.loads(data))
    if not objects:
        raise ValueError("Exa returned an empty event stream")
    # A tools/call response is one event today; taking the last JSON-RPC
    # object also handles servers that emit preliminary notifications.
    return objects[-1]


def _looks_rate_limited(meta, text):
    marker = json.dumps(meta, ensure_ascii=False).casefold() if meta else ""
    body = text.casefold()
    phrases = ("rate limit", "rate_limit", "ratelimit", "too many requests", "quota exceeded")
    return any(phrase in marker or phrase in body for phrase in phrases)


def _exa_call(name, arguments):
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    params = {"exaApiKey": api_key} if api_key else None
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}

    def call():
        response = _request(
            "POST",
            EXA_URL,
            params=params,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            json=payload,
            timeout=60,
            follow_redirects=True,
        )
        data = _parse_mcp_response(response)
        if data.get("error"):
            raise RuntimeError(f"Exa JSON-RPC error: {data['error']}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise ValueError("Exa response has no result object")
        content = result.get("content") or []
        text = "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if _looks_rate_limited(result.get("_meta"), text):
            raise RetryableError("Exa rate limited the request", 20.0)
        if result.get("isError"):
            raise RuntimeError(text or "Exa tool returned an error")
        return text

    return with_retry(call, attempts=7, base=2.0, cap=60.0)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
