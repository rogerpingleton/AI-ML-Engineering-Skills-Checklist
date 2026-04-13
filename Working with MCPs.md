# Model Context Protocol (MCP): A Comprehensive Guide for AI Engineers

> **Scope:** Agentic systems architecture, MCP internals, Python implementation, spec history, and security. **Last updated against spec:** `2025-11-25` (latest stable as of April 2026)

---

## Table of Contents

1. [What Is MCP and Why Does It Exist?](#1-what-is-mcp-and-why-does-it-exist)
2. [Architecture and Core Concepts](#2-architecture-and-core-concepts)
3. [Transport Layer](#3-transport-layer)
4. [The Three Primitives: Tools, Resources, and Prompts](#4-the-three-primitives-tools-resources-and-prompts)
5. [The MCP Specification Timeline](#5-the-mcp-specification-timeline)
6. [Building an MCP Server in Python (FastMCP)](#6-building-an-mcp-server-in-python-fastmcp)
7. [Wrapping an Existing Service in an MCP Layer](#7-wrapping-an-existing-service-in-an-mcp-layer)
8. [Advanced Topics](#8-advanced-topics)
9. [Security Considerations](#9-security-considerations)
10. [Ecosystem and Governance](#10-ecosystem-and-governance)
11. [Quick-Reference Cheat Sheet](#11-quick-reference-cheat-sheet)

---

## 1. What Is MCP and Why Does It Exist?

Before MCP, every integration between an LLM application and an external service required a custom connector. Ten LLM apps connecting to ten data sources created a hundred bespoke integrations — the classic **N×M** problem. OpenAI's function-calling API (2023) and the ChatGPT plugin framework reduced this burden but remained vendor-specific.

Anthropic open-sourced the **Model Context Protocol** in November 2024 as a universal, vendor-neutral standard. Its origin story is famously pragmatic: it grew from engineer David Soria Parra's frustration with constantly copying code between Claude Desktop and his IDE. The protocol is modeled closely on the **Language Server Protocol (LSP)** — the same idea that unified IDE ↔ language-server communication — applied to the LLM ↔ tool integration problem.

In essence:

> **MCP is to LLM tool integration what USB-C is to device cables.**

Any MCP client can speak to any MCP server. The protocol is transported over **JSON-RPC 2.0** and is entirely model-agnostic.

---

## 2. Architecture and Core Concepts

MCP uses a three-tier client-server architecture:

```
┌─────────────────────────────────────────────────────────┐
│  HOST (e.g., Claude Desktop, Cursor, ChatGPT desktop)   │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐                   │
│  │  MCP Client  │   │  MCP Client  │  (one per server) │
│  └──────┬───────┘   └──────┬───────┘                   │
└─────────┼─────────────────┼───────────────────────────┘
          │ Transport        │ Transport
          ▼                  ▼
   ┌────────────┐    ┌────────────┐
   │ MCP Server │    │ MCP Server │
   │ (local/    │    │ (remote/   │
   │  stdio)    │    │  HTTP)     │
   └────────────┘    └────────────┘
```

**Key roles:**

- **Host**: The user-facing application (Claude Desktop, Cursor, a custom agent). It manages one or more `MCP Client`instances.
- **MCP Client**: Maintains a 1:1 connection to a single MCP server. Handles capability negotiation and message routing.
- **MCP Server**: A lightweight process (or HTTP service) that exposes data and functionality via the three primitives. Each server has a focused domain (e.g., filesystem access, a database, a SaaS API).

**Connection lifecycle:**

1. Client sends `initialize` request with its capabilities and protocol version.
2. Server responds with `InitializeResult` (its capabilities and chosen spec version).
3. Client sends `initialized` notification.
4. Normal operation: tools/list, tools/call, resources/read, prompts/get, etc.
5. Either side can send a `close` notification.

---

## 3. Transport Layer

MCP is transport-agnostic. The spec currently defines two official transports:

### 3.1 stdio (Standard I/O)

The host spawns the MCP server as a **subprocess**. The client writes newline-delimited JSON-RPC to the server's `stdin`; the server responds on `stdout`. Stderr is reserved for logs and is never parsed as protocol messages.

**When to use:**

- Local-only tools (filesystem access, local databases, CLI wrappers)
- Maximum simplicity; zero network stack
- Development and prototyping

```python
# server.py — stdio transport
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tool")

@mcp.tool()
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="stdio")   # default
```

The host's `claude_desktop_config.json` entry looks like:

```json
{
  "mcpServers": {
    "my-tool": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### 3.2 Streamable HTTP (current remote standard)

Introduced in spec `2025-03-26` to replace the deprecated **HTTP+SSE** transport. The server exposes a **single endpoint** (e.g., `https://example.com/mcp`) that accepts both `POST` and `GET`.

- Client sends JSON-RPC via HTTP `POST`.
- Server responds with a `200 OK` (single JSON response) or upgrades the response to a **Server-Sent Events stream**when it needs to push multiple messages back.
- Session state is tracked via an `MCP-Session-Id` header (a cryptographically random UUID or JWT).

**Why the move away from HTTP+SSE?**

The original `2024-11-05` spec required two separate HTTP connections — one persistent SSE stream (server→client) and separate POSTs (client→server). This was complex to proxy, difficult to secure, and fought with corporate firewalls. Streamable HTTP collapses this to a single endpoint and standard HTTP semantics, making it compatible with WAFs, load balancers, and OAuth middleware out of the box.

```python
# server.py — Streamable HTTP transport
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-remote-tool")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

The server is then reachable at `http://localhost:8000/mcp`.

### Transport Selection Guide

|Scenario|Recommended Transport|
|---|---|
|Local CLI tool or script|stdio|
|Desktop app plugin|stdio|
|Multi-user remote service|Streamable HTTP + OAuth 2.1|
|Microservice behind an API gateway|Streamable HTTP|
|Legacy client that only knows SSE|HTTP+SSE (backwards compat only)|

---

## 4. The Three Primitives: Tools, Resources, and Prompts

Everything an MCP server exposes falls into one of three categories.

### 4.1 Tools

The most important primitive. Tools are the **action layer** — they are callable by the LLM to execute code, query databases, call APIs, mutate state. Think of them as `POST` endpoints.

**The model can call tools autonomously** during inference when the host has granted permission.

```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("calculator")

class DivisionResult(BaseModel):
    quotient: float
    remainder: float

@mcp.tool()
def divide(dividend: float, divisor: float) -> DivisionResult:
    """
    Divide two numbers and return quotient and remainder.
    Raises ValueError if divisor is zero.
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero")
    return DivisionResult(
        quotient=dividend // divisor,
        remainder=dividend % divisor
    )
```

**Tool annotations** (introduced `2025-03-26`) allow servers to declare behavioral metadata:

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool

# Via decorator kwargs (FastMCP)
@mcp.tool(
    annotations={
        "readOnlyHint": True,       # Does not mutate state
        "idempotentHint": True,     # Safe to retry
        "openWorldHint": False      # No external network calls
    }
)
def get_user(user_id: str) -> dict:
    """Fetch a user record from the local database."""
    ...
```

Annotation hints allow hosts to present smarter confirmation dialogs to users before executing potentially destructive actions.

**Structured tool output** (introduced `2025-06-18`) lets tools return both human-readable content and machine-readable structured data:

```python
@mcp.tool(output_schema=DivisionResult)
def divide_structured(dividend: float, divisor: float):
    """Returns structured output validated against a schema."""
    result = DivisionResult(quotient=dividend // divisor, remainder=dividend % divisor)
    # Return tuple: (human-readable content, structured data)
    return f"{dividend} ÷ {divisor} = {result.quotient} rem {result.remainder}", result
```

### 4.2 Resources

Resources are the **read-only data layer** — content loaded into the LLM's context window. Think of them as `GET`endpoints. The model does not automatically fetch resources; the host or user typically initiates resource reads.

Resources are identified by URIs (e.g., `file:///path/to/doc`, `db://customers/42`, `greeting://alice`).

```python
import json
from pathlib import Path

@mcp.resource("file://config/{config_name}")
def read_config(config_name: str) -> str:
    """Read a configuration file by name."""
    config_path = Path("configs") / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config '{config_name}' not found")
    return config_path.read_text()

@mcp.resource("db://customers/{customer_id}")
def get_customer_record(customer_id: str) -> str:
    """Fetch a customer record (returns JSON string)."""
    # In practice: query your database
    record = {"id": customer_id, "name": "Acme Corp", "plan": "enterprise"}
    return json.dumps(record, indent=2)
```

### 4.3 Prompts

Prompts are **reusable conversation templates** — parameterized instruction stubs that can be injected into conversations. They are invoked by users or hosts (not autonomously by the model).

```python
@mcp.prompt()
def code_review_prompt(
    language: str,
    focus: str = "general",
    severity: str = "all"
) -> str:
    """
    Generate a code review instruction prompt.
    
    Args:
        language: Programming language being reviewed
        focus: Area of focus (security, performance, readability, general)
        severity: Minimum severity to report (critical, high, medium, all)
    """
    return f"""
You are an expert {language} code reviewer. 
Focus area: {focus}
Report issues with severity: {severity} and above.

Review the provided code for:
- Correctness and logic errors
- Security vulnerabilities (if focus includes security)
- Performance bottlenecks (if focus includes performance)  
- Code style and readability
- Missing error handling

Provide actionable, specific feedback with line references where possible.
""".strip()
```

---

## 5. The MCP Specification Timeline

MCP has gone through four formal spec versions in under 18 months. Understanding the changes helps you reason about compatibility and choose the right features.

### `2024-11-05` — Initial Release

Released November 2024 by Anthropic.

- **Transport:** HTTP+SSE (two-connection model: GET for SSE stream, POST for client messages) plus stdio.
- **Primitives:** Tools, Resources, Prompts (all three present from day one).
- **Auth:** No standardized authorization model. Ad hoc implementations only.
- **Notable:** The baseline that kicked off the ecosystem. Python and TypeScript SDKs shipped simultaneously.

### `2025-03-26` — Authorization & Transport Overhaul

Released March 2025, coinciding with OpenAI's official MCP adoption.

**Major changes:**

- **Streamable HTTP transport**: Single endpoint replaces the two-connection SSE approach. SSE is now deprecated.
- **OAuth 2.1 authorization framework**: The first standardized auth model in the spec. Servers can act as OAuth Authorization Servers.
- **Tool annotations**: Behavioral hints (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`).
- **JSON-RPC batching**: Multiple requests in a single call _(later removed)_.
- **Capability negotiation improvements**.

### `2025-06-18` — Structured Outputs & Security Hardening

**Major changes:**

- **Structured tool output**: Tools can return validated JSON alongside human-readable content (`outputSchema`).
- **Enhanced OAuth**: MCP servers now classified as OAuth Resource Servers with additional requirements (audience validation, resource indicators per RFC 8707).
- **Elicitation**: Servers can proactively request structured information from users mid-conversation. This enables multi-step workflows where the server needs to gather input before proceeding.
- **Resource links in tool results**: Tool outputs can reference MCP resources by URI, creating richer integrations.
- **JSON-RPC batching removed**: Dropped for simplicity; the maintainers found no compelling production use case.
- **Improved security best practices** documentation.

### `2025-11-25` — Enterprise Scale & Tasks (Current Stable)

Released on MCP's one-year anniversary.

**Major changes:**

- **Tasks abstraction**: Any request can now be augmented with a `Task`, allowing clients to query status and retrieve results asynchronously. Critical for long-running operations (minutes to hours).
- **OpenID Connect Discovery support** for server identity.
- **Icons metadata** for tools, resources, and prompts (UI polish).
- **Incremental OAuth scope consent**: Request only the scopes you need when you need them, rather than all upfront.
- **URL mode elicitation**: Richer elicitation patterns.
- **Sampling tool calling support**: Servers can request the client/LLM to perform sampling (inference) as part of a tool's execution.
- **OAuth Client ID Metadata Documents** (RFC 7591 dynamic registration).
- **SEP process formalized**: Community-driven Specification Enhancement Proposals now the standard change mechanism.
- **MCP Registry launched**: Official catalog for discovering and indexing MCP servers.

### Spec Version Summary Table

|Version|Transport|Auth|Key Addition|
|---|---|---|---|
|`2024-11-05`|HTTP+SSE, stdio|None|Initial release|
|`2025-03-26`|Streamable HTTP (SSE deprecated), stdio|OAuth 2.1 draft|Tool annotations, streaming HTTP|
|`2025-06-18`|Streamable HTTP, stdio|OAuth 2.1 + Resource Server|Structured outputs, Elicitation|
|`2025-11-25`|Streamable HTTP, stdio|OAuth 2.1 + OIDC + scopes|Tasks, server identity, MCP Registry|

---

## 6. Building an MCP Server in Python (FastMCP)

The official Python SDK ships with **FastMCP**, a decorator-based high-level API inspired by FastAPI. It is the recommended approach for most servers.

### Installation

```bash
# Using uv (recommended)
uv init my-mcp-server
cd my-mcp-server
uv add "mcp[cli]"

# Or pip
pip install "mcp[cli]"
```

### A Complete Example: Weather Service MCP Server

```python
"""
weather_server.py

A complete FastMCP server demonstrating tools, resources, and prompts.
Run with:  python weather_server.py
Test with: mcp dev weather_server.py
"""

import json
import httpx
from mcp.server.fastmcp import FastMCP

# ── Server instantiation ──────────────────────────────────────────────────────
mcp = FastMCP(
    "weather-service",
    description="Provides current weather and forecasts via Open-Meteo API",
)

# ── Helper ────────────────────────────────────────────────────────────────────

async def _fetch_weather(lat: float, lon: float) -> dict:
    """Call the Open-Meteo API (free, no key required)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,weather_code",
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": True}
)
async def get_current_weather(latitude: float, longitude: float) -> str:
    """
    Fetch current weather conditions for a geographic coordinate.

    Args:
        latitude:  Decimal latitude  (e.g., 51.5074 for London)
        longitude: Decimal longitude (e.g., -0.1278 for London)

    Returns:
        A human-readable weather summary.
    """
    data = await _fetch_weather(latitude, longitude)
    current = data["current"]
    temp = current["temperature_2m"]
    wind = current["wind_speed_10m"]
    return (
        f"Current conditions at ({latitude}, {longitude}): "
        f"{temp}°C, wind {wind} km/h"
    )

@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": True}
)
async def get_hourly_forecast(latitude: float, longitude: float) -> str:
    """
    Return a 24-hour temperature forecast for the given coordinates.

    Args:
        latitude:  Decimal latitude
        longitude: Decimal longitude
    """
    data = await _fetch_weather(latitude, longitude)
    hours = data["hourly"]["time"][:24]
    temps = data["hourly"]["temperature_2m"][:24]
    rows = "\n".join(f"  {h}: {t}°C" for h, t in zip(hours, temps))
    return f"24-hour forecast for ({latitude}, {longitude}):\n{rows}"

# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("weather://cache/{lat}/{lon}")
async def cached_weather_data(lat: str, lon: str) -> str:
    """
    Return raw JSON weather data for a location.
    Useful when an agent needs structured data, not a summary.
    """
    data = await _fetch_weather(float(lat), float(lon))
    return json.dumps(data, indent=2)

# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def weather_analysis_prompt(city: str, use_case: str = "general travel") -> str:
    """
    Generate a prompt that instructs the LLM to provide a weather-aware
    recommendation for the given city and use case.

    Args:
        city:     City name (will be used to frame the analysis)
        use_case: Context for the recommendation (e.g., 'outdoor wedding', 'marathon')
    """
    return (
        f"You have access to real-time weather tools. "
        f"The user is planning a {use_case} in {city}. "
        f"First retrieve the current weather and today's forecast, "
        f"then provide a practical recommendation including what to wear, "
        f"any weather-related risks, and the best time of day for the activity."
    )

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # stdio for local use; change to "streamable-http" for remote
    mcp.run(transport="stdio")
```

### Testing with the MCP Inspector

```bash
# Launch the browser-based inspector
mcp dev weather_server.py

# Or run a quick CLI check
mcp run weather_server.py
```

The inspector lets you call tools interactively, browse resources, and invoke prompts without writing a client.

---

## 7. Wrapping an Existing Service in an MCP Layer

The most common real-world task is taking a service that already exists — a REST API, a database, an internal tool — and exposing it through MCP so agents can use it.

### Pattern: REST API Wrapper

This example wraps a generic REST API (using GitHub as the concrete case).

```python
"""
github_mcp.py

Wrap a subset of the GitHub REST API as an MCP server.
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github-wrapper")

BASE_URL = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h

# ── Read-only tools ───────────────────────────────────────────────────────────

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_repo(owner: str, repo: str) -> str:
    """
    Fetch repository metadata from GitHub.

    Args:
        owner: Repository owner (user or org)
        repo:  Repository name
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/repos/{owner}/{repo}", headers=_headers())
        r.raise_for_status()
        data = r.json()
    return (
        f"Repo: {data['full_name']}\n"
        f"Stars: {data['stargazers_count']}\n"
        f"Language: {data['language']}\n"
        f"Description: {data['description']}\n"
        f"Open issues: {data['open_issues_count']}"
    )

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def list_open_issues(owner: str, repo: str, limit: int = 10) -> str:
    """
    List open issues for a GitHub repository.

    Args:
        owner: Repository owner
        repo:  Repository name
        limit: Maximum issues to return (default 10, max 30)
    """
    limit = min(limit, 30)
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{BASE_URL}/repos/{owner}/{repo}/issues",
            headers=_headers(),
            params={"state": "open", "per_page": limit},
        )
        r.raise_for_status()
        issues = r.json()

    if not issues:
        return "No open issues found."
    lines = [f"#{i['number']}: {i['title']} (@{i['user']['login']})" for i in issues]
    return "\n".join(lines)

# ── Write tool (flagged as destructive) ───────────────────────────────────────

@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,   # creating an issue is additive, not destructive
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def create_issue(owner: str, repo: str, title: str, body: str = "") -> str:
    """
    Create a new GitHub issue. Requires GITHUB_TOKEN with repo write scope.

    Args:
        owner: Repository owner
        repo:  Repository name
        title: Issue title
        body:  Issue description (Markdown supported)
    """
    if not TOKEN:
        return "Error: GITHUB_TOKEN environment variable not set."
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/repos/{owner}/{repo}/issues",
            headers=_headers(),
            json={"title": title, "body": body},
        )
        r.raise_for_status()
        issue = r.json()
    return f"Created issue #{issue['number']}: {issue['html_url']}"

# ── Resource: raw issue JSON ──────────────────────────────────────────────────

@mcp.resource("github://issues/{owner}/{repo}/{issue_number}")
async def get_issue_detail(owner: str, repo: str, issue_number: str) -> str:
    """Return full JSON detail for a specific GitHub issue."""
    import json
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}",
            headers=_headers(),
        )
        r.raise_for_status()
    return json.dumps(r.json(), indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Pattern: Database Wrapper

```python
"""
postgres_mcp.py

Safely expose a PostgreSQL database as a read-only MCP server.
"""

import json
import asyncpg
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("postgres-readonly")

DB_DSN = "postgresql://user:password@localhost/mydb"

async def _query(sql: str, *args) -> list[dict]:
    conn = await asyncpg.connect(DB_DSN)
    try:
        rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]
    finally:
        await conn.close()

@mcp.tool(annotations={"readOnlyHint": True})
async def run_select_query(table: str, where_clause: str = "", limit: int = 20) -> str:
    """
    Execute a safe SELECT query on an approved table.

    Args:
        table:        Table name (validated against allowlist)
        where_clause: Optional WHERE conditions (e.g., "status = 'active'")
        limit:        Row limit (default 20, max 100)
    """
    ALLOWED_TABLES = {"customers", "orders", "products", "analytics_summary"}
    if table not in ALLOWED_TABLES:
        return f"Error: Table '{table}' is not in the allowed list: {ALLOWED_TABLES}"
    limit = min(limit, 100)
    sql = f"SELECT * FROM {table}"
    if where_clause:
        # NOTE: In production, use parameterized queries and a proper SQL parser.
        sql += f" WHERE {where_clause}"
    sql += f" LIMIT {limit}"
    rows = await _query(sql)
    return json.dumps(rows, indent=2, default=str)

@mcp.resource("db://schema/{table_name}")
async def get_table_schema(table_name: str) -> str:
    """Return column names and types for a table."""
    rows = await _query(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
        """,
        table_name,
    )
    return json.dumps(rows, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Deployment Pattern: Remote HTTP Server with Auth

For production remote deployments:

```python
"""
production_server.py

Remote MCP server with OAuth 2.1 token verification.
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.auth import OAuthServerProvider  # available in SDK ≥1.5

mcp = FastMCP("production-service")

# OAuth resource server validation
REQUIRED_AUDIENCE = "https://api.example.com/mcp"
JWKS_URI = "https://auth.example.com/.well-known/jwks.json"

@mcp.tool()
async def get_data(query: str) -> str:
    """Retrieve data from the production service."""
    # Tool logic here
    return f"Results for: {query}"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
```

The `mcp[cli]` package also provides a CLI runner for quick deployment:

```bash
# Run as HTTP server on port 9000
mcp run production_server.py --transport streamable-http --port 9000
```

---

## 8. Advanced Topics

### 8.1 Context Object

FastMCP tools can request a `Context` object as a parameter to access MCP lifecycle features:

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("advanced-server")

@mcp.tool()
async def long_running_task(input_data: str, ctx: Context) -> str:
    """Demonstrates progress reporting and logging."""

    await ctx.info(f"Starting processing of: {input_data[:50]}...")
    await ctx.report_progress(0, 100, "Initializing")

    # Do work in stages
    for i in range(1, 6):
        result = await do_work_chunk(input_data, i)
        await ctx.report_progress(i * 20, 100, f"Stage {i}/5 complete")
        await ctx.debug(f"Stage {i} produced: {result}")

    await ctx.info("Processing complete")
    return "Done"

async def do_work_chunk(data: str, stage: int) -> str:
    # Simulate work
    return f"chunk_{stage}"
```

### 8.2 Elicitation (spec `2025-06-18+`)

Elicitation allows a server to request structured input from the user mid-execution:

```python
@mcp.tool()
async def create_deployment(service: str, ctx: Context) -> str:
    """Deploy a service to production — asks for confirmation first."""

    # Request confirmation before destructive action
    response = await ctx.elicit(
        message=f"Deploy '{service}' to production?",
        schema={
            "type": "object",
            "properties": {
                "confirmed": {"type": "boolean", "description": "Confirm deployment"},
                "notes": {"type": "string", "description": "Optional deployment notes"},
            },
            "required": ["confirmed"],
        }
    )

    if not response.get("confirmed"):
        return "Deployment cancelled by user."

    notes = response.get("notes", "")
    # Proceed with deployment...
    return f"Deployed {service} to production. Notes: {notes}"
```

### 8.3 Sampling (Server-Initiated Inference)

Since spec `2025-11-25`, servers can request the LLM to perform inference as part of their own execution:

```python
@mcp.tool()
async def summarize_document(document_uri: str, ctx: Context) -> str:
    """Fetch a document and have the LLM summarize it."""

    # Read the document
    content = await fetch_document(document_uri)

    # Ask the connected LLM to summarize it
    summary_response = await ctx.sample(
        messages=[{"role": "user", "content": f"Summarize this:\n\n{content}"}],
        max_tokens=500,
    )
    return summary_response.content

async def fetch_document(uri: str) -> str:
    # Fetch implementation
    return "Document content..."
```

### 8.4 Low-Level Server (when you need full control)

FastMCP covers 95% of use cases. For the rest, the low-level `Server` class gives direct access to the JSON-RPC layer:

```python
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server("low-level-example")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="raw_tool",
            description="A tool using the low-level API",
            inputSchema={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "raw_tool":
        return [types.TextContent(type="text", text=f"Got x={arguments.get('x')}")]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 9. Security Considerations

MCP's rapid growth outpaced its security design. The April 2025 security analysis by independent researchers documented multiple outstanding issues. As of `2025-11-25` the spec has improved significantly, but implementation-side vigilance remains essential.

### 9.1 Threat Categories

**Prompt Injection via Tool Descriptions**

Tool descriptions are sent directly to the LLM as part of the system context. A malicious MCP server (or a compromised one) can embed hidden instructions in its tool descriptions that the model follows without the user's knowledge. Never connect to MCP servers you haven't audited.

**Token Storage Risk**

MCP servers often store OAuth tokens for multiple downstream services. A single server compromise can cascade into access to Gmail, Drive, CRM, and more. Use short-lived tokens, token binding (DPoP, RFC 9449), and scope minimization.

**Tool Chaining / "Toxic Agent" Flows**

Individual tools may appear safe in isolation, but an agent can chain them in ways that exfiltrate data. For example: `read_file` → `web_search` → `send_email`. Each tool alone is innocuous; the chain is not.

**Lookalike Tool Substitution**

A malicious server can register a tool with the same name as a trusted one, silently replacing it. Always validate tool provenance.

**Rug Pull / Tool Redefinition**

MCP allows servers to change tool definitions mid-session via `notifications/tools/list_changed`. A compromised server could redefine a tool's behavior after initial trust was granted.

### 9.2 Mitigations

```
Defense Strategy              Implementation
───────────────────────────── ──────────────────────────────────────────────
Least-privilege OAuth scopes  Request only what's needed; use incremental
                              consent (spec 2025-11-25 SEP-835)

Input validation              Sanitize all tool inputs; never pass raw user
                              input to shell commands or SQL without validation

Output validation             Use structured tool outputs (2025-06-18) with
                              schemas to constrain what tools can return

Human-in-the-loop (HITL)      Require confirmation before destructive tools;
                              use elicitation for multi-step approval flows

Tool annotations              Honor readOnlyHint / destructiveHint to drive
                              UI-level confirmation dialogs

Rate limiting                 Apply at the gateway layer (not just the server)

Audit logging                 Log every tool invocation with user identity,
                              timestamp, inputs, and outputs

Use MCP Registry              Only connect to servers listed in the official
                              registry or your organization's private registry

Avoid public exposure         Prefer stdio for local tools; only expose HTTP
                              servers internally unless public access is needed
```

### 9.3 Authentication Reference

```python
# Minimal OAuth token verification middleware (conceptual)
import jwt
from functools import wraps

def require_bearer_token(f):
    """Decorator to verify JWT bearer tokens on Streamable HTTP endpoints."""
    @wraps(f)
    async def wrapper(request, *args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return {"error": "Missing bearer token"}, 401
        token = auth.removeprefix("Bearer ")
        try:
            payload = jwt.decode(
                token,
                key=get_public_key(),          # fetch from JWKS URI
                algorithms=["RS256"],
                audience="https://api.example.com/mcp",
            )
            request.state.user = payload["sub"]
        except jwt.InvalidTokenError as e:
            return {"error": f"Invalid token: {e}"}, 401
        return await f(request, *args, **kwargs)
    return wrapper
```

---

## 10. Ecosystem and Governance

**Adoption (as of early 2026):**

- 97M+ monthly SDK downloads
- 10,000+ active MCP servers in the wild
- First-class client support in Claude, ChatGPT, Cursor, Gemini, Microsoft Copilot, VS Code, and more

**Governance:**

In December 2025, Anthropic donated MCP to the **Agentic AI Foundation (AAIF)**, a directed fund under the **Linux Foundation**, co-founded with Block and OpenAI and supported by AWS, Google, Microsoft, and Cloudflare. MCP is now vendor-neutral infrastructure.

**Specification Enhancement Proposals (SEPs):**

Changes to the spec now go through a formal SEP process with public GitHub discussion. If you encounter a production pain point, file a SEP rather than building a proprietary workaround.

**MCP Registry:**

The official registry (`registry.modelcontextprotocol.io`) is a community-driven catalog of MCP servers. Treat it like npm or PyPI — it provides discovery, version information, and (growing) security metadata. Organizations can operate private sub-registries for internal servers.

**SDKs (official):**

|Language|Package|Maturity|
|---|---|---|
|Python|`mcp` (PyPI)|Stable, v1.27+|
|TypeScript|`@modelcontextprotocol/sdk`|Stable|
|Java|`io.modelcontextprotocol.sdk`|GA|
|Kotlin|`io.modelcontextprotocol:kotlin-sdk`|GA|
|C#|`ModelContextProtocol`|GA|
|Go|`github.com/modelcontextprotocol/go-sdk`|GA|
|Rust|`rmcp` (community)|Beta|

---

## 11. Quick-Reference Cheat Sheet

```
INSTALL
  pip install "mcp[cli]"
  uv add "mcp[cli]"

CREATE A SERVER
  from mcp.server.fastmcp import FastMCP
  mcp = FastMCP("name")

TOOL           @mcp.tool()              # callable by LLM; like POST
RESOURCE       @mcp.resource("uri://…") # read-only context; like GET
PROMPT         @mcp.prompt()            # reusable instruction template

RUN
  mcp.run()                            # stdio (default)
  mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

TEST
  mcp dev server.py                   # browser inspector
  mcp run server.py                   # CLI

TOOL ANNOTATIONS (2025-03-26+)
  readOnlyHint      – does not modify state
  destructiveHint   – may delete/overwrite
  idempotentHint    – safe to retry
  openWorldHint     – makes external network calls

STRUCTURED OUTPUT (2025-06-18+)
  @mcp.tool(output_schema=MyPydanticModel)
  def my_tool(...) -> tuple[str, MyPydanticModel]:
      return "human text", MyPydanticModel(...)

CONTEXT OBJECT (advanced)
  async def my_tool(x: str, ctx: Context) -> str:
      await ctx.info("log message")
      await ctx.report_progress(50, 100, "halfway")
      response = await ctx.elicit(message="Confirm?", schema={…})

TRANSPORT CHOICE
  Local / Desktop   → stdio
  Remote / Cloud    → streamable-http + OAuth 2.1

SPEC VERSIONS
  2024-11-05  Initial release (HTTP+SSE)
  2025-03-26  Streamable HTTP, OAuth 2.1, tool annotations
  2025-06-18  Structured outputs, elicitation, enhanced OAuth
  2025-11-25  Tasks, OIDC, scopes, MCP Registry (CURRENT STABLE)
```

---

_Report generated April 2026. Spec reference: `2025-11-25`. For the latest, see [modelcontextprotocol.io](https://modelcontextprotocol.io/) and [blog.modelcontextprotocol.io](https://blog.modelcontextprotocol.io/)._