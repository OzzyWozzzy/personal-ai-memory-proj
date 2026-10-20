# Local MCP server

This zero-dependency stdio bridge gives an MCP-capable agent access to the local
temporal memory API. It speaks JSON-RPC 2.0 over stdin/stdout and never exposes
an HTTP listener.

Create an expiring agent key from the local UI, then configure your MCP client:

```json
{
  "mcpServers": {
    "temporal-memory": {
      "command": "/absolute/path/to/.venv/bin/temporal-memory-mcp",
      "env": {
        "MEMORY_API_KEY": "mem_agent_...",
        "MEMORY_API_URL": "http://127.0.0.1:8321"
      }
    }
  }
}
```

`MEMORY_API_URL` defaults to `http://127.0.0.1:8321`. Cleartext HTTP is accepted
only for loopback hosts; remote backends must use HTTPS.

| Tool | Purpose |
|---|---|
| `remember {text, when?}` | Store one source episode and extract graph facts |
| `search_memory {query}` | Search entities, edges, and hydrated passages |
| `get_context {query, max_words?}` | Build token-bounded context for an agent |
| `get_story {name}` | Read an entity's chronological temporal history |

Errors are returned as MCP tool results rather than terminating the server.
