# From the Heart Game Bridge

This built-in passive plugin serves the `NEKO-FromTheHeart-DEMO` game. It is
deliberately separate from `galgame_plugin`: the game explicitly calls bounded
entries, and the plugin never observes, OCRs, clicks, or controls an external
game window.

## Entries

- `resolve_interaction` validates a versioned node contract, optionally asks the
  configured `conversation` model for a one-shot intent and YUI line, then applies
  deterministic policy before returning dialogue and visual tags.
- `ensure_cg` accepts only a generation key previously issued by
  `resolve_interaction`. It queries the central cache, mirrors a ready WebP into
  the local plugin cache, and never accepts raw player text.

Neither entry can return story jumps, relationship changes, clues, endings, or
game-variable mutations. Raw player text is not written to plugin logs or CG
recipes.

## Persistent data

The plugin registers `data_path("static_ui")` as its writable static root. Only
content-addressed assets are exposed there. Generation recipes, metadata, and the
LRU index remain private under `data_path("cg_cache")`. Future generated PNGs
are served below:

```text
/plugin/from_the_heart/ui/cg/<prefix>/<asset_id>.webp
```

Runs are transport records only and are not used as persistent CG storage.

## Configuration

Environment variables provide hard gates and budgets:

- `NEKO_FROM_THE_HEART_DIALOGUE_ENABLED` (default `true`)
- `NEKO_FROM_THE_HEART_DYNAMIC_CG_ENABLED` (default `false`)
- `NEKO_FROM_THE_HEART_AI_TIMEOUT_SECONDS` (default `2.5`)
- `NEKO_FROM_THE_HEART_CG_CACHE_MAX_BYTES` (default `1073741824`)
- `NEKO_FROM_THE_HEART_FAILED_GENERATION_TTL_SECONDS` (default `600`)
- `NEKO_FROM_THE_HEART_CG_GENERATION_TIMEOUT_SECONDS` (default `120`)
- `NEKO_FROM_THE_HEART_CENTRAL_CG_URL` (HTTPS, or loopback HTTP for development)
- `NEKO_FROM_THE_HEART_CENTRAL_CG_TOKEN` (client/query authority only)

No model name or credential is hardcoded. Dialogue uses the configured
`conversation` tier and fails closed to the node's handwritten response.

## Central CG service

The deployable FastAPI app lives at
`plugin.plugins.from_the_heart.central.app:app`. It owns canonical recipe
normalization, the unique `generation_key`, database states, worker leases, and
content-addressed WebP objects. The local plugin submits only an allowlisted
`visual_variant_key`; the central service rebuilds the official recipe and
rejects a client hash mismatch.

Development launch:

```powershell
$env:FROM_THE_HEART_CENTRAL_CLIENT_TOKEN = "replace-me"
$env:FROM_THE_HEART_CENTRAL_WORKER_TOKEN = "replace-worker-secret"
uv run uvicorn plugin.plugins.from_the_heart.central.app:app --host 127.0.0.1 --port 48918
```

The bundled reference backend uses SQLite plus atomic filesystem object
storage. All database and file operations are offloaded from the event loop.
Production can replace those adapters with PostgreSQL and S3-compatible object
storage without changing the game or plugin protocol.

Only a holder of the worker token may claim a recipe or upload its final
1920x1080 WebP. No image model is connected in this implementation; queued jobs
remain harmless until a trusted worker is deployed. Dynamic CG stays disabled
by default.

## Tests

The central repository suite covers contract handshakes, exact-answer routing,
LLM output policy, prompt-injection fallback, CG key issuance, content hashing,
LRU eviction, and the disabled generation gate:

```powershell
uv run pytest plugin/tests/unit/plugins/test_from_the_heart.py -q
uv run python -m plugin.neko_plugin_cli.cli check plugin/plugins/from_the_heart
```
