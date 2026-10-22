# Runtime assembly with an unfilled compiler port

[中文](ASSEMBLY.zh-CN.md)

`examples/assembly-no-compiler.json` explicitly selects `compiler.type=disabled`.
It assembles the existing source store, canonical store, raw retrieval, API owner
thread and restart path. It does not select a model, install a service or generate
placeholder semantic Blocks. No external request is made by the disabled backend.

The existing AML endpoint owns raw input for this standalone ingestion mode.
Hosts with durable native history must keep using native SourceRefs; do not mirror
their complete transcripts into this endpoint merely to connect the modules.

`GET /health` remains a liveness endpoint. It reports `status=degraded`,
`compiler_configured=false`, and `semantic_validation=not_checked` while the port
is unfilled. `/search` can retrieve existing canonical data, or raw data with
`raw_base=true`. `/add` preserves raw input but returns retryable HTTP 503 before
semantic completion. Repeating the request does not duplicate raw records. This
also applies to tree/window input; an unavailable compiler cannot mark it done.
`/admin/flush` refuses semantic work while the compiler is disabled. HyDE cannot
be enabled with a disabled compiler because it needs a model backend.

Once an implementation is ready, configure the existing `command` / `openai`
backend or inject `MemoryRuntime(config, compiler=implementation)`. Close the old
single-writer runtime and reopen the same data directory. Retry the same request
ID and exact payload; in window mode flush the remaining window. Original source
time and identities survive. The accepted compiler output then follows existing
admission, projection and Thread/Path A stages. Successful configuration alone
does not prove semantic fidelity; health always distinguishes that unmeasured fact.

Tests in `test_aml_assembly.py` use an explicitly accepted fixture compiler. They
verify pending input, recovery, exactly-once application and scope isolation, not
semantic inference. Existing runtime/window tests cover the normal configured path.
The original development overlay is preserved separately from this change; the
compiler-reliability research candidate was not promoted into production.
