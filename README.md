# MR-Mem

Standalone memory runtime extracted from Mind Runtime.

MR-Mem owns a single canonical Memory substrate plus local derived product
projections. Canonical Memory remains factual authority; Thread/attention/LCE
integration are projections over that authority rather than second truth stores.

Initial extraction boundary:

- structured Scope + sync identity
- canonical Memory contracts
- SQLite canonical Memory store
- rebuildable projection outbox
- product state (attention + bounded Thread)
- runtime-independent `MemoryCore`

LCE, retrieval providers, admission, benchmark adapters, and Mind Runtime host
composition remain outside this first core extraction and will attach through
explicit interfaces.
