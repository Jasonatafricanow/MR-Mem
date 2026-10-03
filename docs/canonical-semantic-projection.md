# Canonical SemanticBlock projection contract

`mr_mem.memory.semantic_projection` is the public consumer boundary. Construct
`SqliteCanonicalSemanticBlockReader(canonical_database_path)` for a separate
query-only connection; close it when finished. It provides immutable Block and
relation views, with no admission or lifecycle mutation capability.

Occurrence and knowledge time come directly from `CommittedMemory`. SourceRefs
are native provenance metadata; consumers must not reopen native transcripts or
revalidate source drift. Canonical relations come only from `semantic_relations`.
Their knowledge time is the originating committed Block's knowledge time.
Supersession may use that relation authority as transition time. Other lifecycle
transitions without a timestamp expose `None` (UNKNOWN).

Native pre-Delta admissions remain readable with `schema_version=None`; their
compiler version is the canonical provenance version. Legacy factual memories
without native occurrence authority return `None`. No point IDs, point objects,
dependency objects or cohabit internals appear in the public view. MR-Mem owns
canonical cognition; consumers own rebuildable derived structure only.
