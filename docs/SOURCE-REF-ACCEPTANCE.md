# Issue #3 acceptance

Base: ADR-0001 at `fbf96dd`. The implementation keeps CommittedMemory naming.

- Native host raw bodies remain exclusively in the host. Admission accepts
  semantic content/attributes and verifies scoped SourceRefs without a factual pair.
- Restart replays return the same immutable record and persisted known_at.
- Source edit/delete uses an indexed support lookup; unrelated records survive.
- Supersession and support/contradiction links require same-scope canonical IDs.
- Existing factual-pair admission remains a compatibility adapter. Old payloads
  decode with empty native/semantic fields without changing historical IDs.
- Thread and retrieval operate over native-backed semantics and reject stale
  support. Fragments/revisions of one native event are not independent support.

Executable evidence is in `tests/test_native_semantics.py` and the existing
admission/store/retrieval/Thread suite. These checks validate the core boundary,
not real Hermes semantics, LCE algorithm conformance, three-lane fusion, or AML.
Those remain the ordered downstream work described by ADR-0001.
