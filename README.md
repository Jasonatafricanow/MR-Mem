# MR-Mem

Standalone memory runtime extracted from Mind Runtime.

MR-Mem owns canonical Memory authority, Memory admission governance, and local
derived product projections. Canonical Memory remains factual authority;
Thread/attention/LCE integration are projections over that authority rather
than second truth stores.

Current standalone boundary:

- structured Scope + sync identity
- provider-neutral source/admission contracts
- durable-pair admission governance
- deterministic bounded extraction
- canonical Memory contracts
- SQLite canonical Memory store
- rebuildable projection outbox
- product state (attention + bounded Thread)
- runtime-independent `MemoryCore`

The admission layer no longer depends on Mind Runtime's concrete FactBackend.
Any host can provide `SourceEvidence`, `SourceObservation`, a read-only
`DurableFactReader`, and a `Clock`. MR's existing Evidence/Observation plane
can satisfy that boundary through an adapter; another host can use a different
fact store without changing MR-Mem.

Retrieval providers, Thread auto-update policy, LCE, benchmark adapters, and
Mind Runtime host composition are separate extraction/integration work.

## Quick start

```python
from pathlib import Path

from mr_mem import MemoryCore

with MemoryCore(Path("memory.sqlite"), read_only=True) as memory:
    rows = memory.load_all()
```

`MemoryCore` deliberately does not expose a second public canonical write
authority. Canonical writes flow through Memory admission governance.

## Extraction provenance

The standalone core originated from:

- Mind Runtime `main`: `18d9275175ed6781962164e530efe4433e29ac7f`
- independent MemoryCore branch: `5071955449aff079f11fb958948fbabb28050acb`

Admission governance preserves the existing MR semantics while replacing its
concrete FactBackend dependency with provider-neutral source contracts.

See `docs/EXTRACTION_BOUNDARY.md` for the ownership boundary.
