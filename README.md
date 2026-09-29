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


## Quick start

```python
from pathlib import Path

from mr_mem import MemoryCore

with MemoryCore(Path("memory.sqlite"), read_only=True) as memory:
    rows = memory.load_all()
```

`MemoryCore` deliberately does not expose a new public canonical write authority.
Canonical writes are expected to arrive through an admission layer; this first
extraction preserves the same authority boundary as Mind Runtime.

## Extraction provenance

The initial standalone core was extracted from:

- Mind Runtime `main`: `18d9275175ed6781962164e530efe4433e29ac7f`
- independent MemoryCore branch: `5071955449aff079f11fb958948fbabb28050acb`

See `docs/EXTRACTION_BOUNDARY.md` for the ownership boundary.
