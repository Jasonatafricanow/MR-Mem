# Line graph read composition

The tree writer already persists compiler-accepted progression edges. Search must
consume that graph instead of substituting the older flat Thread summaries.

The shared reader opens existing graph/tense databases read-only. Canonical or
raw retrieval supplies seed Memory IDs; a bounded neighbourhood supplies related
steps. Only stored, knowledge-visible edges are rendered. No similarity-derived
edge, new summary, lifecycle mutation or source write is permitted. Graph gaps
are allowed; a partial persisted database is an error, not an empty graph.

Scope and canonical knowledge time apply to every step, not just the seed. A
consumer may impose a stricter visibility predicate (MR automatic surfacing).
Traversal cannot pass through a hidden step. Displayed edges always have both
endpoints in the returned context. Truncated neighbourhoods are labelled partial;
they are never labelled complete or the latest answer. Item and character budgets
apply to final output, without cutting a Block's text in half.

`knowledge_at` controls information availability. `now` controls the existing
message-time tense rule. These are separate query inputs. A later settling Block
cannot settle an earlier query unless its canonical state is available at that
knowledge cutoff. Missing message timestamps remain unknown. This reader is not
a reconstruction of all historical canonical lifecycle/attention policies.

AML tree/path_a uses the shared reader on both raw and canonical discovery paths;
baseline and flat modes retain their semantics. MR opts into the same reader by
supplying the MR-Mem `lce_tree` directory and keeps its automatic visibility rules.
Absent graph files mean no graph has yet been written; corrupt or partial files
fail explicitly. Graph content stays historical context, not MR canonical state.

Validation uses accepted fixture blocks/edges, not model-quality certification.
