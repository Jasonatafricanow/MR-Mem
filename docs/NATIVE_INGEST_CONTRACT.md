# Native source ingestion

The host owns raw messages. Native ingestion persists SourceRefs, payload hashes,
batch status and accepted compiler output; it does not persist a transcript copy.
The host supplies a scope-checked reader and a complete snapshot of a session's
visible messages. Neither user text nor compiler output can choose a source binding.

Each synchronization registers the unseen source versions as one durable batch.
The existing compiler, compilation journal, canonical admission and Path A writer
process this batch. Earlier messages are read from the native store as context.
This host batch is the compilation boundary; AML's optional cross-request window
policy is not used by this adapter. The native adapter rejects that configuration.

Before processing or exposing context, synchronize source revisions and deletions.
Invalidate canonical support whose source changed. Cancel an unfinished batch if
any of its inputs changed, invalidate its partial canonical admissions, then
register its still-current inputs in a new batch. Completed batches retain their
unaffected support. A restart reuses accepted compilation without another model
call. Source verification is repeated by admission; concurrent changes fail closed
and are reconciled on the next synchronization. Database/ownership errors are not
interpreted as an empty session or deletion.

The compiler may remain disabled: batches stay pending, explicitly reported as
`compiler_not_configured`. An accepted fixture proves wiring only. Native writes
do not imply compiler semantic acceptance or Path B research acceptance.
