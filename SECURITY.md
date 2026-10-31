# Privacy and publication

MR-Mem can store conversation text, derived summaries and embeddings. Treat the entire
runtime data directory and its backups as private. SQLite files and local caches are not
encrypted by this project. Use host filesystem access controls and protect backups.

## Credentials and provider access

- Credentials belong in process environment variables, never committed JSON or source.
- `.env.example` contains variable names only. `.env` files are not loaded automatically.
- Keep private configurations in ignored `runtime.json` or `runtime.private*.json` files.
- Hosted compilation, embedding and decision services receive the input needed for their
  requests. Select local providers or disable optional services when remote processing is
  inappropriate. Requests may incur provider charges.
- The sample service binds to loopback. For remote access use an authenticated TLS proxy.
  Setting `auth_token_env` requires a nonempty environment variable; absence never disables
  authentication silently. A configuration without `auth_token_env` has no authentication.
- Decision health contains counters and outcome categories, not source text or error bodies.
  Other diagnostic logs may contain provider errors; keep logs private and review before sharing.

## Files that must stay local

Do not commit credentials, raw conversations, evaluation datasets, prompt/response dumps,
SQLite databases (including WAL/SHM sidecars), embeddings, model caches, provider logs,
personal filesystem paths or account exports. `.gitignore` is a first barrier, not a secret
scanner: previously tracked files and Git history remain publishable despite ignore rules.

Synthetic test fixtures and public configuration examples may be committed. Keep actual
secrets out of screenshots, issue descriptions, PR bodies and test failure attachments.

## Before pushing

Review the named staged files and the commits being published, not only the current tree:

```bash
git diff --cached --stat
git diff --cached --check
git log --oneline origin/feat/aml-20261013..HEAD
gitleaks git . --redact --log-opts="origin/feat/aml-20261013..HEAD"
```

CI also scans reachable Git history with a checksum-pinned Gitleaks release. Do not use
scanner allowlists to suppress a real credential. If a secret has entered Git, revoke it
first, remove it from the affected unpublished history and rescan before publication.
For already published history, coordinate cleanup with repository administrators; deleting
the current file does not remove old blobs or cached copies.

Report suspected exposure privately to repository administrators. Never post the secret
itself in a public issue. No hosted reporting service or contact address is implied here.
