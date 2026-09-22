# Voice PR integration checkpoint

This document records the source heads used by the #3149 integration-only
branch. Product fixes remain in their source PRs; this branch contains only
the merge result and cross-PR validation.

| Source | Head SHA |
| --- | --- |
| `upstream/main` | `edae4bbc7c0e0df8cf644ccfd7622f7646e912de` |
| PR #3078 | `5449e71672f6a0f869feb02ea0afad8480cba498` |
| PR #3089 | `a9ad1ad46420c36bf20fad22d594fc2694043e03` |
| PR #3103 | `9f111b33edb1647792e616ea0b8f3000f36d079a` |
| PR #3130 | `6548ed9dd221da81b49711a8e77c00fc4ac389a7` |

The #3089 source head includes the independent-ASR → speaker-shadow → response
arbiter → model text → TTS diagnostics and the parked external-ASR response
fix. The integration branch does not add product behavior beyond those source
heads.

Validation used an isolated pytest basetemp:

`162 passed, 1 skipped, 1 warning`

The initial run also exposed Windows default pytest temp-directory permission
errors; those disappeared with the isolated basetemp and are recorded as an
environment issue rather than a product failure.
