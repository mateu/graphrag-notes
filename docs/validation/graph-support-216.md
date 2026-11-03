# Reviewed accepted neighbors at a fixed budget (#216)

Aggregate counts only. This note records no record IDs, query text, titles,
snippets, source paths or credentials.

## Protocol

- **Data:** an isolated restore of the verified post-migration schema-28
  backup taken at the `e76b741` deploy. It holds 31 accepted reviewed edges
  (`derived_from` 26, `supports` 3, `contradicts` 1, `related_to` 1).
- **Service:** each variant ran as a scratch loopback service with a scratch
  read-only credential. The live service and database were not used.
- **Runner:** the unchanged runner (`runner_sha256` `b96bcc10…0e7e`), with the
  frozen 8-question suite in both cap modes and the 12-edge probe with the cap
  off. Policies were off, on, on, off, at `max_chunks=2`,
  `max_total_tokens=1000` and `max_chunk_tokens=400`.
- **Control:** the deployed `e76b741` binary on the copy reproduced the live
  post-deploy rerun in 112/112 rounds, byte for byte. The experiment build with
  every variant switch off reproduced that control in 112/112 rounds.

## Where the probe misses lost

Graph-off misses the target chunk for 6 of the 12 probe edges. In the control,
5 of those 6 targets were already inside the eight-hit window as direct hits.
They lost packing to another direct hit, because the decayed path, 0.8 times
the anchor's score, stayed below their own direct score. Only 1 target was cut
by the window.

Over all graph-on rounds, 190 graph candidates were rank-limited. Of these, 178
were entity-mention seeds (hop 0). Only 12 were accepted-edge neighbors, and 8
of those had no query evidence.

## Variants

All variants are budget-neutral: the same chunk and token budgets, and no
reserved graph slot.

| Variant | Probe misses gained (of 6) | Support chunks displaced | Additions other than the expected chunk |
| --- | --- | --- | --- |
| V0, deployed control | 0 | 0 | 0 |
| V1, no decay over reviewed edges | 5 | 1 | 2 |
| V2, V1 plus window admission of a selected anchor's neighbor | same as V1 (0 admissions) | 1 | 2 |
| V2a, window admission only | 0 | 0 | 0 |
| **V3, V1 except toward a derivative (this change)** | **5** | **0** | **0** |

- **V1's failures:** both came from reviewed `derived_from` edges followed
  toward the derived note. In one probe the derivative replaced the target it
  restates. In another it filled the second slot next to that target.
- **V2:** with V1 every corroborated neighbor of a selected anchor already
  enters the window, so admission had nothing to add.
- **V2a:** admission alone placed 4 neighbors in the window. None was packed,
  because a candidate below the window's weakest hit normalizes to zero
  relevance.

### Frozen suite

- **Additions:** V1 and V3 add one question's support chunk in both cap modes.
  That is Q4, the case documented in [#186](graph-support-186.md). It displaces
  a chunk that is not a support chunk.
- **Verdicts:** unchanged at 0/8 pass. Q4 still fails on its frozen
  `allowed_relations`.
- **Other questions:** unchanged.

### Probe

- **Verdicts:** 0/12 pass before and after. With V3, 3 probes meet every
  mechanical criterion and await the human materiality judgement (status
  `needs_judgement`). Two still fail only because the target's file is already
  present graph-off.
- **Seed chunk:** in one gained probe the target replaces its own seed chunk,
  which is not a support chunk.

## Counters, tokens and latency

Values are per suite (nocap / cap1) and probe. Counters are totals over the
graph-on rounds.

| | added | annotating | rank_limited | graph-on tokens per call |
| --- | --- | --- | --- | --- |
| V0 | 0 / 0 / 0 | 8 / 8 / 28 | 56 / 56 / 78 | 747.6 / 767.9 / 750.0 |
| V3 | 2 / 2 / 2 | 8 / 8 / 34 | 56 / 56 / 76 | 748.8 / 769.0 / 783.2 |

- **Graph-off:** byte-identical in 56/56 off rounds for every variant, tokens
  included.
- **Graph-on tokens:** stay within the unchanged 1000-token budget (maximum
  851 per call). They change only because different chunks are packed.
- **Repeat rounds:** every question was stable across its repeated off and on
  rounds.
- **Latency:** the graph-on overhead over graph-off, within the same
  non-LTO experiment build, was 186 / 184 / 331 ms with every switch off and
  186 / 192 / 329 ms for this change.

## Limitations

- **Small sample:** 12 probe edges and 8 suite questions, from one corpus.
- **Selection bias:** V3's derivative exception was chosen after observing
  V1's two failures. It rests on those cases and on the fixture negatives, not
  on held-out questions.
- **Unlabeled displacements:** the displaced direct hits carry no relevance
  labels. The measured claim is only that no support chunk was displaced.
