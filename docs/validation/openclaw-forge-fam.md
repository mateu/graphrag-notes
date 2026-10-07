# Forge OpenClaw connection to Fam

The 2026-10-07 UTC qualification connected Forge OpenClaw to the existing Fam GraphRAG service using published `v0.1.0-rc.5`, SurrealDB `3.2.4`, and application schema `21`. Forge received the clients-only installation: all 109 published client payloads matched, and all 13 plugin payloads were copied and verified. No native GraphRAG installation or corpus import occurred on Forge.

The connection uses an authenticated, supervised SSH reverse forward terminating on Forge loopback. A dedicated principal has read and capture capabilities. Enrollment added one row while preserving both original principal rows and the existing Fam authority ledger. The accepted Fam binary, configuration, native owner, existing Clawd route, and selected model settings remained unchanged.

Forge's integration adds a separate private token environment file and gateway service drop-in, the published plugin, and the three permitted MCP tools. It preserves the original dotenv file, full tool profile, and other model settings. The gateway and its watchdog returned to their normal service state. SDK resolution used the installed OpenClaw package's public exports.

## Observed checks

| Check | Observed result |
| --- | --- |
| Native MCP catalog | Three expected projected tools; no tools dispatched by the catalog check. |
| Published registered read handler | Keyword/off, hybrid/off, and hybrid/on each returned five records. All 15 revision-bound independent reads matched the full shared record fields and exact displayed order. |
| Normal conversational gateway turn | One new verification session, two successful GraphRAG read RPCs, five search hits, and an exact first-hit inspection. A separate authenticated protocol read matched that full inspection. |
| Fictional capture and explicit replay | One registered-handler capture followed by one explicitly authorized SDK replay. Three revision-bound reads matched. Replay changed from false to true while the complete saved record, timestamps, tags, provenance, and request arguments stayed identical. |
| Captured marker search | Three registered read modes and 11 independent pinned reads passed; the captured record appeared in every mode. |
| Final Fam preservation | Three principals present; the original two rows, authority ledger, accepted binary, existing service and route, and models were preserved. The additional Forge tunnel owner was verified. |

The read and capture handler checks invoked the actual published plugin registration in a temporary harness. They establish registered-handler behavior, not visible UI interaction. The capture check did not exercise a normal gateway or agent write. The captured marker's keyword searchability does not establish completed vector embedding or entity extraction, and hybrid presence is not a substitute for those checks. Capture may use the service's existing models.

## Single-run read observations

The meaningful read-handler check displayed the following total elapsed times:

| Mode | Graph policy | Displayed total |
| --- | --- | ---: |
| Keyword | Off | 24 ms |
| Hybrid | Off | 447 ms |
| Hybrid | On | 484 ms |

These are one observation per mode from the temporary registered-handler check, including its connection/search work. They are neither latency percentiles nor conversational model timings. An earlier query with no keyword hit was retained separately and was not used to qualify meaningful keyword retrieval.

## Conversational evidence and limits

The normal gateway turn used the configured OpenAI `gpt-6.1-sol` model, with one successful assistant attempt, no fallback, and no model or thinking override. This establishes the observed provider/model selection; it does not independently establish the authentication mechanism.

The complete selected-run evidence contained nine ordered runtime events. The two GraphRAG tool calls and results paired exactly by run, session, thread, turn, and call identity. Their MCP response text contained complete JSON envelopes, sufficient to verify all search arguments, returned records, inspection arguments, and full inspection data. Structured snapshots contained depth truncation, so those snapshots alone were not used as the full-record proof.

The saved message snapshot also contained JavaScript tool discovery and dispatch through the agent's `exec` orchestration surface, and its end was count-truncated. Thus the evidence establishes two GraphRAG read RPCs, not a total count of all agent orchestration calls. No shell or process invocation appeared in the retained orchestration arguments.

Visible Forge UI behavior remains untested. This qualification makes no broader performance, vector-readiness, extraction-readiness, or authentication-modality claim. Private source content, queries, record identities, credential material, configuration paths, and evidence fingerprints remain outside this report.
