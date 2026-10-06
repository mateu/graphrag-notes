# Visible OpenClaw browser acceptance (#108)

On 2026-10-06, an operator used the normal signed-in OpenClaw control UI on clawd in one fresh acceptance chat. All five read-only routes visibly rendered a notes reply with the expected policy and the known canonical note ID: keyword, hybrid with graph off, explicit graph on, the saved hybrid/graph-on default, and explicit hybrid with graph on. The operator also inspected an actual screenshot: bounded titles, snippets, IDs, escaped actors and link labels were readable at a narrow viewport. The chat was archived, and its archived footer and composer state were visibly verified.

This completes the visible-browser criterion that remained pending in the [earlier dispatch acceptance](openclaw-dispatch-108.md). The [sanitized supplement](openclaw-browser-108.json) keeps the five route denominators and individual displayed observations. Queries, content, record IDs, actors, session URLs/identifiers, screenshots and configuration or credential fingerprints remain private.

| Route | Displayed handler total | Displayed connect | Displayed search |
| --- | ---: | ---: | ---: |
| Keyword | 46 ms | 26 ms | 21 ms |
| Hybrid, graph off | 508 ms | 0 ms | 508 ms |
| Explicit graph on | 509 ms | 0 ms | 509 ms |
| Saved hybrid/graph-on default | 488 ms | 0 ms | 488 ms |
| Explicit hybrid, graph on | 504 ms | 20 ms | 484 ms |

These are five single direct-command acceptance observations. They establish no percentile, latency budget, throughput gain, causal timing comparison or natural-language model orchestration result. The displayed phase values are rounded and nested; they may not sum exactly. No capture or model turn was requested in this browser walkthrough. Capture uncertainty and recovery retain their separate correctness fixtures and earlier evidence.

A final read-only identity check compared the normal configuration, default models, configured permissions and installed plugin runtime/package/manifest hashes with the accepted historical installed-normal baseline. They remained unchanged, the installed plugin was `0.5.0`, and profiling flags were absent. The effective gateway bearer matched normal MCP configuration. One authenticated `service_status` call verified the expected principal and storage readiness with `inference_probed=false`; the identity check created no gateway session or chat command and read no source record. The observed Shiva corpus owner still ran the published rc.3 binary. The snapshot was taken after visible UI completion; it is not described as preceding the first browser command.

After the visible walkthrough, five independent current MCP searches used the same normal configuration and authenticated principal. Their 25 results matched the saved UI order, bounded titles/snippets and displayed provenance. Each result received a revision-pinned `get_record` call, confirming the current full content and provenance. Five DOM GitHub links display a shorter label; comparison restored their persisted link targets before exact snippet comparison. These are independent current canonical readbacks. The original UI formatter prints no revisions, so this is not described as revision pinning of the original browser RPCs. No new chat, session or capture was created, and configuration, models, permissions, plugin sources and gateway owner remained unchanged.

The separate 186-command gateway acceptance retains its own pinned readbacks and timing cohort. The browser used the independently installed `0.5.0` plugin with the published rc.3 backend. It establishes no rc.4 build, installation, deployment or performance claim.
