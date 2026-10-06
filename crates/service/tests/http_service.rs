use async_trait::async_trait;
use graphrag_application::*;
use graphrag_core::{Note, ProposedEdgeStatus};
use graphrag_db::{repository::DbStats, InspectionProvenance, RecordInspection};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tempfile::TempDir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Notify,
};
use tokio_util::sync::CancellationToken;

const TOKEN_A: &str = "a-fixed-test-token-with-32-or-more-bytes";
const TOKEN_B: &str = "b-fixed-test-token-with-32-or-more-bytes";
const TOKEN_READ: &str = "read-fixed-test-token-with-32-or-more-bytes";
const TOKEN_CAPTURE: &str = "capture-fixed-test-token-with-32-or-more-bytes";

#[derive(Default)]
struct TestApplication {
    records: Mutex<BTreeMap<(String, String), (String, RemoteCaptureResponse)>>,
    captures: AtomicUsize,
    uploads: AtomicUsize,
    searches: AtomicUsize,
    job_reads: AtomicUsize,
    slow: bool,
    huge: bool,
    near_limit: bool,
    started: Notify,
    release: Notify,
}

fn unsupported<T>() -> ApplicationResult<T> {
    Err(ApplicationError::Internal("unused test operation".into()))
}

#[tokio::test]
async fn readiness_reports_independent_evidence_without_reads_writes_or_job_permission() {
    let fixture = Fixture::new(TestApplication::default()).await;
    let response = fixture.tool(TOKEN_READ, "service_status", json!({})).await;
    let report = data(&response);
    assert_eq!(report["read_only"], true);
    assert_eq!(report["inference_probed"], false);
    assert_eq!(report["instance_id"], "reader");
    assert_eq!(report["application"]["storage"]["state"], "ready");
    assert_eq!(report["application"]["embeddings"]["state"], "unavailable");
    assert_eq!(report["application"]["sources"]["state"], "partial");
    assert_eq!(report["application"]["backup"]["state"], "unknown");
    assert_eq!(report["jobs"]["readiness"]["state"], "forbidden");
    assert_eq!(fixture.application.searches.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.application.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.application.job_reads.load(Ordering::SeqCst), 0);
    credentials(
        &fixture.credentials,
        &[(
            "reader",
            TOKEN_READ,
            vec![Capability::Read, Capability::Jobs],
        )],
    );
    let owned = fixture.tool(TOKEN_READ, "service_status", json!({})).await;
    assert_eq!(data(&owned)["jobs"]["sampled"], 1);
    assert_eq!(data(&owned)["jobs"]["counts"]["running"], 1);
    let encoded = owned.to_string();
    assert!(!encoded.contains("other-private-principal"));
    assert!(!encoded.contains("private-checkpoint"));
    assert!(!encoded.contains("/private/job/result"));
    let invalid = fixture
        .tool(TOKEN_READ, "service_status", json!({"deep":true}))
        .await;
    assert_eq!(error(&invalid)["code"], "invalid_input");
    credentials(
        &fixture.credentials,
        &[("reader", TOKEN_READ, vec![Capability::Capture])],
    );
    let denied = fixture.tool(TOKEN_READ, "service_status", json!({})).await;
    assert_eq!(error(&denied)["code"], "forbidden");
    fixture.task.abort();
}

#[async_trait]
impl ApplicationOperations for TestApplication {
    async fn search(
        &self,
        request: SearchRequest,
        _: ActionCancellation,
    ) -> ApplicationResult<Vec<RecordSummary>> {
        self.searches.fetch_add(1, Ordering::SeqCst);
        if request.query == "provider-error" {
            return Err(ApplicationError::ProviderUnavailable(
                "private-provider-token /private/corpus/config.toml".into(),
            ));
        }
        Ok(vec![RecordSummary {
            id: "note:shared".into(),
            revision: Some("a".repeat(64)),
            hit_type: "note".into(),
            title: Some("Shared corpus".into()),
            content: if self.huge {
                "x".repeat(2 * 1024 * 1024)
            } else if self.near_limit {
                "x".repeat(1_000_000)
            } else {
                request.query
            },
            provenance: Some(InspectionProvenance::default()),
            score: Some(1.0),
            warnings: vec![],
        }])
    }
    async fn recent(&self, _: usize) -> ApplicationResult<Vec<RecordSummary>> {
        unsupported()
    }
    async fn inspect(&self, reference: RecordRef, _: usize) -> ApplicationResult<RecordInspection> {
        if reference
            .revision
            .as_deref()
            .is_some_and(|revision| revision != "a".repeat(64))
        {
            return Err(ApplicationError::RevisionConflict("private details".into()));
        }
        Ok(RecordInspection {
            id: reference.id,
            hit_type: "note".into(),
            title: None,
            content: "Shared corpus".into(),
            revision: "a".repeat(64),
            provenance: InspectionProvenance::default(),
            conversations: vec![],
            messages: vec![],
            messages_truncated: false,
            warnings: vec![],
        })
    }
    async fn capture(&self, _: CaptureRequest, _: ActionCancellation) -> ApplicationResult<Note> {
        unsupported()
    }
    async fn source_status(&self, _: usize) -> ApplicationResult<Vec<SourceStatus>> {
        unsupported()
    }
    async fn proposals(
        &self,
        _: Option<ProposedEdgeStatus>,
        _: usize,
    ) -> ApplicationResult<Vec<ProposalCard>> {
        unsupported()
    }
    async fn proposal(&self, _: &str) -> ApplicationResult<ProposalCard> {
        unsupported()
    }
    async fn decide_proposal(&self, _: ProposalDecisionRequest) -> ApplicationResult<ProposalCard> {
        unsupported()
    }
    async fn stats(&self) -> ApplicationResult<DbStats> {
        unsupported()
    }
}

#[async_trait]
impl RemoteApplicationOperations for TestApplication {
    async fn list_remote_jobs(
        &self,
        caller: CallerIdentity,
        limit: usize,
    ) -> ApplicationResult<RemoteJobList> {
        self.job_reads.fetch_add(1, Ordering::SeqCst);
        assert_eq!(limit, 10);
        let job = |instance_id: String| RemoteJobStatus {
            id: "processing_job:private".into(),
            job_type: "upload".into(),
            instance_id,
            status: "running".into(),
            phase: "private-phase".into(),
            cancellation_requested: false,
            source_id: "source:private".into(),
            generation: Some(1),
            total: 1,
            completed: 0,
            failed: 0,
            checkpoint: Some("private-checkpoint".into()),
            result: Some(json!({"private_path":"/private/job/result"})),
            error_code: None,
            created_at: "2026-10-05T00:00:00Z".into(),
            updated_at: "2026-10-05T00:00:00Z".into(),
        };
        Ok(RemoteJobList {
            jobs: vec![
                job(caller.instance_id),
                job("other-private-principal".into()),
            ],
        })
    }
    async fn service_readiness(&self) -> ApplicationResult<ApplicationReadiness> {
        Ok(ApplicationReadiness {
            storage: Readiness::new(ReadinessState::Ready, "Owner read succeeded.", None),
            embeddings: Readiness::new(
                ReadinessState::Unavailable,
                "Cached stopped-provider evidence.",
                None,
            ),
            sources: Readiness::new(
                ReadinessState::Partial,
                "Registered source refresh incomplete.",
                None,
            ),
            ..Default::default()
        })
    }
    async fn upload_source(
        &self,
        caller: CallerIdentity,
        request: UploadSourceRequest,
    ) -> ApplicationResult<UploadAdmission> {
        if self.slow {
            self.started.notify_one();
            self.release.notified().await;
        }
        self.uploads.fetch_add(1, Ordering::SeqCst);
        assert_eq!(caller.instance_id, "openclaw-a");
        Ok(UploadAdmission {
            request_id: request.request_id,
            job_id: format!("processing_job:{}", "a".repeat(64)),
            source_id: format!("source:{}", "b".repeat(64)),
            source_uri: format!("mcp://upload/{}", "b".repeat(64)),
            replayed: false,
        })
    }
    async fn build_context(
        &self,
        request: BuildContextRequest,
        _: ActionCancellation,
    ) -> ApplicationResult<ContextResponse> {
        if request.max_chunks != Some(0)
            && request.max_total_tokens != Some(0)
            && request.max_chunk_tokens != Some(0)
        {
            return Err(ApplicationError::ProviderUnavailable(
                "context provider intentionally blocked".into(),
            ));
        }
        Ok(ContextResponse {
            query: request.query,
            scope: request.scope,
            chunks: vec![],
            total_tokens: 0,
            diagnostics: json!({}),
            rendered_context: String::new(),
        })
    }
    async fn capture_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteCaptureRequest,
        _: ActionCancellation,
    ) -> ApplicationResult<RemoteCaptureResponse> {
        if self.slow {
            self.started.notify_one();
            self.release.notified().await;
        }
        let payload = serde_json::to_string(&request).unwrap();
        let key = (caller.instance_id.clone(), request.request_id.clone());
        let mut records = self.records.lock().unwrap();
        if let Some((existing, response)) = records.get(&key) {
            if existing != &payload {
                return Err(ApplicationError::RevisionConflict("request changed".into()));
            }
            let mut response = response.clone();
            response.replayed = true;
            return Ok(response);
        }
        let index = self.captures.fetch_add(1, Ordering::SeqCst);
        let response = RemoteCaptureResponse {
            request_id: request.request_id,
            replayed: false,
            record: SavedRecord {
                id: format!("note:capture-{index}"),
                revision: "b".repeat(64),
                title: request.title,
                content: request.content,
                tags: request.tags,
                created_at: "2026-10-03T00:00:00Z".into(),
                updated_at: "2026-10-03T00:00:00Z".into(),
                provenance: json!({"instance_id":caller.instance_id,"source":request.provenance}),
            },
        };
        records.insert(key, (payload, response.clone()));
        Ok(response)
    }
}

fn credentials(path: &Path, entries: &[(&str, &str, Vec<Capability>)]) {
    let file = CredentialFile {
        schema_version: 1,
        credentials: entries
            .iter()
            .map(|(name, token, capabilities)| Credential {
                instance_id: (*name).into(),
                token_sha256: format!("{:x}", Sha256::digest(token.as_bytes())),
                capabilities: capabilities.clone(),
            })
            .collect(),
    };
    std::fs::write(path, serde_json::to_vec(&file).unwrap()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
}

struct Fixture {
    _temp: TempDir,
    credentials: std::path::PathBuf,
    url: String,
    client: Client,
    application: Arc<TestApplication>,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}

impl Fixture {
    async fn new(application: TestApplication) -> Self {
        Self::with_capacity(application, 8).await
    }
    async fn with_capacity(application: TestApplication, capacity: usize) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("credentials.json");
        credentials(
            &path,
            &[
                (
                    "openclaw-a",
                    TOKEN_A,
                    vec![Capability::Read, Capability::Capture, Capability::Upload],
                ),
                (
                    "hermes-b",
                    TOKEN_B,
                    vec![Capability::Read, Capability::Capture, Capability::Upload],
                ),
                ("reader", TOKEN_READ, vec![Capability::Read]),
                (
                    "capture-client",
                    TOKEN_CAPTURE,
                    vec![Capability::Read, Capability::Capture],
                ),
            ],
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listen = listener.local_addr().unwrap();
        let options = ServiceOptions {
            listen,
            credentials_file: path.clone(),
            max_concurrent_requests: capacity,
            ..ServiceOptions::default()
        };
        let application = Arc::new(application);
        let shared = application.clone();
        let shutdown = CancellationToken::new();
        let trigger = shutdown.clone();
        let task = tokio::spawn(async move { serve(listener, shared, options, trigger).await });
        Self {
            _temp: temp,
            credentials: path,
            url: format!("http://{listen}/mcp"),
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            application,
            shutdown,
            task,
        }
    }
    fn request(&self, token: &str, message: Value) -> reqwest::RequestBuilder {
        self.client
            .post(&self.url)
            .bearer_auth(token)
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2025-11-25")
            .json(&message)
    }
    async fn rpc(&self, token: &str, message: Value) -> Value {
        let response = self.request(token, message).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.json().await.unwrap()
    }
    async fn modern_catalog_rpc(&self, token: &str, method: &str) -> Value {
        let response = self
            .client
            .post(&self.url)
            .bearer_auth(token)
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2026-07-28")
            .header("mcp-method", method)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": {
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "modern-catalog-test", "version": "1"
                        },
                        "io.modelcontextprotocol/clientCapabilities": {}
                    }
                }
            }))
            .send()
            .await
            .unwrap();
        let status = response.status();
        assert!(response.headers().get("mcp-session-id").is_none());
        let body = response.json().await.unwrap();
        assert_eq!(status, StatusCode::OK, "{method} response: {body}");
        body
    }
    async fn tool(&self, token: &str, name: &str, arguments: Value) -> Value {
        self.rpc(token, json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":arguments}})).await
    }
    async fn stop(self) {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

fn capture(request_id: &str) -> Value {
    json!({"request_id":request_id,"content":"Atlas shared capture","title":"Atlas","tags":["shared"],"provenance":{"uri":"client:opaque","label":"Client source","metadata":{"instance_id":"spoofed"}}})
}
fn search() -> Value {
    json!({"query":"Atlas","mode":"keyword","scope":"all","limit":10,"graph":"off"})
}
fn data(result: &Value) -> &Value {
    &result["result"]["structuredContent"]["data"]
}
fn error(result: &Value) -> &Value {
    &result["result"]["structuredContent"]["error"]
}

#[tokio::test]
async fn modern_catalog_cache_metadata_is_private_and_preserves_principal_filtering() {
    let fixture = Fixture::new(TestApplication::default()).await;
    for (token, can_capture) in [(TOKEN_CAPTURE, true), (TOKEN_READ, false)] {
        let discovered = fixture.modern_catalog_rpc(token, "server/discover").await;
        let discovery = &discovered["result"];
        assert_eq!(discovery["resultType"], "complete");
        assert_eq!(discovery["ttlMs"], 0);
        assert_eq!(discovery["cacheScope"], "private");
        assert!(discovery["supportedVersions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|version| version == "2026-07-28"));
        assert!(discovery["capabilities"].get("tools").is_some());

        let listed = fixture.modern_catalog_rpc(token, "tools/list").await;
        let catalog = &listed["result"];
        assert_eq!(catalog["resultType"], "complete");
        assert_eq!(catalog["ttlMs"], 0);
        assert_eq!(catalog["cacheScope"], "private");
        let tools = catalog["tools"].as_array().unwrap();
        let mut names: Vec<_> = tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        names.sort_unstable();
        let mut expected = vec![
            "build_context",
            "get_note",
            "get_proposal",
            "get_record",
            "get_source",
            "list_proposals",
            "search_notes",
            "service_status",
        ];
        if can_capture {
            expected.push("capture_note");
            expected.sort_unstable();
        }
        assert_eq!(names, expected);
        assert!(tools.iter().all(|tool| {
            tool["annotations"]["readOnlyHint"] == (tool["name"] != "capture_note")
        }));

        let legacy = fixture
            .rpc(token, json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}))
            .await;
        assert_eq!(legacy["result"]["tools"], catalog["tools"]);
        // rmcp strips the modern discriminator, retaining additive cache hints.
        assert!(legacy["result"].get("resultType").is_none());
        assert_eq!(legacy["result"]["ttlMs"], 0);
        assert_eq!(legacy["result"]["cacheScope"], "private");
    }
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.application.searches.load(Ordering::SeqCst), 0);
    fixture.stop().await;
}

#[tokio::test]
async fn capture_discovery_publishes_provenance_string_and_map_bounds() {
    let fixture = Fixture::new(TestApplication::default()).await;
    let listed = fixture
        .rpc(
            TOKEN_A,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        )
        .await;
    let capture = listed["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "capture_note")
        .unwrap();
    let schema = &capture["inputSchema"];
    assert_eq!(schema["properties"]["tags"]["maxItems"], 32);
    assert_eq!(schema["properties"]["tags"]["items"]["minLength"], 1);
    assert_eq!(schema["properties"]["tags"]["items"]["maxLength"], 64);
    let reference = schema["properties"]["provenance"]["anyOf"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|branch| branch["$ref"].as_str())
        .unwrap();
    let provenance = schema
        .pointer(reference.strip_prefix('#').unwrap())
        .unwrap();
    assert_eq!(provenance["additionalProperties"], false);
    assert_eq!(provenance["properties"]["uri"]["minLength"], 1);
    assert_eq!(provenance["properties"]["uri"]["maxLength"], 2048);
    assert_eq!(provenance["properties"]["label"]["maxLength"], 512);
    let metadata = &provenance["properties"]["metadata"];
    assert_eq!(metadata["type"], "object");
    assert_eq!(metadata["maxProperties"], 32);
    assert_eq!(metadata["propertyNames"]["type"], "string");
    assert_eq!(metadata["propertyNames"]["minLength"], 1);
    assert_eq!(metadata["propertyNames"]["maxLength"], 64);
    assert_eq!(metadata["additionalProperties"]["type"], "string");
    assert_eq!(metadata["additionalProperties"]["maxLength"], 1024);
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 0);
    fixture.stop().await;
}

#[tokio::test]
async fn legacy_handshake_catalog_and_trusted_instance_retry_contract() {
    let fixture = Fixture::new(TestApplication::default()).await;
    let response = fixture.request(TOKEN_A, json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"compatibility-test","version":"1"}}})).send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers().get("mcp-session-id").is_none(),
        "stateless legacy clients must not acquire bearer-independent sessions"
    );
    let initialized: Value = response.json().await.unwrap();
    assert_eq!(initialized["result"]["protocolVersion"], "2025-11-25");
    let notification = fixture
        .request(
            TOKEN_A,
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(notification.status(), StatusCode::ACCEPTED);
    let listed = fixture
        .rpc(
            TOKEN_READ,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        )
        .await;
    let tools = listed["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 8);
    assert!(tools
        .iter()
        .all(|tool| tool["annotations"]["readOnlyHint"] == true
            && tool["outputSchema"]["type"] == "object"));
    let zero = fixture
        .tool(
            TOKEN_READ,
            "build_context",
            json!({"query":"Atlas","max_total_tokens":0}),
        )
        .await;
    assert_eq!(data(&zero)["total_tokens"], 0);
    assert_eq!(data(&zero)["chunks"], json!([]));
    assert_eq!(
        data(&fixture.tool(TOKEN_READ, "search_notes", search()).await)["records"][0]["id"],
        "note:shared"
    );
    let denied = fixture
        .tool(TOKEN_READ, "capture_note", capture("same-id"))
        .await;
    assert_eq!(error(&denied)["code"], "forbidden");
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 0);
    let first = fixture
        .tool(TOKEN_A, "capture_note", capture("same-id"))
        .await;
    let replay = fixture
        .tool(TOKEN_A, "capture_note", capture("same-id"))
        .await;
    let other = fixture
        .tool(TOKEN_B, "capture_note", capture("same-id"))
        .await;
    assert_eq!(
        data(&first)["record"]["provenance"]["instance_id"],
        "openclaw-a"
    );
    assert_eq!(
        data(&other)["record"]["provenance"]["instance_id"],
        "hermes-b"
    );
    assert_eq!(data(&first)["record"]["id"], data(&replay)["record"]["id"]);
    assert_eq!(data(&replay)["replayed"], true);
    assert_ne!(data(&first)["record"]["id"], data(&other)["record"]["id"]);
    let mut changed = capture("same-id");
    changed["content"] = json!("changed draft");
    assert_eq!(
        error(&fixture.tool(TOKEN_A, "capture_note", changed).await)["code"],
        "revision_conflict"
    );
    let mut spoof = capture("spoof");
    spoof["instance_id"] = json!("hermes-b");
    assert_eq!(
        error(&fixture.tool(TOKEN_A, "capture_note", spoof).await)["code"],
        "invalid_input"
    );
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 2);
    fixture.stop().await;
}

#[tokio::test]
async fn authentication_rotation_origin_host_and_body_limits_fail_closed() {
    let fixture = Fixture::new(TestApplication::default()).await;
    let message = json!({"jsonrpc":"2.0","id":1,"method":"tools/list"});
    assert_eq!(
        fixture
            .client
            .post(&fixture.url)
            .json(&message)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        fixture
            .request("invalid-invalid-invalid-invalid-token", message.clone())
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        fixture
            .request(TOKEN_A, message.clone())
            .header("origin", "https://evil.example")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        fixture
            .request(TOKEN_A, message.clone())
            .header("host", "evil.example")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        fixture
            .request(TOKEN_A, message.clone())
            .header("origin", "null")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    let huge = "x".repeat(128 * 1024 + 1);
    assert_eq!(
        fixture
            .client
            .post(&fixture.url)
            .bearer_auth(TOKEN_A)
            .header("accept", "application/json, text/event-stream")
            .header("content-type", "application/json")
            .body(huge)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    credentials(
        &fixture.credentials,
        &[("rotated", TOKEN_B, vec![Capability::Read])],
    );
    assert_eq!(
        fixture
            .request(TOKEN_A, message.clone())
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        fixture
            .request(TOKEN_B, message.clone())
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    std::fs::write(&fixture.credentials, b"{ private malformed token").unwrap();
    let response = fixture.request(TOKEN_B, message).send().await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response.text().await.unwrap();
    assert!(!body.contains("private malformed token"));
    assert!(!body.contains(fixture.credentials.to_str().unwrap()));
    fixture.stop().await;
}

#[tokio::test]
async fn response_and_argument_bounds_and_errors_are_actionable_and_sanitized() {
    let fixture = Fixture::new(TestApplication {
        huge: true,
        ..TestApplication::default()
    })
    .await;
    assert_eq!(
        error(&fixture.tool(TOKEN_READ, "search_notes", search()).await)["code"],
        "response_too_large"
    );
    let mut request = search();
    request["limit"] = json!(201);
    assert_eq!(
        error(&fixture.tool(TOKEN_READ, "search_notes", request).await)["code"],
        "invalid_input"
    );
    let mut request = search();
    request["since_days"] = json!(u32::MAX);
    assert_eq!(
        error(
            &fixture
                .tool(TOKEN_READ, "search_notes", request.clone())
                .await
        )["code"],
        "invalid_input"
    );
    request.as_object_mut().unwrap().remove("mode");
    request.as_object_mut().unwrap().remove("limit");
    assert_eq!(
        error(&fixture.tool(TOKEN_READ, "build_context", request).await)["code"],
        "invalid_input"
    );
    assert_eq!(fixture.application.searches.load(Ordering::SeqCst), 1);
    let mut request = search();
    request["query"] = json!("provider-error");
    let response = fixture.tool(TOKEN_READ, "search_notes", request).await;
    assert_eq!(error(&response)["code"], "provider_unavailable");
    assert!(error(&response)["message"]
        .as_str()
        .unwrap()
        .contains("mode=keyword"));
    assert!(!response.to_string().contains("private-provider-token"));
    assert!(!response.to_string().contains("/private/corpus"));
    let mut capture = capture("oversize");
    capture["content"] = json!("😀".repeat(17000));
    assert_eq!(
        error(&fixture.tool(TOKEN_A, "capture_note", capture).await)["code"],
        "invalid_input"
    );
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 0);
    fixture.stop().await;
}

#[tokio::test]
async fn disconnected_capture_finishes_and_shutdown_drains_it() {
    let fixture = Fixture::new(TestApplication {
        slow: true,
        ..TestApplication::default()
    })
    .await;
    let request = fixture.request(TOKEN_A, json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"capture_note","arguments":capture("disconnect-id")}}));
    let client = tokio::spawn(async move { request.send().await });
    tokio::time::timeout(
        Duration::from_secs(5),
        fixture.application.started.notified(),
    )
    .await
    .unwrap();
    client.abort();
    let _ = client.await;
    fixture.shutdown.cancel();
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(
        !fixture.task.is_finished(),
        "shutdown must retain the application while the detached write is pending"
    );
    fixture.application.release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), fixture.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(fixture.application.captures.load(Ordering::SeqCst), 1);
    let records = fixture.application.records.lock().unwrap();
    assert_eq!(
        records.keys().next().unwrap(),
        &("openclaw-a".into(), "disconnect-id".into())
    );
}

#[tokio::test]
async fn authenticated_stalled_bodies_release_capacity_and_bound_shutdown() {
    for shutting_down in [false, true] {
        let fixture = Fixture::new(TestApplication::default()).await;
        let url = reqwest::Url::parse(&fixture.url).unwrap();
        let address = (url.host_str().unwrap(), url.port().unwrap());
        let mut stalled = Vec::new();
        // This combined service reserves two short ingress/control slots for jobs.
        for _ in 0..ServiceOptions::default().max_concurrent_requests + 2 {
            let mut stream = TcpStream::connect(address).await.unwrap();
            let headers = format!(
                "POST /mcp HTTP/1.1\r\nHost: {}:{}\r\nAuthorization: Bearer {TOKEN_READ}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: 2\r\n\r\n{{",
                address.0, address.1
            );
            stream.write_all(headers.as_bytes()).await.unwrap();
            stalled.push(stream);
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let response = fixture
                    .request(
                        TOKEN_READ,
                        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
                    )
                    .send()
                    .await
                    .unwrap();
                if response.status() == StatusCode::TOO_MANY_REQUESTS {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        if shutting_down {
            fixture.shutdown.cancel();
        }
        // Keep every incomplete connection open. Only server-side deadlines
        // may release their slots or let graceful shutdown finish.
        tokio::time::timeout(Duration::from_secs(7), async {
            for stream in &mut stalled {
                let mut status = [0; 12];
                stream.read_exact(&mut status).await.unwrap();
                assert_eq!(&status, b"HTTP/1.1 408");
            }
        })
        .await
        .unwrap();
        if shutting_down {
            tokio::time::timeout(Duration::from_secs(2), fixture.task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        } else {
            let response = fixture.tool(TOKEN_READ, "search_notes", search()).await;
            assert!(response["result"]["structuredContent"]["error"].is_null());
            fixture.stop().await;
        }
    }
}

#[test]
fn bind_policy_requires_encryption_and_explicit_hosts_before_corpus_open() {
    let options = ServiceOptions {
        listen: "0.0.0.0:3000".parse().unwrap(),
        ..ServiceOptions::default()
    };
    assert!(options
        .validate()
        .unwrap_err()
        .to_string()
        .contains("external encryption"));
    let options = ServiceOptions {
        external_encryption: true,
        ..options
    };
    assert!(options
        .validate()
        .unwrap_err()
        .to_string()
        .contains("explicit allowed hosts"));
}

#[tokio::test]
async fn complete_rpc_body_size_includes_large_echoed_request_ids() {
    let fixture = Fixture::new(TestApplication {
        near_limit: true,
        ..Default::default()
    })
    .await;
    let normal = fixture.tool(TOKEN_READ, "search_notes", search()).await;
    assert_eq!(
        normal["result"]["structuredContent"]["data"]["records"][0]["content"]
            .as_str()
            .unwrap()
            .len(),
        1_000_000
    );
    let id = "rpc-id".repeat(20_000);
    let response = fixture.request(TOKEN_READ, json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":"search_notes","arguments":search()}}))
        .send().await.unwrap();
    assert!(response.status().is_success());
    let bytes = response.bytes().await.unwrap();
    assert!(bytes.len() <= 2 * 1024 * 1024);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["id"], id);
    assert_eq!(error(&value)["code"], "response_too_large");
    fixture.stop().await;
}

#[tokio::test]
async fn lost_upload_acknowledgment_does_not_drop_accepted_admission_and_shutdown_drains_it() {
    let fixture = Fixture::new(TestApplication {
        slow: true,
        ..Default::default()
    })
    .await;
    let request = fixture.request(TOKEN_A, json!({"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"upload_source","arguments":{"request_id":"lost-upload-ack","document_key":"synthetic-document","content":"Upload admitted after client leaves","title":null,"provenance":null,"extract_entities":false}}}));
    let client = tokio::spawn(async move { request.send().await });
    tokio::time::timeout(
        Duration::from_secs(5),
        fixture.application.started.notified(),
    )
    .await
    .unwrap();
    assert!(
        !client.is_finished(),
        "the response must still be pending when the client disconnects"
    );
    client.abort();
    let _ = client.await;
    fixture.shutdown.cancel();
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(
        !fixture.task.is_finished(),
        "accepted upload admission must drain before storage is released"
    );
    fixture.application.release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), fixture.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(fixture.application.uploads.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn disconnected_admissions_keep_bounded_capacity_without_blocking_cancellation() {
    let fixture = Fixture::with_capacity(
        TestApplication {
            slow: true,
            ..Default::default()
        },
        1,
    )
    .await;
    credentials(
        &fixture.credentials,
        &[(
            "openclaw-a",
            TOKEN_A,
            vec![Capability::Read, Capability::Upload, Capability::Jobs],
        )],
    );
    let message = |request_id: &str| json!({"request_id":request_id,"document_key":"capacity-document","content":"Synthetic bounded admission","title":null,"provenance":null,"extract_entities":false});
    let first = fixture.request(TOKEN_A, json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"upload_source","arguments":message("accepted")}}));
    let waiter = tokio::spawn(async move { first.send().await });
    tokio::time::timeout(
        Duration::from_secs(2),
        fixture.application.started.notified(),
    )
    .await
    .unwrap();
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    // The HTTP slot becomes free while the accepted admission remains blocked.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let response = fixture.request(TOKEN_A, json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_notes","arguments":search()}})).send().await.unwrap();
            if response.status() == StatusCode::OK { break; }
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }).await.unwrap();
    for index in 0..20 {
        let rejected = fixture
            .tool(
                TOKEN_A,
                "upload_source",
                message(&format!("excess-{index}")),
            )
            .await;
        assert_eq!(error(&rejected)["code"], "busy");
    }
    // This adapter deliberately has no job implementation: compatibility proves
    // the authenticated control call reached it despite full admission capacity.
    let control = fixture
        .tool(
            TOKEN_A,
            "cancel_job",
            json!({"id":format!("processing_job:{}", "a".repeat(64))}),
        )
        .await;
    assert_eq!(error(&control)["code"], "compatibility");
    fixture.application.release.notify_waiters();
    tokio::time::timeout(Duration::from_secs(2), async {
        while fixture.application.uploads.load(Ordering::SeqCst) != 1 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    fixture.stop().await;
}
