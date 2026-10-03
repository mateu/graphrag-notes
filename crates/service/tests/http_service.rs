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
use tokio::{net::TcpListener, sync::Notify};
use tokio_util::sync::CancellationToken;

const TOKEN_A: &str = "a-fixed-test-token-with-32-or-more-bytes";
const TOKEN_B: &str = "b-fixed-test-token-with-32-or-more-bytes";
const TOKEN_READ: &str = "read-fixed-test-token-with-32-or-more-bytes";

#[derive(Default)]
struct TestApplication {
    records: Mutex<BTreeMap<(String, String), (String, RemoteCaptureResponse)>>,
    captures: AtomicUsize,
    searches: AtomicUsize,
    slow: bool,
    huge: bool,
    started: Notify,
    release: Notify,
}

fn unsupported<T>() -> ApplicationResult<T> {
    Err(ApplicationError::Internal("unused test operation".into()))
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
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("credentials.json");
        credentials(
            &path,
            &[
                (
                    "openclaw-a",
                    TOKEN_A,
                    vec![Capability::Read, Capability::Capture],
                ),
                (
                    "hermes-b",
                    TOKEN_B,
                    vec![Capability::Read, Capability::Capture],
                ),
                ("reader", TOKEN_READ, vec![Capability::Read]),
            ],
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listen = listener.local_addr().unwrap();
        let options = ServiceOptions {
            listen,
            credentials_file: path.clone(),
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
    assert_eq!(tools.len(), 3);
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
