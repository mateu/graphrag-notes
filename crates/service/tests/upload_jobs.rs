use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, FixtureEntityExtractor, InferenceCapabilities,
    LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::EmbeddedApplication;
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{net::TcpListener, sync::Notify};
use tokio_util::sync::CancellationToken;

const OWNER: &str = "owner-fixed-synthetic-test-token-1234567890";
const OTHER: &str = "other-fixed-synthetic-test-token-1234567890";
const READER: &str = "reader-fixed-synthetic-test-token-1234567890";
struct GateEmbedder {
    blocked: AtomicBool,
    blocked_text: Option<&'static str>,
    started: Notify,
    released: Notify,
}
impl GateEmbedder {
    fn blocks(&self, text: &str) -> bool {
        self.blocked_text.is_none_or(|marker| text.contains(marker))
    }
    fn release(&self) {
        self.blocked.store(false, Ordering::SeqCst);
        self.released.notify_waiters();
    }
    async fn wait(&self) {
        loop {
            let signal = self.released.notified();
            if !self.blocked.load(Ordering::SeqCst) {
                return;
            }
            self.started.notify_one();
            signal.await;
        }
    }
}
#[async_trait]
impl Embedder for GateEmbedder {
    async fn embed(&self, text: &str, query: bool) -> graphrag_agents::Result<Vec<f32>> {
        if self.blocks(text) {
            self.wait().await;
        }
        DeterministicEmbedder::default().embed(text, query).await
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        query: bool,
    ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        if texts.iter().any(|text| self.blocks(text)) {
            self.wait().await;
        }
        DeterministicEmbedder::default()
            .embed_batch(texts, query)
            .await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}
struct Fixture {
    _temp: tempfile::TempDir,
    url: String,
    client: reqwest::Client,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
    repo: Repository,
    provider: Arc<GateEmbedder>,
}
impl Fixture {
    async fn new() -> Self {
        Self::with_capacity(2).await
    }
    async fn with_capacity(capacity: usize) -> Self {
        Self::with_blocked_text(capacity, None).await
    }
    async fn with_blocked_text(capacity: usize, blocked_text: Option<&'static str>) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let credentials = temp.path().join("credentials.json");
        let entries = [
            (
                "openclaw",
                OWNER,
                vec![
                    Capability::Read,
                    Capability::Capture,
                    Capability::Upload,
                    Capability::Jobs,
                ],
            ),
            ("hermes", OTHER, vec![Capability::Read, Capability::Jobs]),
            ("reader", READER, vec![Capability::Read]),
        ];
        std::fs::write(
            &credentials,
            serde_json::to_vec(&CredentialFile {
                schema_version: 1,
                credentials: entries
                    .into_iter()
                    .map(|(name, token, capabilities)| Credential {
                        instance_id: name.into(),
                        token_sha256: format!("{:x}", Sha256::digest(token)),
                        capabilities,
                    })
                    .collect(),
            })
            .unwrap(),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let repo = Repository::new(init_memory().await.unwrap());
        let provider = Arc::new(GateEmbedder {
            blocked: AtomicBool::new(true),
            blocked_text,
            started: Notify::new(),
            released: Notify::new(),
        });
        let app = Arc::new(EmbeddedApplication::new(
            repo.clone(),
            SearchAgent::new(repo.clone(), provider.clone()),
            provider.clone(),
            Arc::new(FixtureEntityExtractor::default()),
            LibrarianRuntimeConfig {
                min_chunk_size: 1,
                skip_entity_extraction: true,
                ..Default::default()
            },
        ));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let options = ServiceOptions {
            listen: address,
            credentials_file: credentials,
            max_concurrent_requests: capacity,
            max_job_workers: 2,
            ..Default::default()
        };
        let task = tokio::spawn(async move { serve(listener, app, options, stop).await });
        Self {
            _temp: temp,
            url: format!("http://{address}/mcp"),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            shutdown,
            task,
            repo,
            provider,
        }
    }
    fn request(&self, token: &str, name: &str, args: Value) -> reqwest::RequestBuilder {
        self.client.post(&self.url).bearer_auth(token).header("accept","application/json, text/event-stream").header("mcp-protocol-version","2025-11-25").json(&json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":args}}))
    }
    async fn call(&self, token: &str, name: &str, args: Value) -> Value {
        self.request(token, name, args)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }
    async fn terminal(&self, id: &str) -> Value {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let response = self.call(OWNER, "get_job", json!({"id":id})).await;
                if matches!(
                    data(&response)["status"].as_str(),
                    Some("completed" | "failed" | "cancelled")
                ) {
                    return response;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap()
    }
    async fn stop(self) {
        self.provider.release();
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
fn upload(request: &str) -> Value {
    json!({"request_id":request,"document_key":"stable-document","content":"# Uploaded corpus\n\nserverjoblexeme survives disconnects","title":"Shared uploaded file","provenance":{"uri":"file:///this-client-only/notes.md","label":"Synthetic client","metadata":{"instance_id":"spoofed"}},"extract_entities":false})
}
fn data(value: &Value) -> &Value {
    &value["result"]["structuredContent"]["data"]
}
fn error(value: &Value) -> &Value {
    &value["result"]["structuredContent"]["error"]
}

#[tokio::test]
async fn same_document_uploads_queue_while_another_document_completes() {
    let fixture = Fixture::with_blocked_text(2, Some("blockedfirstversion")).await;
    let mut first_input = upload("same-document-first");
    first_input["content"] = json!("# First version\n\nblockedfirstversion waits for preparation");
    first_input["title"] = json!("First supplied title");
    let first = fixture
        .call(OWNER, "upload_source", first_input.clone())
        .await;
    assert!(error(&first).is_null(), "{first}");
    let first_id = data(&first)["job_id"].as_str().unwrap();
    let source_id = data(&first)["source_id"].as_str().unwrap();
    tokio::time::timeout(Duration::from_secs(5), fixture.provider.started.notified())
        .await
        .unwrap();

    let mut second_input = upload("same-document-second");
    second_input["content"] =
        json!("# Second version\n\nqueuedsecondversion replaces the first supplied content");
    second_input["title"] = json!("/Projects/topic");
    let second = fixture
        .call(OWNER, "upload_source", second_input.clone())
        .await;
    assert!(error(&second).is_null(), "{second}");
    let second_id = data(&second)["job_id"].as_str().unwrap();
    assert_ne!(first_id, second_id);
    assert_eq!(data(&second)["source_id"], source_id);
    let mut unrelated_input = upload("other-document");
    unrelated_input["document_key"] = json!("unrelated-document");
    unrelated_input["content"] = json!(
        "# Independent upload\n\nunrelatedprogress completes while another document is blocked"
    );
    let unrelated = fixture.call(OWNER, "upload_source", unrelated_input).await;
    assert!(error(&unrelated).is_null(), "{unrelated}");
    let unrelated_id = data(&unrelated)["job_id"].as_str().unwrap();
    assert_ne!(data(&unrelated)["source_id"], source_id);
    let independent_done = fixture.terminal(unrelated_id).await;
    assert_eq!(data(&independent_done)["status"], "completed");
    assert_eq!(data(&independent_done)["generation"], 1);

    // The second worker has had time to finish independent work, while the
    // first provider call remains blocked. Accepted same-document input must
    // still be queued, without a source generation or terminal conflict.
    let first_running = fixture.call(OWNER, "get_job", json!({"id":first_id})).await;
    assert_eq!(data(&first_running)["status"], "running");
    assert_eq!(data(&first_running)["phase"], "preparing");
    assert_eq!(data(&first_running)["generation"], 1);
    let second_queued = fixture
        .call(OWNER, "get_job", json!({"id":second_id}))
        .await;
    assert_eq!(data(&second_queued)["status"], "queued", "{second_queued}");
    assert!(data(&second_queued)["generation"].is_null());
    assert!(data(&second_queued)["error_code"].is_null());

    fixture.provider.release();
    let first_done = fixture.terminal(first_id).await;
    let second_done = fixture.terminal(second_id).await;
    assert_eq!(data(&first_done)["id"], first_id);
    assert_eq!(data(&first_done)["status"], "completed");
    assert_eq!(data(&first_done)["generation"], 1);
    assert_eq!(data(&first_done)["result"]["generation"], 1);
    assert_eq!(data(&first_done)["result"]["source_id"], source_id);
    assert_eq!(data(&second_done)["id"], second_id);
    assert_eq!(data(&second_done)["status"], "completed");
    assert_eq!(data(&second_done)["generation"], 2);
    assert_eq!(data(&second_done)["result"]["generation"], 2);
    assert_eq!(data(&second_done)["result"]["source_id"], source_id);
    let source = fixture
        .call(READER, "get_source", json!({"id":source_id}))
        .await;
    assert_eq!(data(&source)["generation"], 2);
    assert_eq!(data(&source)["successful_generation"], 2);
    assert_eq!(data(&source)["status"], "ready");
    assert_eq!(data(&source)["content"], second_input["content"]);
    assert_eq!(data(&source)["title"], second_input["title"]);
    for (input, id) in [(first_input, first_id), (second_input, second_id)] {
        let replay = fixture.call(OWNER, "upload_source", input).await;
        assert_eq!(data(&replay)["job_id"], id);
        assert_eq!(data(&replay)["replayed"], true);
    }
    assert_eq!(fixture.repo.get_stats().await.unwrap().note_count, 2);
    fixture.stop().await;
}

#[tokio::test]
async fn admitted_jobs_outlive_their_http_action_and_cancel_during_provider_work() {
    let fixture = Fixture::new().await;
    assert_eq!(
        error(
            &fixture
                .call(READER, "upload_source", upload("denied"))
                .await
        )["code"],
        "forbidden"
    );
    assert_eq!(
        error(&fixture.call(OTHER, "upload_source", upload("denied")).await)["code"],
        "forbidden"
    );
    let mut spoof = upload("spoof");
    spoof["instance_id"] = json!("hermes");
    assert_eq!(
        error(&fixture.call(OWNER, "upload_source", spoof).await)["code"],
        "invalid_input"
    );
    let mut path = upload("path");
    path["path"] = json!("/private/server/corpus.md");
    assert_eq!(
        error(&fixture.call(OWNER, "upload_source", path).await)["code"],
        "invalid_input"
    );
    let mut huge = upload("huge");
    huge["content"] = json!("😀".repeat(17000));
    assert_eq!(
        error(&fixture.call(OWNER, "upload_source", huge).await)["code"],
        "invalid_input"
    );
    let admitted = fixture
        .call(OWNER, "upload_source", upload("disconnect"))
        .await;
    let job = data(&admitted)["job_id"].as_str().unwrap().to_owned();
    let source = data(&admitted)["source_id"].as_str().unwrap().to_owned();
    // The request/response owner has gone; processing remains server-owned.
    drop(admitted);
    tokio::time::timeout(Duration::from_secs(5), fixture.provider.started.notified())
        .await
        .unwrap();
    assert_eq!(
        error(&fixture.call(OTHER, "get_job", json!({"id":job})).await)["code"],
        "not_found"
    );
    assert_eq!(
        error(&fixture.call(READER, "get_job", json!({"id":job})).await)["code"],
        "forbidden"
    );
    fixture.provider.release();
    let done = fixture.terminal(&job).await;
    assert_eq!(data(&done)["status"], "completed");
    let id = data(&done)["result"]["note_ids"][0].as_str().unwrap();
    let inspected = fixture
        .call(READER, "get_record", json!({"id":id,"neighbors":0}))
        .await;
    assert_eq!(data(&inspected)["provenance"]["instance_id"], "openclaw");
    assert_eq!(
        data(&inspected)["provenance"]["source"]["uri"],
        "file:///this-client-only/notes.md"
    );
    assert_eq!(
        data(
            &fixture
                .call(READER, "get_source", json!({"id":source}))
                .await
        )["content"],
        upload("disconnect")["content"]
    );
    let replay = fixture
        .call(OWNER, "upload_source", upload("disconnect"))
        .await;
    assert_eq!(data(&replay)["job_id"], job);
    assert_eq!(data(&replay)["replayed"], true);
    assert_eq!(fixture.repo.get_stats().await.unwrap().note_count, 1);
    fixture.provider.blocked.store(true, Ordering::SeqCst);
    let mut changed = upload("cancel");
    changed["content"] = json!("# Changed upload\n\nnewjoblexeme waits safely");
    let admission = fixture.call(OWNER, "upload_source", changed).await;
    let cancel_job = data(&admission)["job_id"].as_str().unwrap().to_owned();
    tokio::time::timeout(Duration::from_secs(5), fixture.provider.started.notified())
        .await
        .unwrap();
    let request=fixture.request(OWNER,"capture_note",json!({"request_id":"busy-capture","content":"busycapturelexeme","title":null,"tags":[],"provenance":null}));
    let capture = tokio::spawn(async move { request.send().await });
    tokio::time::timeout(Duration::from_secs(5), fixture.provider.started.notified())
        .await
        .unwrap();
    let cancelled = tokio::time::timeout(
        Duration::from_secs(1),
        fixture.call(OWNER, "cancel_job", json!({"id":cancel_job})),
    )
    .await
    .unwrap();
    assert!(error(&cancelled).is_null());
    let cancelled = fixture.terminal(&cancel_job).await;
    assert_eq!(data(&cancelled)["status"], "cancelled");
    fixture.provider.release();
    capture.await.unwrap().unwrap();
    assert_eq!(
        data(
            &fixture
                .call(OWNER, "resume_job", json!({"id":cancel_job}))
                .await
        )["status"],
        "queued"
    );
    assert_eq!(
        data(&fixture.terminal(&cancel_job).await)["status"],
        "completed"
    );
    fixture.stop().await;
}

#[tokio::test]
async fn service_shutdown_interrupts_blocked_worker_then_restart_can_resume_saved_input() {
    let mut fixture = Fixture::new().await;
    let admission = fixture
        .call(OWNER, "upload_source", upload("service-restart"))
        .await;
    let id = data(&admission)["job_id"].as_str().unwrap().to_owned();
    tokio::time::timeout(Duration::from_secs(5), fixture.provider.started.notified())
        .await
        .unwrap();
    fixture.shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), fixture.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let interrupted = fixture
        .repo
        .get_remote_upload_job("openclaw", &id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(interrupted.job.status, "failed");
    assert_eq!(interrupted.job.last_error.as_deref(), Some("interrupted"));
    assert_eq!(fixture.repo.get_stats().await.unwrap().note_count, 0);
    fixture.provider.release();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Arc::new(EmbeddedApplication::new(
        fixture.repo.clone(),
        SearchAgent::new(fixture.repo.clone(), fixture.provider.clone()),
        fixture.provider.clone(),
        Arc::new(FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig {
            min_chunk_size: 1,
            skip_entity_extraction: true,
            ..Default::default()
        },
    ));
    let options = ServiceOptions {
        listen: address,
        credentials_file: fixture._temp.path().join("credentials.json"),
        ..Default::default()
    };
    fixture.shutdown = CancellationToken::new();
    let signal = fixture.shutdown.clone();
    fixture.url = format!("http://{address}/mcp");
    fixture.task = tokio::spawn(async move { serve(listener, app, options, signal).await });
    assert_eq!(
        data(&fixture.call(OWNER, "get_job", json!({"id":id})).await)["status"],
        "failed"
    );
    assert_eq!(
        data(&fixture.call(OWNER, "resume_job", json!({"id":id})).await)["status"],
        "queued"
    );
    assert_eq!(data(&fixture.terminal(&id).await)["status"], "completed");
    assert_eq!(fixture.repo.get_stats().await.unwrap().note_count, 1);
    fixture.stop().await;
}

#[tokio::test]
async fn authenticated_cancellation_remains_available_with_all_ordinary_http_slots_occupied() {
    for capacity in [1, 2] {
        let fixture = Fixture::with_capacity(capacity).await;
        let admission = fixture
            .call(OWNER, "upload_source", upload("capacity-cancel"))
            .await;
        let job = data(&admission)["job_id"].as_str().unwrap().to_owned();
        tokio::time::timeout(Duration::from_secs(2), fixture.provider.started.notified())
            .await
            .unwrap();
        let mut captures = Vec::new();
        for index in 0..capacity {
            let request = fixture.request(OWNER, "capture_note", json!({"request_id":format!("occupy-{index}"),"content":format!("Blocked synthetic capture {index}"),"title":null,"tags":[],"provenance":null}));
            captures.push(tokio::spawn(async move { request.send().await }));
            tokio::time::timeout(Duration::from_secs(2), fixture.provider.started.notified())
                .await
                .unwrap();
        }
        let ordinary = fixture
            .request(OWNER, "get_job", json!({"id":job}))
            .send()
            .await
            .unwrap();
        assert_eq!(ordinary.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
        // A disconnected native MCP runtime must also be able to handshake and
        // rediscover cancellation while every ordinary request slot is held.
        for rpc in [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"capacity-reconnect","version":"1"}}}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/list"}),
        ] {
            let response = fixture
                .client
                .post(&fixture.url)
                .bearer_auth(OWNER)
                .header("accept", "application/json, text/event-stream")
                .header("mcp-protocol-version", "2025-11-25")
                .json(&rpc)
                .send()
                .await
                .unwrap();
            assert!(
                response.status().is_success(),
                "control handshake: {}",
                response.status()
            );
        }
        let unauthorized = fixture
            .request("invalid-synthetic-token", "cancel_job", json!({"id":job}))
            .send()
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), reqwest::StatusCode::UNAUTHORIZED);
        let cancelled = tokio::time::timeout(
            Duration::from_secs(2),
            fixture.call(OWNER, "cancel_job", json!({"id":job})),
        )
        .await
        .unwrap();
        assert!(error(&cancelled).is_null(), "{cancelled}");
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if fixture
                    .repo
                    .get_remote_upload_job("openclaw", &job)
                    .await
                    .unwrap()
                    .unwrap()
                    .job
                    .status
                    == "cancelled"
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        fixture.provider.release();
        for capture in captures {
            assert_eq!(
                capture.await.unwrap().unwrap().status(),
                reqwest::StatusCode::OK
            );
        }
        assert_eq!(
            data(&fixture.call(OWNER, "get_job", json!({"id":job})).await)["status"],
            "cancelled"
        );
        fixture.stop().await;
    }
}
