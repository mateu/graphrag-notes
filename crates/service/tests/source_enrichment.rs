use graphrag_agents::{Embedder, LibrarianRuntimeConfig, SearchAgent};
use graphrag_application::*;
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

fn application(repo: &Repository, embedder: Arc<dyn Embedder>) -> Arc<EmbeddedApplication> {
    Arc::new(EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        Arc::new(graphrag_agents::FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    ))
}
fn token(name: &str) -> String {
    format!("{name}-synthetic-token-at-least-thirty-two-bytes")
}
struct Fixture {
    _temp: tempfile::TempDir,
    url: String,
    client: reqwest::Client,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}
impl Fixture {
    async fn new(application: Arc<dyn RemoteApplicationOperations>) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("credentials.json");
        let credentials = CredentialFile {
            schema_version: 1,
            credentials: [
                ("reader", Capability::Read),
                ("editor", Capability::Edit),
                ("deleter", Capability::Delete),
                ("acceptor", Capability::Accept),
                ("rejector", Capability::Reject),
                ("undoer", Capability::Undo),
                ("capturer", Capability::Capture),
                ("proposer", Capability::Propose),
                ("uploader", Capability::Upload),
                ("jobs", Capability::Jobs),
                ("enricher", Capability::Enrich),
                ("foreign", Capability::Enrich),
            ]
            .into_iter()
            .map(|(name, cap)| Credential {
                instance_id: name.into(),
                token_sha256: format!("{:x}", Sha256::digest(token(name).as_bytes())),
                capabilities: vec![cap],
            })
            .collect(),
        };
        std::fs::write(&file, serde_json::to_vec(&credentials).unwrap()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listen = listener.local_addr().unwrap();
        let options = ServiceOptions {
            listen,
            credentials_file: file,
            ..Default::default()
        };
        let shutdown = CancellationToken::new();
        let trigger = shutdown.clone();
        let task =
            tokio::spawn(async move { serve(listener, application, options, trigger).await });
        Self {
            _temp: temp,
            url: format!("http://{listen}/mcp"),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            shutdown,
            task,
        }
    }
    fn request(&self, name: &str, method: &str, params: Value) -> reqwest::RequestBuilder {
        self.client
            .post(&self.url)
            .bearer_auth(token(name))
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2025-11-25")
            .json(&json!({"jsonrpc":"2.0","id":1,"method":method,"params":params}))
    }
    async fn rpc(&self, name: &str, method: &str, params: Value) -> Value {
        let response = self.request(name, method, params).send().await.unwrap();
        let status = response.status();
        let body = response.text().await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK, "{body}");
        serde_json::from_str(&body).unwrap()
    }
    async fn tool(&self, name: &str, tool: &str, args: Value) -> Value {
        self.rpc(name, "tools/call", json!({"name":tool,"arguments":args}))
            .await
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
fn data(v: &Value) -> &Value {
    &v["result"]["structuredContent"]["data"]
}
fn error(v: &Value) -> &Value {
    &v["result"]["structuredContent"]["error"]["code"]
}

#[tokio::test]
async fn authenticated_enrichment_is_separately_opt_in_owned_reviewed_and_durable() {
    let db = init_memory().await.unwrap();
    let repo = Repository::new(db.clone());
    let app = application(
        &repo,
        Arc::new(graphrag_agents::DeterministicEmbedder::default()),
    );
    let caller = CallerIdentity {
        instance_id: "enricher".into(),
    };
    let upload = app
        .upload_source(
            caller.clone(),
            UploadSourceRequest {
                request_id: "vector-upload".into(),
                document_key: "fictional-doc".into(),
                content: "# Atlas\n\nAtlas launch evidence.".into(),
                title: None,
                provenance: None,
                extract_entities: false,
                preserve_unchanged: false,
                create_only: false,
                expected_source_revision: None,
                policy_migration: None,
            },
        )
        .await
        .unwrap();
    let execution = app
        .claim_remote_job("setup-epoch", "setup-worker")
        .await
        .unwrap()
        .unwrap();
    app.execute_remote_job(execution, ActionCancellation::new())
        .await
        .unwrap();
    let source = app.get_uploaded_source(&upload.source_id).await.unwrap();
    let fixture = Fixture::new(app.clone()).await;
    for name in [
        "reader", "proposer", "uploader", "jobs", "acceptor", "capturer",
    ] {
        let list = fixture.rpc(name, "tools/list", json!({})).await;
        assert!(!list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "execute_source_enrichment"));
        assert_eq!(error(&fixture.tool(name,"prepare_source_enrichment",json!({"request_id":"review","source_id":source.id,"revision":source.revision})).await),"forbidden");
    }
    for args in [
        json!({"id":source.id}),
        json!({"document_key":source.document_key}),
    ] {
        assert_eq!(
            error(&fixture.tool("reader", "get_source", args).await),
            "not_found"
        );
    }
    let list = fixture.rpc("enricher", "tools/list", json!({})).await;
    assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 7);
    assert_eq!(
        error(
            &fixture
                .tool(
                    "foreign",
                    "prepare_source_enrichment",
                    json!({"request_id":"review","source_id":source.id,"revision":source.revision})
                )
                .await
        ),
        "revision_conflict"
    );
    let prepared = fixture
        .tool(
            "enricher",
            "prepare_source_enrichment",
            json!({"request_id":"review","source_id":source.id,"revision":source.revision}),
        )
        .await;
    assert!(data(&prepared)["plan_sha256"].is_string(), "{prepared}");
    let request = json!({"request_id":"review","reviewed":data(&prepared),"confirmed":true});
    let mut denied = request.clone();
    denied["confirmed"] = false.into();
    assert_eq!(
        error(
            &fixture
                .tool("enricher", "execute_source_enrichment", denied)
                .await
        ),
        "invalid_input"
    );
    assert_eq!(
        error(
            &fixture
                .tool("foreign", "execute_source_enrichment", request.clone())
                .await
        ),
        "invalid_input"
    );
    let admitted = fixture
        .tool("enricher", "execute_source_enrichment", request.clone())
        .await;
    assert!(data(&admitted)["job_id"].is_string(), "{admitted}");
    let id = data(&admitted)["job_id"].clone();
    assert_eq!(
        data(
            &fixture
                .tool("enricher", "execute_source_enrichment", request)
                .await
        )["replayed"],
        true
    );
    assert_eq!(
        data(
            &fixture
                .tool("enricher", "read_source_enrichment_plan", json!({"id":id}))
                .await
        )["plan_sha256"],
        data(&prepared)["plan_sha256"]
    );
    assert_eq!(
        error(
            &fixture
                .tool("foreign", "read_source_enrichment_plan", json!({"id":id}))
                .await
        ),
        "not_found"
    );
    let completed = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let result = fixture
                .tool("enricher", "get_source_enrichment_job", json!({"id":id}))
                .await;
            if data(&result)["status"] == "completed" {
                break result;
            }
            assert_ne!(data(&result)["status"], "failed", "{result}");
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        data(&completed)["result"]["current_source"]["extraction_policy_current"],
        true
    );
    assert_eq!(
        data(&completed)["result"]["reviewed"]["plan_sha256"],
        data(&prepared)["plan_sha256"]
    );
    let rollback = json!({"request_id":"rollback-review","reviewed":data(&prepared),"confirmed":true,"rollback_source_revision":data(&completed)["result"]["current_source"]["revision"]});
    let admitted = fixture
        .tool("enricher", "rollback_source_enrichment", rollback)
        .await;
    assert!(data(&admitted)["job_id"].is_string(), "{admitted}");
    let id = data(&admitted)["job_id"].clone();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let result = fixture
                .tool("enricher", "get_source_enrichment_job", json!({"id":id}))
                .await;
            if data(&result)["status"] == "completed" {
                assert_eq!(
                    data(&result)["result"]["current_source"]["extract_entities"],
                    false
                );
                break;
            }
            assert_ne!(data(&result)["status"], "failed", "{result}");
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    })
    .await
    .unwrap();
    // Queue a full generic readiness window behind the ordinary upload. Their
    // intentionally damaged conversion envelopes must be classified in the
    // database, without deserializing a plan, before the bounded window.
    let hidden_template = repo
        .get_remote_upload_job("enricher", id.as_str().unwrap())
        .await
        .unwrap()
        .unwrap()
        .input;
    for index in 0..10 {
        let mut hidden_input = hidden_template.clone();
        hidden_input.request_id = format!("damaged-enrichment-{index}");
        hidden_input.payload_fingerprint =
            format!("{:x}", Sha256::digest(hidden_input.request_id.as_bytes()));
        let hidden = repo.admit_remote_upload(hidden_input).await.unwrap();
        db.query(
            "UPDATE $job SET status = 'failed', remote_input.enrichment = { broken: true }, updated_at = time::now()",
        )
        .bind((
            "job",
            graphrag_db::parse_record_id(hidden.result["job_id"].as_str().unwrap(), Some("processing_job")).unwrap(),
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
    }
    // Rotate a Jobs+Read+Enrich bearer onto the same owner to retain all three
    // durable job families before enrichment authority is revoked.
    let path = fixture._temp.path().join("credentials.json");
    let mut credentials: CredentialFile =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for credential in &mut credentials.credentials {
        if credential.instance_id == "enricher" {
            credential.instance_id = "former-enricher".into();
        } else if credential.instance_id == "jobs" {
            credential.instance_id = "enricher".into();
            credential
                .capabilities
                .extend([Capability::Read, Capability::Enrich]);
        }
    }
    std::fs::write(&path, serde_json::to_vec(&credentials).unwrap()).unwrap();
    let authorized = fixture
        .tool("jobs", "list_jobs", json!({"limit": 100}))
        .await;
    assert_eq!(data(&authorized)["jobs"].as_array().unwrap().len(), 13);
    assert_eq!(
        data(&authorized)["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|job| job["job_type"] == "remote_enrichment")
            .count(),
        12
    );
    let authorized_readiness = fixture.tool("jobs", "service_status", json!({})).await;
    assert_eq!(data(&authorized_readiness)["jobs"]["sampled"], 10);
    assert_eq!(data(&authorized_readiness)["jobs"]["counts"]["failed"], 10);
    assert_eq!(
        data(&authorized_readiness)["jobs"]["readiness"]["state"],
        "partial"
    );
    // Revocation must select the ordinary family before sampling: the hidden
    // newer failures cannot displace the older owned ordinary success.
    for credential in &mut credentials.credentials {
        if credential.instance_id == "enricher" {
            credential
                .capabilities
                .retain(|cap| *cap != Capability::Enrich);
        }
    }
    std::fs::write(&path, serde_json::to_vec(&credentials).unwrap()).unwrap();
    let listed = fixture
        .tool("jobs", "list_jobs", json!({"limit": 100}))
        .await;
    let jobs = data(&listed)["jobs"].as_array().unwrap();
    assert_eq!(
        jobs.len(),
        1,
        "Jobs-only list must omit conversion and rollback: {listed}"
    );
    assert_eq!(jobs[0]["id"], upload.job_id);
    let readiness = fixture.tool("jobs", "service_status", json!({})).await;
    assert_eq!(data(&readiness)["jobs"]["sampled"], 1);
    assert_eq!(data(&readiness)["jobs"]["counts"]["completed"], 1);
    assert!(data(&readiness)["jobs"]["counts"]["failed"].is_null());
    assert_eq!(data(&readiness)["jobs"]["readiness"]["state"], "ready");
    let ordinary_jobs = data(&listed)["jobs"].clone();
    let ordinary_readiness = data(&readiness)["jobs"].clone();
    db.query(
        "UPDATE processing_job SET remote_phase = 'hidden-update', updated_at = time::now() WHERE remote_input.enrichment IS NOT NONE AND remote_input.enrichment IS NOT NULL",
    )
    .await
    .unwrap()
    .check()
    .unwrap();
    let relisted = fixture
        .tool("jobs", "list_jobs", json!({"limit": 100}))
        .await;
    assert_eq!(data(&relisted)["jobs"], ordinary_jobs);
    let rereadiness = fixture.tool("jobs", "service_status", json!({})).await;
    assert_eq!(data(&rereadiness)["jobs"], ordinary_readiness);
    for generic in ["get_job", "cancel_job", "resume_job"] {
        assert_eq!(
            error(&fixture.tool("jobs", generic, json!({"id":id})).await),
            "forbidden"
        );
    }
    for mutation in [
        "SET remote_input.enrichment = NULL",
        "UNSET remote_input.enrichment",
    ] {
        db.query(format!(
            "UPDATE processing_job {mutation} WHERE remote_instance_id = 'enricher' AND id != $ordinary"
        ))
        .bind(("ordinary", graphrag_db::parse_record_id(&upload.job_id, Some("processing_job")).unwrap()))
        .await
        .unwrap()
        .check()
        .unwrap();
        let list = fixture
            .tool("jobs", "list_jobs", json!({"limit": 100}))
            .await;
        assert_eq!(
            data(&list)["jobs"],
            ordinary_jobs,
            "Missing/null saved input must not erase durable enrichment authority"
        );
        let status = fixture.tool("jobs", "service_status", json!({})).await;
        assert_eq!(data(&status)["jobs"], ordinary_readiness);
        for generic in ["get_job", "cancel_job", "resume_job"] {
            assert_eq!(
                error(&fixture.tool("jobs", generic, json!({"id":id})).await),
                "forbidden"
            );
        }
        assert!(
            repo.get_remote_upload_job("enricher", id.as_str().unwrap())
                .await
                .is_err(),
            "Missing enrichment input must not decode as an executable ordinary upload"
        );
    }
    // Re-grant Enrich to the owner only after Jobs-only visibility has been
    // checked. The matrix below corrupts the whole private envelope while the
    // immutable admission family remains true.
    for credential in &mut credentials.credentials {
        if credential.instance_id == "enricher" {
            credential.capabilities.push(Capability::Enrich);
        }
    }
    std::fs::write(&path, serde_json::to_vec(&credentials).unwrap()).unwrap();
    let enrichment_job =
        graphrag_db::parse_record_id(id.as_str().unwrap(), Some("processing_job")).unwrap();
    let ordinary_job =
        graphrag_db::parse_record_id(&upload.job_id, Some("processing_job")).unwrap();
    for mutation in [
        "UNSET remote_input",
        "SET remote_input.enrichment = NULL",
        "SET remote_input = { broken: true }",
    ] {
        db.query(format!("UPDATE $job {mutation}"))
            .bind(("job", enrichment_job.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        for retained_result in [Some(json!({"durable": "checkpoint"})), None] {
            db.query(
                "UPDATE $job SET status = 'completed', remote_result = $result, remote_cancel_requested = false, updated_at = time::now()",
            )
            .bind(("job", enrichment_job.clone()))
            .bind(("result", retained_result.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
            for alias in ["get_source_enrichment_job", "cancel_source_enrichment_job"] {
                let response = fixture.tool("jobs", alias, json!({"id":id})).await;
                assert!(error(&response).is_null(), "{mutation} {alias}: {response}");
                assert_eq!(data(&response)["status"], "completed");
                assert_eq!(
                    &data(&response)["result"],
                    retained_result.as_ref().unwrap_or(&Value::Null)
                );
            }
        }
        db.query(
            "UPDATE $job SET status = 'failed', remote_cancel_requested = false, updated_at = time::now()",
        )
        .bind(("job", enrichment_job.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
        for alias in ["read_source_enrichment_plan", "retry_source_enrichment_job"] {
            assert_eq!(
                error(&fixture.tool("jobs", alias, json!({"id":id})).await),
                "invalid_input",
                "{mutation} {alias}"
            );
        }
        let unchanged = repo
            .get_remote_upload_job_status("enricher", id.as_str().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(unchanged.job.status, "failed");
        assert!(!unchanged.cancel_requested);
        // Hold a synthetic active lease so background workers cannot claim or
        // execute the damaged input while the owner requests cancellation.
        db.query(
            "UPDATE $job SET status = 'running', remote_service_epoch = 'held-fixture', remote_worker_token = 'held-fixture', remote_cancel_requested = false",
        )
        .bind(("job", enrichment_job.clone()))
        .await
        .unwrap()
        .check()
        .unwrap();
        let cancelled = fixture
            .tool("jobs", "cancel_source_enrichment_job", json!({"id":id}))
            .await;
        assert!(error(&cancelled).is_null(), "{mutation}: {cancelled}");
        let persisted = repo
            .get_remote_upload_job_status("enricher", id.as_str().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert!(persisted.cancel_requested);
        assert_eq!(persisted.job.status, "running");
    }
    for alias in [
        "get_source_enrichment_job",
        "cancel_source_enrichment_job",
        "retry_source_enrichment_job",
        "read_source_enrichment_plan",
    ] {
        assert_eq!(
            error(&fixture.tool("foreign", alias, json!({"id":id})).await),
            "not_found",
            "foreign owner must not pass {alias}"
        );
        assert_eq!(
            error(
                &fixture
                    .tool("jobs", alias, json!({"id":upload.job_id}))
                    .await
            ),
            "not_found",
            "ordinary upload must not pass {alias}"
        );
    }
    let ordinary = repo
        .get_remote_upload_job_status("enricher", &upload.job_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ordinary.job.id, Some(ordinary_job));
    assert!(!ordinary.cancel_requested);
    // A privileged legacy/corruption setup can leave no immutable marker at
    // all. It is neither an ordinary nor an enrichment family: Enrich aliases
    // fail closed, and Jobs-only generic control must keep it hidden.
    db.query(
        "REMOVE FIELD remote_enrichment_job ON processing_job; \
         UPDATE processing_job UNSET remote_enrichment_job",
    )
    .await
    .unwrap()
    .check()
    .unwrap();
    assert_eq!(
        error(
            &fixture
                .tool("jobs", "get_source_enrichment_job", json!({"id":id}))
                .await
        ),
        "not_found"
    );
    for credential in &mut credentials.credentials {
        if credential.instance_id == "enricher" {
            credential
                .capabilities
                .retain(|capability| *capability != Capability::Enrich);
        }
    }
    std::fs::write(&path, serde_json::to_vec(&credentials).unwrap()).unwrap();
    assert_eq!(
        error(&fixture.tool("jobs", "get_job", json!({"id":id})).await),
        "forbidden"
    );
    fixture.stop().await;
}
