//! DEVELOPMENT ONLY actual service serialization, ADD-only fixture exporter.
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
            credentials: vec![Credential {
                instance_id: "shiva-importer".into(),
                token_sha256: format!("{:x}", Sha256::digest(token("shiva-importer").as_bytes())),
                capabilities: vec![
                    Capability::Read,
                    Capability::Upload,
                    Capability::Jobs,
                    Capability::Enrich,
                ],
            }],
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
                .no_proxy()
                .timeout(Duration::from_secs(30))
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
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        response.json().await.unwrap()
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
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
async fn tool(f: &Fixture, name: &str, args: Value) -> Value {
    let v = f.tool("shiva-importer", name, args).await;
    assert!(data(&v).is_object(), "{v}");
    v
}
async fn completed(f: &Fixture, name: &str, id: &Value) -> Value {
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            let v = tool(f, name, json!({"id":id})).await;
            let status = data(&v)["status"].as_str().unwrap();
            if status == "completed" {
                break v;
            }
            assert!(
                !["failed", "cancelled", "interrupted"].contains(&status),
                "{v}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap()
}
async fn sources(f: &Fixture, id: &Value, key: &str) -> Value {
    let by_id = tool(f, "get_source", json!({"id":id})).await;
    let by_key = tool(f, "get_source", json!({"document_key":key})).await;
    assert_eq!(data(&by_id), data(&by_key));
    json!({"by_id":by_id,"by_key":by_key})
}
#[tokio::test]
async fn export_actual_service_importer_adoption_roundtrip() {
    let repo = Repository::new(init_memory().await.unwrap());
    let f = Fixture::new(application(
        &repo,
        Arc::new(graphrag_agents::DeterministicEmbedder::default()),
    ))
    .await;
    let status = tool(&f, "service_status", json!({})).await;
    assert_eq!(data(&status)["instance_id"], "shiva-importer");
    let mut documents = Vec::new();
    for (slug, raw) in [
        (
            "service-fixture",
            "# Atlas synthetic technical fixture

Atlas uses bounded retry evidence.
"
            .to_string(),
        ),
        (
            "service-multipart",
            "Atlas synthetic technical retry evidence.
"
            .repeat(1700),
        ),
    ] {
        let path = format!("memory/2026-10-07-{slug}.md");
        // ASCII-only synthetic bytes: identical to staged importer's UTF-8 splitting.
        let chunks: Vec<&str> = if raw.len() <= 65536 {
            vec![&raw]
        } else {
            raw.as_bytes()
                .chunks(49152)
                .map(|v| std::str::from_utf8(v).unwrap())
                .collect()
        };
        let mut parts = Vec::new();
        for (index, content) in chunks.iter().enumerate() {
            let part = index + 1;
            let key = format!("openclaw-memory/clawd/main/{path}/part-{part:04}");
            let title = if chunks.len() == 1 {
                format!("2026-10-07-{slug}.md")
            } else {
                format!("2026-10-07-{slug}.md [part {part}/{}]", chunks.len())
            };
            let upload_request = json!({"request_id":format!("synthetic-original-{slug}-{part}"),"document_key":key,
                "content":content,"title":title,"extract_entities":false,
                "provenance":{"uri":format!("openclaw://clawd/main/{path}"),"label":"OpenClaw indexed memory",
                "metadata":{"host":"clawd","agent":"main","source_path":path,"original_sha256":hash(raw.as_bytes()),"part":part.to_string(),"parts":chunks.len().to_string(),"custom":"synthetic-only"}}});
            let upload_admission = tool(&f, "upload_source", upload_request.clone()).await;
            let upload_job = completed(&f, "get_job", &data(&upload_admission)["job_id"]).await;
            let id = data(&upload_admission)["source_id"].clone();
            let opening = sources(&f, &id, &key).await;
            assert_eq!(data(&opening["by_id"])["extract_entities"], false);
            assert_eq!(data(&opening["by_id"])["graph_policy_epoch"], 0);
            let prepare_request = json!({"request_id":format!("synthetic-enrich-{slug}-{part}"),"source_id":id,"revision":data(&opening["by_id"])["revision"]});
            let prepared = tool(&f, "prepare_source_enrichment", prepare_request.clone()).await;
            let enrich_request = json!({"request_id":prepare_request["request_id"],"reviewed":data(&prepared),"confirmed":true});
            let enrich_admission =
                tool(&f, "execute_source_enrichment", enrich_request.clone()).await;
            let enrich_job = completed(
                &f,
                "get_source_enrichment_job",
                &data(&enrich_admission)["job_id"],
            )
            .await;
            let enriched = sources(&f, &id, &key).await;
            let enrich_replay = tool(&f, "execute_source_enrichment", enrich_request.clone()).await;
            assert_eq!(data(&enrich_replay)["replayed"], true);
            assert_eq!(
                data(&enriched["by_id"])["latest_upload_request_id"],
                upload_request["request_id"]
            );
            assert_eq!(data(&enriched["by_id"])["graph_policy_epoch"], 1);
            assert!(
                !data(&enriched["by_id"])["reviewed_enrichment_v1"]["endpoint_revisions"]
                    .as_object()
                    .unwrap()
                    .is_empty()
            );
            let rollback_request = json!({"request_id":format!("synthetic-rollback-{slug}-{part}"),"reviewed":data(&prepared),"confirmed":true,
                "rollback_source_revision":data(&enriched["by_id"])["revision"]});
            let rollback_admission =
                tool(&f, "rollback_source_enrichment", rollback_request.clone()).await;
            let rollback_job = completed(
                &f,
                "get_source_enrichment_job",
                &data(&rollback_admission)["job_id"],
            )
            .await;
            let rolled_back = sources(&f, &id, &key).await;
            assert_eq!(data(&rolled_back["by_id"])["graph_policy_epoch"], 2);
            assert_eq!(
                data(&rolled_back["by_id"])["latest_upload_request_id"],
                upload_request["request_id"]
            );
            parts.push(json!({"upload_request":upload_request,"upload_admission":upload_admission,"upload_job":upload_job,
                "opening":opening,"prepare_request":prepare_request,"prepared":prepared,"enrich_request":enrich_request,
                "enrich_admission":enrich_admission,"enrich_job":enrich_job,"enriched":enriched,"enrich_replay":enrich_replay,
                "rollback_request":rollback_request,"rollback_admission":rollback_admission,"rollback_job":rollback_job,"rolled_back":rolled_back}));
        }
        documents.push(json!({"path":path,"raw":raw,"sha256":hash(raw.as_bytes()),"parts":parts}));
    }
    f.stop().await;
    let export = json!({"fixture_schema_version":1,"synthetic_only":true,"service_status":status,"documents":documents});
    if let Some(directory) = std::env::var_os("GRAPHRAG_ADOPTION_FIXTURE_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("actual-service.json"),
            serde_json::to_vec_pretty(&export).unwrap(),
        )
        .unwrap();
    }
}
