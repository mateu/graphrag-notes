use crate::{credentials::CredentialError, tools::ToolService, CredentialFile};
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json, Router,
};
use graphrag_application::RemoteApplicationOperations;
use rmcp::transport::streamable_http_server::{
    session::never::NeverSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
};
use thiserror::Error;
use tokio::{
    net::TcpListener,
    sync::{Notify, Semaphore},
};
use tokio_util::sync::CancellationToken;
use tower_http::limit::RequestBodyLimitLayer;

#[derive(Debug, Clone)]
pub struct ServiceOptions {
    pub listen: SocketAddr,
    pub credentials_file: PathBuf,
    /// Acknowledges encryption supplied by a reverse proxy or private tunnel.
    pub external_encryption: bool,
    /// Exact hostnames or authorities; empty selects loopback defaults only.
    pub allowed_hosts: Vec<String>,
    pub max_request_body_bytes: usize,
    pub max_concurrent_requests: usize,
}

impl Default for ServiceOptions {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([127, 0, 0, 1], 3000)),
            credentials_file: PathBuf::new(),
            external_encryption: false,
            allowed_hosts: Vec::new(),
            max_request_body_bytes: 128 * 1024,
            max_concurrent_requests: 8,
        }
    }
}

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("{0}")]
    Configuration(&'static str),
    #[error(transparent)]
    Credentials(#[from] CredentialError),
    #[error("Cannot bind the service listener; check its address and port.")]
    Bind,
    #[error("The HTTP service stopped unexpectedly.")]
    Transport,
}

impl ServiceOptions {
    /// Validate security policy before opening the corpus or loading providers.
    pub fn validate(&self) -> Result<(), ServiceError> {
        self.validate_bind(self.listen.ip())?;
        if self.credentials_file.as_os_str().is_empty() {
            return Err(ServiceError::Configuration(
                "Provide a private credential file.",
            ));
        }
        if !(1024..=1024 * 1024).contains(&self.max_request_body_bytes) {
            return Err(ServiceError::Configuration(
                "Request body limit must be between 1024 and 1048576 bytes.",
            ));
        }
        if !(1..=64).contains(&self.max_concurrent_requests) {
            return Err(ServiceError::Configuration(
                "Concurrent request limit must be between 1 and 64.",
            ));
        }
        if self.allowed_hosts.len() > 32 || self.allowed_hosts.iter().any(|host| !valid_host(host))
        {
            return Err(ServiceError::Configuration("Allowed hosts must be exact hostnames or host:port authorities, without schemes, paths or wildcards."));
        }
        CredentialFile::load_private(&self.credentials_file)?;
        Ok(())
    }

    fn validate_bind(&self, ip: IpAddr) -> Result<(), ServiceError> {
        if !ip.is_loopback() && (!self.external_encryption || self.allowed_hosts.is_empty()) {
            return Err(ServiceError::Configuration("Non-loopback binding requires external encryption and explicit allowed hosts. Use an encrypted reverse proxy, SSH or a Tailscale tunnel."));
        }
        Ok(())
    }

    fn hosts(&self) -> Vec<String> {
        if self.allowed_hosts.is_empty() {
            vec!["localhost".into(), "127.0.0.1".into(), "[::1]".into()]
        } else {
            self.allowed_hosts.clone()
        }
    }
}

fn valid_host(host: &str) -> bool {
    if host.is_empty()
        || host.len() > 253
        || host.trim() != host
        || host
            .bytes()
            .any(|b| !b.is_ascii_graphic() || b"/@*?#\\".contains(&b))
    {
        return false;
    }
    if host.parse::<IpAddr>().is_ok() {
        return true;
    }
    let Ok(authority) = host.parse::<axum::http::uri::Authority>() else {
        return false;
    };
    !authority.host().is_empty()
        && authority
            .host()
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b".-:[]".contains(&b))
        && (!host.contains(':') || authority.port_u16().is_some() || host.ends_with(']'))
}

#[derive(Clone)]
struct AuthState {
    credentials_file: PathBuf,
    requests: Arc<Semaphore>,
    shutdown: CancellationToken,
}

async fn authenticate(
    State(state): State<AuthState>,
    mut request: Request,
    next: Next,
) -> Response {
    if state.shutdown.is_cancelled() {
        return http_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "service_unavailable",
            "The service is shutting down.",
        );
    }
    if request
        .headers()
        .get_all(header::AUTHORIZATION)
        .iter()
        .count()
        != 1
    {
        return unauthorized();
    }
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, token)| token.to_owned());
    let Some(token) = token else {
        return unauthorized();
    };
    let Ok(_permit) = state.requests.clone().try_acquire_owned() else {
        return http_failure(
            StatusCode::TOO_MANY_REQUESTS,
            "busy",
            "The service is busy; retry shortly.",
        );
    };
    let path = state.credentials_file.clone();
    let principal = tokio::task::spawn_blocking(move || {
        CredentialFile::load_private(&path).map(|credentials| credentials.authenticate(&token))
    })
    .await;
    let principal = match principal {
        Ok(Ok(Some(principal))) => principal,
        Ok(Ok(None)) => return unauthorized(),
        _ => {
            return http_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "credentials_unavailable",
                "The service credential file is unavailable or invalid; contact its owner.",
            )
        }
    };
    request.extensions_mut().insert(principal);
    // Only this boundary needs the bearer. SDK request contexts and diagnostic
    // logging receive trusted identity without retaining the raw credential.
    request.headers_mut().remove(header::AUTHORIZATION);
    next.run(request).await
}

fn unauthorized() -> Response {
    let mut response = http_failure(
        StatusCode::UNAUTHORIZED,
        "unauthorized",
        "A valid instance bearer token is required.",
    );
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        axum::http::HeaderValue::from_static("Bearer realm=\"graphrag-notes\""),
    );
    response
}

fn http_failure(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"schema_version":1,"error":{"code":code,"message":message}})),
    )
        .into_response()
}

/// Detached capture tasks retain their application and write permit until done.
#[derive(Default)]
pub(crate) struct Writes {
    state: Mutex<WriteState>,
    idle: Notify,
}

#[derive(Default)]
struct WriteState {
    active: usize,
    closed: bool,
}

impl Writes {
    pub(crate) fn start(self: &Arc<Self>) -> Option<WriteGuard> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.closed {
            return None;
        }
        state.active += 1;
        Some(WriteGuard(Arc::clone(self)))
    }
    async fn drain(&self) {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .closed = true;
        loop {
            let notified = self.idle.notified();
            if self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .active
                == 0
            {
                return;
            }
            notified.await;
        }
    }
}

pub(crate) struct WriteGuard(Arc<Writes>);
impl Drop for WriteGuard {
    fn drop(&mut self) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.active -= 1;
        if state.active == 0 {
            self.0.idle.notify_one();
        }
    }
}

pub async fn run(
    application: Arc<dyn RemoteApplicationOperations>,
    options: ServiceOptions,
) -> Result<(), ServiceError> {
    options.validate()?;
    let listener = TcpListener::bind(options.listen)
        .await
        .map_err(|_| ServiceError::Bind)?;
    let shutdown = CancellationToken::new();
    let trigger = shutdown.clone();
    let signal = tokio::spawn(async move {
        #[cfg(unix)]
        if let Ok(mut terminate) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => trigger.cancel(),
                _ = terminate.recv() => trigger.cancel(),
            }
            return;
        }
        if tokio::signal::ctrl_c().await.is_ok() {
            trigger.cancel();
        }
    });
    let result = serve(listener, application, options, shutdown).await;
    signal.abort();
    result
}

pub async fn serve(
    listener: TcpListener,
    application: Arc<dyn RemoteApplicationOperations>,
    options: ServiceOptions,
    shutdown: CancellationToken,
) -> Result<(), ServiceError> {
    options.validate()?;
    options.validate_bind(listener.local_addr().map_err(|_| ServiceError::Bind)?.ip())?;
    let writes = Arc::new(Writes::default());
    let handlers = ToolService::new(
        application,
        Arc::clone(&writes),
        options.max_concurrent_requests,
    );
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_allowed_hosts(options.hosts())
        .enforce_origin_validation()
        .with_max_request_body_bytes(options.max_request_body_bytes)
        .with_cancellation_token(shutdown.clone());
    let service = StreamableHttpService::new(
        move || Ok(handlers.clone()),
        Arc::new(NeverSessionManager::default()),
        config,
    );
    let auth = AuthState {
        credentials_file: options.credentials_file,
        requests: Arc::new(Semaphore::new(options.max_concurrent_requests)),
        shutdown: shutdown.clone(),
    };
    let router = Router::new()
        .nest_service("/mcp", service)
        .layer(RequestBodyLimitLayer::new(options.max_request_body_bytes))
        .layer(middleware::from_fn_with_state(auth, authenticate));
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await;
    writes.drain().await;
    result.map_err(|_| ServiceError::Transport)
}

#[cfg(test)]
mod tests {
    use super::valid_host;

    #[test]
    fn exact_host_policy_refuses_wildcards_paths_and_invalid_ports() {
        for host in [
            "localhost",
            "notes.example.test:443",
            "[::1]",
            "::1",
            "[::1]:3000",
        ] {
            assert!(valid_host(host), "{host}");
        }
        for host in [
            "",
            "*",
            "*.example.test",
            "https://notes.example.test",
            "notes.test/path",
            "user@notes.test",
            " notes.test",
            "notes.test:65536",
            "[::1]:65536",
            "notes.test:*",
        ] {
            assert!(!valid_host(host), "{host}");
        }
    }
}
