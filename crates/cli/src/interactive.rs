//! A terminal client over shared application operations and local adapters.

use crate::commands::workspace::{self, WorkspaceContext};
pub(crate) use crate::output::safe_text;
use anyhow::{bail, Context, Result};
use graphrag_application::{
    ActionCancellation, GraphPolicy, ProposalAction, ProposalCard, ProposalDecisionRequest,
    RecordRef, RecordSummary, Scope, SearchMode, SearchRequest,
};
use graphrag_core::ProposedEdgeStatus;
use graphrag_db::repository::RecordInspection;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::MemHistory;
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, Editor, Helper};
use std::future::Future;
use std::io::{self, BufRead, IsTerminal, Write};

const COMMANDS: &[&str] = &[
    "back",
    "capture",
    "copy",
    "help",
    "history",
    "inspect",
    "keyword",
    "next",
    "open",
    "prev",
    "proposals",
    "quit",
    "recent",
    "review",
    "search",
    "select",
    "stats",
    "status",
];
const HISTORY_LIMIT: usize = 100;
const SNAPSHOT_HISTORY_LIMIT: usize = 10;

#[derive(Clone)]
enum Snapshot {
    Records(Vec<RecordSummary>),
    Proposals(Vec<ProposalCard>),
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::Records(Vec::new())
    }
}

impl Snapshot {
    fn len(&self) -> usize {
        match self {
            Self::Records(records) => records.len(),
            Self::Proposals(proposals) => proposals.len(),
        }
    }

    fn id(&self, index: usize) -> Option<&str> {
        match self {
            Self::Records(records) => records.get(index).map(|record| record.id.as_str()),
            Self::Proposals(proposals) => proposals.get(index).map(|card| card.id.as_str()),
        }
    }

    fn index(&self, argument: &str, selected: Option<usize>) -> Result<usize> {
        let index = if argument.is_empty() {
            selected.context("No selection. Use recent, keyword, search, or proposals, then select a result number.")?
        } else if let Ok(position) = argument.parse::<usize>() {
            position
                .checked_sub(1)
                .context("Result positions start at 1.")?
        } else {
            (0..self.len()).find(|index| self.id(*index) == Some(argument))
                .context("That ID is not in the displayed snapshot. Run recent, keyword, search, or proposals again.")?
        };
        if index >= self.len() {
            bail!(
                "No result at position {}. Choose a displayed result number.",
                index + 1
            );
        }
        Ok(index)
    }

    fn record(&self, index: usize) -> Result<RecordRef> {
        let Self::Records(records) = self else {
            bail!("A proposal is selected. Use review, or search/recent to select a note or chat record.");
        };
        let record = records.get(index).context("No record selected.")?;
        let revision = record.revision.clone().context("This result changed while it was displayed. Run recent or search again before selecting it.")?;
        Ok(RecordRef {
            id: record.id.clone(),
            revision: Some(revision),
        })
    }
}

#[derive(Default)]
struct Session {
    snapshot: Snapshot,
    selected: Option<usize>,
    previous: Vec<Snapshot>,
    history: Vec<String>,
    review: Option<ReviewState>,
}

enum ReviewState {
    Choose(ProposalCard),
    Reason {
        card: ProposalCard,
        action: ProposalAction,
    },
    Confirm {
        card: ProposalCard,
        action: ProposalAction,
        reason: Option<String>,
    },
}

impl ReviewState {
    fn prompt(&self) -> &'static str {
        match self {
            Self::Choose(_) => "review> ",
            Self::Reason { .. } => "reason (optional)> ",
            Self::Confirm { .. } => "confirm yes/no> ",
        }
    }
}

impl Session {
    fn replace(&mut self, snapshot: Snapshot) {
        if self.snapshot.len() > 0 {
            self.previous.push(self.snapshot.clone());
            if self.previous.len() > SNAPSHOT_HISTORY_LIMIT {
                self.previous.remove(0);
            }
        }
        self.snapshot = snapshot;
        self.selected = None;
    }

    fn remember(&mut self, line: &str) {
        if !line.trim().is_empty() && self.history.last().map(String::as_str) != Some(line) {
            self.history.push(line.into());
            if self.history.len() > HISTORY_LIMIT {
                self.history.remove(0);
            }
        }
    }
}

struct WorkspaceHelper {
    ids: Vec<String>,
    review_choices: Option<&'static [&'static str]>,
}

impl Completer for WorkspaceHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        position: usize,
        _context: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let prefix = line.get(..position).unwrap_or(line);
        let (start, candidates) = if let Some(choices) = self.review_choices {
            (
                0,
                choices
                    .iter()
                    .filter(|choice| choice.starts_with(prefix))
                    .map(|choice| (*choice).into())
                    .collect(),
            )
        } else {
            completion_candidates(prefix, &self.ids)
        };
        Ok((
            start,
            candidates
                .into_iter()
                .map(|value| Pair {
                    display: value.clone(),
                    replacement: value,
                })
                .collect(),
        ))
    }
}

impl Hinter for WorkspaceHelper {
    type Hint = String;
}
impl Highlighter for WorkspaceHelper {}
impl Validator for WorkspaceHelper {}
impl Helper for WorkspaceHelper {}

fn completion_candidates(prefix: &str, ids: &[String]) -> (usize, Vec<String>) {
    let Some((command, argument)) = prefix.split_once(char::is_whitespace) else {
        return (
            0,
            COMMANDS
                .iter()
                .filter(|command| command.starts_with(prefix))
                .map(|command| (*command).into())
                .collect(),
        );
    };
    if !matches!(command, "select" | "inspect" | "open" | "copy" | "review") {
        return (prefix.len(), Vec::new());
    }
    let argument = argument.trim_start();
    let start = prefix.len() - argument.len();
    let candidates = ids
        .iter()
        .filter(|id| {
            !id.chars().any(|character| {
                character.is_control() || matches!(character, '\u{2028}' | '\u{2029}')
            })
        })
        .cloned()
        .chain((1..=ids.len()).map(|position| position.to_string()))
        .filter(|candidate| candidate.starts_with(argument))
        .collect();
    (start, candidates)
}

enum Input {
    Terminal(Box<Editor<WorkspaceHelper, MemHistory>>),
    Pipe,
}
enum InputLine {
    Line(String),
    Interrupted,
    Eof,
}

impl Input {
    fn new() -> Result<Self> {
        if io::stdin().is_terminal() && io::stdout().is_terminal() {
            let config = Config::builder()
                .completion_type(CompletionType::List)
                .max_history_size(HISTORY_LIMIT)?
                .build();
            let history = MemHistory::with_config(&config);
            let mut editor = Editor::with_history(config, history)?;
            editor.set_helper(Some(WorkspaceHelper {
                ids: Vec::new(),
                review_choices: None,
            }));
            Ok(Self::Terminal(Box::new(editor)))
        } else {
            Ok(Self::Pipe)
        }
    }

    fn read(&mut self, session: &Session) -> Result<InputLine> {
        let prompt = session
            .review
            .as_ref()
            .map(ReviewState::prompt)
            .unwrap_or("graphrag> ");
        match self {
            Self::Terminal(editor) => {
                if let Some(helper) = editor.helper_mut() {
                    helper.ids = (0..session.snapshot.len())
                        .filter_map(|index| session.snapshot.id(index).map(str::to_owned))
                        .collect();
                    helper.review_choices = session.review.as_ref().map(|state| match state {
                        ReviewState::Choose(_) => &["accept", "reject", "undo", "skip"][..],
                        ReviewState::Reason { .. } => &[":cancel"][..],
                        ReviewState::Confirm { .. } => &["yes", "no"][..],
                    });
                }
                match editor.readline(prompt) {
                    Ok(line) => {
                        editor.add_history_entry(&line)?;
                        Ok(InputLine::Line(line))
                    }
                    Err(ReadlineError::Interrupted) => Ok(InputLine::Interrupted),
                    Err(ReadlineError::Eof) => Ok(InputLine::Eof),
                    Err(error) => Err(error.into()),
                }
            }
            Self::Pipe => {
                print!("{prompt}");
                io::stdout().flush()?;
                let mut line = String::new();
                match io::stdin().lock().read_line(&mut line) {
                    Ok(0) => Ok(InputLine::Eof),
                    Ok(_) => Ok(InputLine::Line(line.trim_end_matches(['\r', '\n']).into())),
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                        Ok(InputLine::Interrupted)
                    }
                    Err(error) => Err(error.into()),
                }
            }
        }
    }
}

/// Own one embedded application for the entire session. Every failed action
/// returns to the prompt; only input/output failures terminate the session.
pub(crate) async fn cmd_interactive(context: WorkspaceContext) -> Result<()> {
    let mut input = Input::new()?;
    let mut session = Session::default();
    println!("GraphRAG Notes - Terminal Workspace");
    println!("Browsing and keyword search work offline. Type help for commands.");
    println!("History stays in this session. Ctrl-C clears input or requests a safe action stop.");
    loop {
        let line = match input.read(&session)? {
            InputLine::Line(line) => line,
            InputLine::Interrupted => {
                if session.review.take().is_some() {
                    println!("Review cancelled; no decision saved.");
                } else {
                    println!("Input cancelled.");
                }
                continue;
            }
            InputLine::Eof => break,
        };
        let line = line.trim();
        if session.review.is_some() {
            session.remember(line);
            match review_input(&mut session, line) {
                Ok(Some(request)) => {
                    let action = async {
                        let card = context.application.decide_proposal(request).await?;
                        render_proposal(&card);
                        println!("Run proposals again to view the updated inbox.");
                        Ok(())
                    };
                    if let Err(error) = run_action(action, ActionCancellation::new(), true).await {
                        println!("Error: {}", safe_text(&format!("{error:#}"), false));
                    }
                }
                Ok(None) => {}
                Err(error) => println!("Error: {}", safe_text(&format!("{error:#}"), false)),
            }
            println!();
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let (command, argument) = command_parts(line);
        if matches!(command, "quit" | "q" | "exit") {
            break;
        }
        session.remember(line);
        let cancellation = ActionCancellation::new();
        let waits_for_completion = matches!(command, "capture" | "add" | "a" | "open");
        let action = execute(
            &context,
            &mut session,
            command,
            argument,
            cancellation.clone(),
        );
        let result = run_action(action, cancellation, waits_for_completion).await;
        if let Err(error) = result {
            println!("Error: {}", safe_text(&format!("{error:#}"), false));
        }
        println!();
    }
    println!("Goodbye!");
    Ok(())
}

async fn run_action(
    action: impl Future<Output = Result<()>>,
    cancellation: ActionCancellation,
    waits_for_completion: bool,
) -> Result<()> {
    tokio::pin!(action);
    tokio::select! {
            result = &mut action => result,
            signal = tokio::signal::ctrl_c() => {
                if let Err(error) = signal {
                    eprintln!("Cannot listen for Ctrl-C: {error}");
                    action.await
                } else {
                    cancellation.cancel();
                    if waits_for_completion {
                        println!("Interrupt requested; waiting for this operation to finish safely.");
                        action.await
                    } else {
                        println!("Action cancelled. The displayed snapshot is unchanged.");
                        Ok(())
                    }
                }
            }
    }
}

fn review_input(session: &mut Session, line: &str) -> Result<Option<ProposalDecisionRequest>> {
    let state = session
        .review
        .take()
        .context("No proposal review is active.")?;
    if matches!(line, ":cancel" | ":skip") {
        println!("Review skipped; no changes saved.");
        return Ok(None);
    }
    match state {
        ReviewState::Choose(card) => {
            let action = match line.to_ascii_lowercase().as_str() {
                "accept" | "a" => ProposalAction::Accept,
                "reject" | "r" => ProposalAction::Reject,
                "undo" | "u" => ProposalAction::Undo,
                "skip" | "s" | "cancel" | "q" | "quit" => {
                    println!("Review skipped; no changes saved.");
                    return Ok(None);
                }
                _ => {
                    session.review = Some(ReviewState::Choose(card));
                    bail!("Choose accept, reject, undo, or skip. Nothing has been changed.");
                }
            };
            session.review = Some(ReviewState::Reason { card, action });
            println!("Enter an optional audit reason, or press Enter. Type :cancel to skip.");
        }
        ReviewState::Reason { card, action } => {
            let reason = (!line.is_empty()).then(|| line.to_owned());
            println!(
                "Confirm {} for {}: type yes to apply, or no to cancel.",
                action_word(action),
                safe_text(&card.id, false)
            );
            session.review = Some(ReviewState::Confirm {
                card,
                action,
                reason,
            });
        }
        ReviewState::Confirm {
            card,
            action,
            reason,
        } => {
            if line.eq_ignore_ascii_case("yes") {
                return Ok(Some(ProposalDecisionRequest {
                    id: card.id,
                    revision: card.revision,
                    action,
                    reason: Some(reason.unwrap_or_else(|| {
                        format!("explicit interactive review {}", decision_word(action))
                    })),
                    confirmed: true,
                    reviewer: "cli interactive review".into(),
                }));
            }
            if line.is_empty()
                || ["no", "n", "cancel", "skip", "s"]
                    .iter()
                    .any(|value| line.eq_ignore_ascii_case(value))
            {
                println!("Decision cancelled; no changes saved.");
            } else {
                session.review = Some(ReviewState::Confirm {
                    card,
                    action,
                    reason,
                });
                bail!("Type yes to confirm or no to cancel. Nothing has been changed.");
            }
        }
    }
    Ok(None)
}

fn action_word(action: ProposalAction) -> &'static str {
    match action {
        ProposalAction::Accept => "acceptance",
        ProposalAction::Reject => "rejection",
        ProposalAction::Undo => "undo",
    }
}

fn decision_word(action: ProposalAction) -> &'static str {
    match action {
        ProposalAction::Accept => "accept",
        ProposalAction::Reject => "reject",
        ProposalAction::Undo => "undo",
    }
}

fn command_parts(line: &str) -> (&str, &str) {
    line.split_once(char::is_whitespace)
        .map(|(command, argument)| (command, argument.trim()))
        .unwrap_or((line, ""))
}

fn limit(argument: &str, default: usize) -> Result<usize> {
    let limit = if argument.is_empty() {
        default
    } else {
        argument
            .parse()
            .context("The limit must be an integer from 1 to 200.")?
    };
    if !(1..=200).contains(&limit) {
        bail!("The limit must be between 1 and 200.");
    }
    Ok(limit)
}

async fn execute(
    context: &WorkspaceContext,
    session: &mut Session,
    command: &str,
    argument: &str,
    cancellation: ActionCancellation,
) -> Result<()> {
    let application = &context.application;
    match command {
        "recent" | "list" | "l" => {
            let records = application
                .recent(limit(
                    argument,
                    context.config.search.default_limit.clamp(1, 200),
                )?)
                .await?;
            session.replace(Snapshot::Records(records));
            render_snapshot(&session.snapshot);
        }
        "keyword" | "search" | "s" => {
            if argument.is_empty() {
                bail!("Usage: {command} <query>");
            }
            let hybrid = command != "keyword";
            if hybrid {
                println!(
                    "Working: hybrid search (Ctrl-C to cancel). Use keyword for offline retrieval."
                );
            }
            let records = application
                .search(
                    SearchRequest {
                        query: argument.into(),
                        mode: if hybrid {
                            SearchMode::Hybrid
                        } else {
                            SearchMode::Keyword
                        },
                        scope: Scope::All,
                        limit: context.config.search.default_limit.clamp(1, 200),
                        graph: if hybrid {
                            GraphPolicy::Auto
                        } else {
                            GraphPolicy::Off
                        },
                        since_days: None,
                        source_uri: None,
                    },
                    cancellation,
                )
                .await?;
            session.replace(Snapshot::Records(records));
            render_snapshot(&session.snapshot);
        }
        "select" | "inspect" => {
            let index = session.snapshot.index(argument, session.selected)?;
            show_selection(context, session, index).await?;
        }
        "next" | "prev" | "previous" => {
            let index = match (command, session.selected) {
                ("next", Some(index)) => index.checked_add(1).context("No next result.")?,
                ("next", None) => 0,
                (_, Some(index)) => index
                    .checked_sub(1)
                    .context("Already at the first result.")?,
                (_, None) => {
                    bail!("No selection. Use select <number> first.");
                }
            };
            if index >= session.snapshot.len() {
                bail!("Already at the last result.");
            }
            show_selection(context, session, index).await?;
        }
        "back" => {
            if session.selected.take().is_none() {
                session.snapshot = session
                    .previous
                    .pop()
                    .context("No previous result snapshot.")?;
            }
            render_snapshot(&session.snapshot);
        }
        "open" | "copy" => {
            let index = session.snapshot.index(argument, session.selected)?;
            let reference = session.snapshot.record(index)?;
            if command == "open" {
                workspace::open(context, reference).await?;
            } else {
                workspace::copy(context, reference).await?;
            }
        }
        "capture" | "add" | "a" => {
            if argument.is_empty() {
                bail!("Usage: capture <content> or capture --editor. Capture never consumes the workspace command stream.");
            }
            println!("Working: capture. Private recovery input is retained until save succeeds.");
            workspace::capture(context, argument, cancellation).await?;
        }
        "status" => {
            let sources = application.source_status(limit(argument, 20)?).await?;
            if sources.is_empty() {
                println!("No sources yet.");
            }
            for source in sources {
                println!(
                    "{} [{}]",
                    safe_text(
                        source.title.as_deref().unwrap_or("(untitled source)"),
                        false
                    ),
                    safe_text(&source.status, false)
                );
                println!("  ID: {}", safe_text(&source.id, false));
                if let Some(uri) = source.uri {
                    println!("  Source: {}", safe_text(&uri, false));
                }
                println!(
                    "  Generation: {}; last successful: {}",
                    source.generation, source.successful_generation
                );
                if let Some(error) = source.last_error {
                    println!("  Last error: {}", safe_text(&error, false));
                }
            }
        }
        "proposals" | "garden" | "g" => {
            let (all, argument) = if argument == "all" {
                (true, "")
            } else if let Some(argument) = argument.strip_prefix("all ") {
                (true, argument.trim())
            } else {
                (false, argument)
            };
            let proposals = application
                .proposals(
                    if all {
                        None
                    } else {
                        Some(ProposedEdgeStatus::Pending)
                    },
                    limit(argument, 20)?,
                )
                .await?;
            session.replace(Snapshot::Proposals(proposals));
            render_snapshot(&session.snapshot);
        }
        "review" => {
            let index = session.snapshot.index(argument, session.selected)?;
            let Snapshot::Proposals(proposals) = &session.snapshot else {
                bail!("Run proposals first, then select a proposal and use review.");
            };
            let card = &proposals[index];
            let current = application.proposal(&card.id).await?;
            if current.revision != card.revision {
                bail!(
                    "The proposal or an endpoint changed. Run proposals again before reviewing it."
                );
            }
            render_proposal(&current);
            session.review = Some(ReviewState::Choose(current));
            println!("Choose accept/a, reject/r, undo/u, or skip/s. Every decision requires explicit yes confirmation.");
        }
        "stats" => {
            let stats = application.stats().await?;
            println!(
                "Notes: {}, entities: {}, sources: {}, edges: {}, conversations: {}, messages: {}",
                stats.note_count,
                stats.entity_count,
                stats.source_count,
                stats.edge_count,
                stats.conversation_count,
                stats.message_count
            );
        }
        "history" => {
            for (index, line) in session.history.iter().enumerate() {
                println!("{}. {}", index + 1, safe_text(line, false));
            }
            println!("History is kept only in memory and is discarded on exit.");
        }
        "help" | "h" | "?" => help(),
        _ => bail!(
            "Unknown command: {}. Type help for available commands.",
            safe_text(command, false)
        ),
    }
    Ok(())
}

async fn show_selection(
    context: &WorkspaceContext,
    session: &mut Session,
    index: usize,
) -> Result<()> {
    match &session.snapshot {
        Snapshot::Records(_) => {
            let record = context
                .application
                .inspect(session.snapshot.record(index)?, 2)
                .await?;
            render_inspection(&record);
        }
        Snapshot::Proposals(proposals) => {
            let shown = &proposals[index];
            let current = context.application.proposal(&shown.id).await?;
            if current.revision != shown.revision {
                bail!("The proposal or an endpoint changed. Run proposals again before selecting or reviewing it.");
            }
            render_proposal(&current);
        }
    }
    session.selected = Some(index);
    println!(
        "Selected result {}. Use next, prev, back, open/copy, or review.",
        index + 1
    );
    Ok(())
}

fn render_snapshot(snapshot: &Snapshot) {
    if snapshot.len() == 0 {
        match snapshot {
            Snapshot::Records(_) => println!("No results. Use capture to add a note, or try another query."),
            Snapshot::Proposals(_) => println!("No matching proposals. Use garden scan outside the workspace, or proposals all for review history."),
        }
        return;
    }
    match snapshot {
        Snapshot::Records(records) => {
            for (index, record) in records.iter().enumerate() {
                println!(
                    "{}. [{}] {}",
                    index + 1,
                    safe_text(&record.hit_type, false),
                    title(record.title.as_deref(), &record.content)
                );
                println!("   ID: {}", safe_text(&record.id, false));
                println!(
                    "   Revision: {}",
                    record
                        .revision
                        .as_deref()
                        .unwrap_or("unavailable; repeat search")
                );
                println!("   {}", preview(&record.content, 100));
                for warning in &record.warnings {
                    println!("   Warning: {}", safe_text(warning, false));
                }
            }
        }
        Snapshot::Proposals(proposals) => {
            for (index, card) in proposals.iter().enumerate() {
                println!(
                    "{}. [{}] {} ({:.2})",
                    index + 1,
                    card.status,
                    safe_text(&card.edge_type, false),
                    card.confidence
                );
                println!("   ID: {}", safe_text(&card.id, false));
                println!("   From: {}", safe_text(&card.from.id, false));
                println!("   To: {}", safe_text(&card.to.id, false));
                println!("   {}", preview(&card.reason, 100));
            }
        }
    }
    println!(
        "Select a result number or its full ID. Numbers refer only to this displayed snapshot."
    );
}

fn render_inspection(record: &RecordInspection) {
    println!(
        "[{}] {}",
        safe_text(&record.hit_type, false),
        title(record.title.as_deref(), &record.content)
    );
    println!("ID: {}", safe_text(&record.id, false));
    println!("Revision: {}", record.revision);
    if let Some(uri) = &record.provenance.source_uri {
        println!("Source: {}", safe_text(uri, false));
    }
    if !record.provenance.heading_path.is_empty() {
        println!(
            "Heading: {}",
            safe_text(&record.provenance.heading_path.join(" > "), false)
        );
    }
    println!("\n{}", safe_text(&record.content, true));
    for conversation in &record.conversations {
        println!(
            "Conversation: {} [{}]",
            safe_text(
                conversation
                    .title
                    .as_deref()
                    .unwrap_or("(untitled conversation)"),
                false
            ),
            safe_text(&conversation.id, false)
        );
    }
    if !record.messages.is_empty() {
        println!("Chat context:");
    }
    for message in &record.messages {
        println!(
            "  #{} [{}] {}",
            message.message_index + 1,
            safe_text(&message.role, false),
            safe_text(&message.id, false)
        );
        println!("{}", safe_text(&message.content, true));
    }
    if record.messages_truncated {
        println!("Additional chat context is omitted; use the ordinary inspect command for more neighbors.");
    }
    for warning in &record.warnings {
        println!("Warning: {}", safe_text(warning, false));
    }
}

fn render_proposal(card: &ProposalCard) {
    println!("Proposal: {} [{}]", safe_text(&card.id, false), card.status);
    println!("Revision: {}", card.revision);
    println!(
        "Relationship: {} ({:.2})",
        safe_text(&card.edge_type, false),
        card.confidence
    );
    println!("Reason: {}", safe_text(&card.reason, true));
    println!("Updated at: {}", safe_text(&card.updated_at, false));
    for (label, endpoint) in [("From", &card.from), ("To", &card.to)] {
        println!(
            "{label}: {} [{}]",
            if endpoint.available {
                title(
                    endpoint.title.as_deref(),
                    endpoint.excerpt.as_deref().unwrap_or(""),
                )
            } else {
                "(unavailable note)".into()
            },
            safe_text(&endpoint.id, false)
        );
        if let Some(excerpt) = &endpoint.excerpt {
            println!("{}", safe_text(excerpt, true));
        }
        for warning in &endpoint.warnings {
            println!("Warning: {}", safe_text(warning, false));
        }
    }
    if let Some(reason) = &card.accept_blocked_reason {
        println!("Acceptance blocked: {}", safe_text(reason, false));
    }
    println!("Use review for explicit, audited accept/reject/skip/undo decisions.");
}

fn title(title: Option<&str>, content: &str) -> String {
    let title = title
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            content
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("(untitled)")
        });
    preview(title, 80)
}

fn preview(value: &str, count: usize) -> String {
    let cleaned = safe_text(value, false);
    let mut characters = cleaned.chars();
    let mut result: String = characters.by_ref().take(count).collect();
    if characters.next().is_some() {
        result.push('…');
    }
    result
}

fn help() {
    println!("Commands:");
    println!("  recent [LIMIT]         Recent visible notes (offline)");
    println!("  keyword QUERY          Search notes and chats without providers");
    println!("  search QUERY           Hybrid retrieval; embeddings checked per action");
    println!("  select NUMBER|ID       Inspect a result from the displayed snapshot");
    println!("  inspect [NUMBER|ID]    Inspect the selected record or proposal");
    println!("  next / prev / back     Navigate results or return to the previous list");
    println!("  open [NUMBER|ID]       Open the original local source, revision checked");
    println!("  copy [NUMBER|ID]       Print selected content; stable citation goes to stderr");
    println!("  capture CONTENT        Save recoverably; capture --editor opens an editor");
    println!("  status [LIMIT]         Source generations and last import errors (offline)");
    println!("  proposals [all] [LIMIT] Pending proposals, or include review history");
    println!("  review [NUMBER|ID]     Explicit governed review of a displayed proposal");
    println!("  stats                  Corpus counts (offline)");
    println!("  history                Commands kept only in this session");
    println!("  help / quit            Show help or close the database and exit");
    println!("Terminal keys: Up/Down recall history; Tab completes commands and displayed IDs.");
    println!(
        "Ctrl-C clears input or cancels at a safe boundary. Committed saves still report success."
    );
    println!(
        "A changed/deleted result cannot redirect a selection. Repeat its search to refresh it."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_uses_character_boundaries_and_current_snapshot_only() {
        let ids = vec!["note:東京".into(), "note:two".into()];
        assert_eq!(
            completion_candidates("select note:東", &ids),
            (7, vec!["note:東京".into()])
        );
        assert_eq!(
            completion_candidates("select 2", &ids),
            (7, vec!["2".into()])
        );
        assert!(completion_candidates("capture note:", &ids).1.is_empty());
    }

    #[test]
    fn printed_content_cannot_execute_terminal_controls() {
        assert_eq!(
            safe_text("hello\u{1b}[2J\nworld\u{85}x", false),
            "hello\\u{1b}[2J world\\u{85}x"
        );
        assert_eq!(safe_text("hello\nworld\t!", true), "hello\nworld\t!");
        assert_eq!(preview("日本語", 2), "日本…");
    }

    #[test]
    fn selection_rejects_unavailable_revisions_and_keeps_previous_snapshots_guarded() {
        let record = RecordSummary {
            id: "note:old".into(),
            revision: Some("old-revision".into()),
            hit_type: "note".into(),
            title: None,
            content: "old contents".into(),
            provenance: None,
            score: None,
            warnings: vec![],
        };
        let mut session = Session::default();
        session.replace(Snapshot::Records(vec![record.clone()]));
        session.selected = Some(0);
        assert_eq!(
            session.snapshot.record(0).unwrap().revision.as_deref(),
            Some("old-revision")
        );
        let mut refreshed = record;
        refreshed.revision = None;
        session.replace(Snapshot::Records(vec![refreshed]));
        assert!(session.selected.is_none());
        assert!(session.snapshot.record(0).is_err());
        assert!(session.snapshot.index("0", None).is_err());
        assert!(session.snapshot.index("2", None).is_err());
        assert!(session.snapshot.index("note:other", None).is_err());
        assert_eq!(
            session.previous[0].record(0).unwrap().revision.as_deref(),
            Some("old-revision")
        );
    }
}
