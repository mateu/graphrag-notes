"""Optional synthetic mutations/jobs through actual native client tool calls.

The coordinator supplies calls(client, instance, requests); every operation is
dispatched by the installed OpenClaw runtime or Hermes registry, never HTTP.
"""
from contextlib import contextmanager
import time


EXPLICIT_GRANTS = {
    "writer": ["read", "capture", "edit", "delete", "accept", "reject", "undo"],
    "uploader": ["read", "capture", "upload", "jobs"],
    "foreign-jobs": ["read", "jobs"],
}


@contextmanager
def provider_pause(provider):
    provider.arm()
    try:
        yield
    finally:
        provider.release.set()


def run_extended(calls, provider, require, timeout, evidence):
    # Mutate the retained report as calls happen, so a failed runtime does not
    # erase which actual native operations completed or what was attempted.
    evidence.update(explicit_grants=EXPLICIT_GRANTS, connection_decisions_exercised=False,
        connection_decision_limit="No synthetic connection proposal is seeded; decision grants alone are not decision evidence.",
        source_refresh_exercised=False, entity_extraction_exercised=False,
        restart_during_running_job_exercised=False,
        native_calls=[])

    def call(client, identity, tool, arguments, error=None):
        request = {"tool": tool, "arguments": arguments}
        if error:
            request["expect_error"] = error
        attempt = {"client": client, "instance_id": identity, "tool": tool,
                   "expected_error": error, "status": "attempted"}
        evidence["native_calls"].append(attempt)
        result = calls(client, identity, [request])
        require(len(result["results"]) == 1, "Extended native call returned an invalid result count")
        attempt.update(status="completed", categorized_error=error, catalog=result["catalog"])
        return result["results"][0]

    # Both native clients use an explicitly provisioned writer identity.
    capture = {"request_id": "native-edit-capture-001", "content": "nativemutationatlas synthetic editable note.",
               "title": "Before native edits", "tags": ["synthetic"], "provenance": None}
    saved = call("openclaw", "writer", "capture_note", capture)
    original = call("openclaw", "writer", "get_note", {"id": saved["record"]["id"]})
    require(original["editable"] and original["revision"] == saved["record"]["revision"],
            "Native writer did not receive an editable exact snapshot")
    edit = {"request_id": "native-edit-001", "id": original["id"], "revision": original["revision"],
            "patch": {"title": "Edited through OpenClaw"}}
    edited = call("openclaw", "writer", "edit_note", edit)
    retry = call("hermes", "writer", "edit_note", edit)
    require(not edited["replayed"] and retry["replayed"] and edited["outcome"] == retry["outcome"],
            "Native edit replay changed the original mutation outcome")
    current = call("hermes", "writer", "get_note", {"id": original["id"]})
    require(current["title"] == edit["patch"]["title"] and current["revision"] != original["revision"],
            "Hermes did not observe OpenClaw's committed edit/revision")
    stale = {**edit, "request_id": "native-stale-edit-001", "patch": {"title": "Stale edit must fail"}}
    call("hermes", "writer", "edit_note", stale, "revision_conflict")
    hermes_edit = {"request_id": "native-edit-002", "id": current["id"], "revision": current["revision"],
                   "patch": {"content": "nativemutationatlas edited by the native Hermes registry."}}
    second = call("hermes", "writer", "edit_note", hermes_edit)
    final = call("openclaw", "writer", "get_note", {"id": current["id"]})
    require(final["content"] == hermes_edit["patch"]["content"] and final["revision"] != current["revision"],
            "OpenClaw did not observe Hermes' committed edit/revision")
    delete = {"request_id": "native-delete-001", "id": final["id"], "revision": final["revision"], "confirmed": True}
    call("openclaw", "writer", "delete_note", {**delete, "request_id": "native-unconfirmed-delete-001", "confirmed": False}, "invalid_input")
    removed = call("hermes", "writer", "delete_note", delete)
    delete_retry = call("openclaw", "writer", "delete_note", delete)
    require(not removed["replayed"] and delete_retry["replayed"] and removed["outcome"] == delete_retry["outcome"],
            "Native confirmed delete replay changed its original outcome")
    call("hermes", "writer", "get_note", {"id": final["id"]}, "not_found")
    capture_retry = call("openclaw", "writer", "capture_note", capture)
    require(capture_retry["replayed"] and capture_retry["record"] == saved["record"],
            "Original capture receipt changed after native edit/delete")
    call("openclaw", "writer", "get_note", {"id": final["id"]}, "not_found")
    evidence["mutations"] = {"id": final["id"], "original_revision": original["revision"],
        "deleted_revision": final["revision"], "openclaw_edit": edited, "hermes_edit": second,
        "delete": removed, "stale_revision_rejected": True, "unconfirmed_delete_rejected": True,
        "capture_replay_did_not_resurrect_deleted_note": True}

    upload = {"request_id": "native-upload-001", "document_key": "synthetic/native-upload-atlas.md",
        "content": "# Native upload atlas\n\nnativeuploadatlas is a fictional project shared by native MCP clients.\n",
        "title": "Native upload atlas", "extract_entities": False,
        "provenance": {"uri": "file:///client-only/native-upload-atlas.md", "label": "synthetic uploaded document",
                       "metadata": {"session": "synthetic-native-upload-session"}}}
    with provider_pause(provider):
        admission = call("openclaw", "uploader", "upload_source", upload)
        require(provider.entered.wait(min(10, timeout())), "Uploaded job never reached the controlled provider pause")
        # The admitting runtime has already disposed; the host still owns work.
        running = call("hermes", "uploader", "get_job", {"id": admission["job_id"]})
        require(running["status"] == "running" and running["instance_id"] == "uploader",
                "Native disconnect did not leave an independently owned running job")
        cancellation = call("hermes", "uploader", "cancel_job", {"id": admission["job_id"]})
        require(cancellation["cancellation_requested"], "Native cancellation did not set the independent durable flag")

    def terminal(client):
        limit = time.monotonic() + min(30, timeout())
        while time.monotonic() < limit:
            status = call(client, "uploader", "get_job", {"id": admission["job_id"]})
            if status["status"] in ("completed", "cancelled", "failed"):
                return status
            time.sleep(0.2)
        require(False, "Uploaded native job exceeded its bounded terminal-state deadline")

    cancelled = terminal("openclaw")
    require(cancelled["status"] == "cancelled", "Explicitly cancelled native job did not stop at a safe boundary")
    resumed = call("openclaw", "uploader", "resume_job", {"id": admission["job_id"]})
    require(resumed["id"] == admission["job_id"] and not resumed["cancellation_requested"],
            "Native resume changed job identity or retained its cancellation flag")
    completed = terminal("hermes")
    require(completed["status"] == "completed" and completed["result"]["note_ids"],
            "Native resumed upload failed to publish a complete source generation")
    replay = call("hermes", "uploader", "upload_source", upload)
    require(replay["replayed"] and not admission["replayed"] and
            all(replay[key] == admission[key] for key in ("request_id", "job_id", "source_id", "source_uri")),
            "Native upload retry did not preserve admission identities")
    call("openclaw", "uploader", "upload_source", {**upload, "content": upload["content"] + "Changed request.\n"}, "revision_conflict")
    for tool in ("get_job", "cancel_job", "resume_job"):
        call("hermes", "foreign-jobs", tool, {"id": admission["job_id"]}, "not_found")
    foreign = call("openclaw", "foreign-jobs", "list_jobs", {"limit": 10})
    require(not foreign["jobs"], "Foreign native job listing exposed another principal's job")
    own = call("hermes", "uploader", "list_jobs", {"limit": 10})
    require([job["id"] for job in own["jobs"]] == [admission["job_id"]], "Native upload replay duplicated durable jobs")
    source = call("openclaw", "openclaw-b", "get_source", {"id": admission["source_id"]})
    shared = call("hermes", "hermes", "get_source", {"id": admission["source_id"]})
    require(source == shared and source["instance_id"] == "uploader" and source["content"] == upload["content"] and
            source["successful_generation"] == source["generation"] and source["uri"] == admission["source_uri"] and
            source["uri"].startswith("mcp://upload/") and source["provenance"] == upload["provenance"],
            "Both native readers did not share exact uploaded content/provenance/generation")
    require("upload_source" not in evidence["native_calls"][-1]["catalog"] and
            "edit_note" not in evidence["native_calls"][-1]["catalog"], "Default Hermes principal gained mutation/upload privileges")
    ids = completed["result"]["note_ids"]
    for record_id in ids:
        inspected = call("openclaw", "openclaw-a", "get_record", {"id": record_id, "neighbors": 0})
        require(inspected["provenance"]["instance_id"] == "uploader" and
                inspected["provenance"]["source_uri"] == source["uri"], "Uploaded chunk lost trusted source provenance")
    evidence["upload"] = {"admission": admission, "cancelled_checkpoint": cancelled,
        "completed_job": completed, "source": source, "job_owner_isolation": True,
        "disconnect_did_not_cancel": True, "explicit_cancel_and_resume": True,
        "changed_retry_payload_rejected": True, "shared_native_source_read": True}
    return evidence, {"capture": capture, "saved_record": saved["record"], "delete": delete,
        "delete_outcome": removed["outcome"], "upload": upload, "admission": admission, "note_ids": ids}


def replay_extended(calls, state, require):
    requests = [{"tool": "delete_note", "arguments": state["delete"]},
                {"tool": "capture_note", "arguments": state["capture"]},
                {"tool": "get_note", "arguments": {"id": state["delete"]["id"]}, "expect_error": "not_found"}]
    deleted, capture, _ = calls("hermes", "writer", requests)["results"]
    require(deleted["replayed"] and deleted["outcome"] == state["delete_outcome"] and
            capture["replayed"] and capture["record"] == state["saved_record"],
            "Restart lost immutable native capture/delete receipts")
    admission, job = calls("openclaw", "uploader", [
        {"tool": "upload_source", "arguments": state["upload"]},
        {"tool": "get_job", "arguments": {"id": state["admission"]["job_id"]}}])["results"]
    require(admission["replayed"] and admission["job_id"] == state["admission"]["job_id"] and
            job["status"] == "completed" and job["result"]["note_ids"] == state["note_ids"],
            "Restart changed native upload admission or completed chunk identities")
    return {"capture_delete_receipts_preserved": True, "upload_admission_preserved": True,
            "completed_chunk_ids_preserved": True, "note_ids": state["note_ids"]}
