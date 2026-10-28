"""Read-only access to Google Docs in one Drive folder. Credentials stay on the backend."""
import io
import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class DriveError(Exception):
    """User-presentable Drive failure (no secrets inside)."""


def run_oauth_flow() -> None:
    """One-time browser login. Run via `python -m app.scripts.google_auth`."""
    creds_file = Path(settings.google_credentials_file)
    if not creds_file.exists():
        raise DriveError(f"OAuth client file not found: {creds_file}")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
    creds = flow.run_local_server(port=0)
    token_path = Path(settings.google_token_file)
    token_path.write_text(creds.to_json())
    token_path.chmod(0o600)


def _load_credentials() -> Credentials:
    token_path = Path(settings.google_token_file)
    if not token_path.exists():
        raise DriveError(
            "Google is not authorized yet. Run `python -m app.scripts.google_auth` in the backend folder."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                raise DriveError(
                    "Google token refresh failed. Re-run `python -m app.scripts.google_auth`."
                ) from exc
            token_path.write_text(creds.to_json())
        else:
            raise DriveError("Google token is invalid. Re-run `python -m app.scripts.google_auth`.")
    return creds


def build_service():
    return build("drive", "v3", credentials=_load_credentials(), cache_discovery=False)


def _http_message(exc: HttpError) -> str:
    status = getattr(exc.resp, "status", None)
    if status == 404:
        return "Drive folder or document not found. Check GOOGLE_DRIVE_FOLDER_ID."
    if status in (401, 403):
        return "Permission denied by Google Drive (or API not enabled / quota exceeded)."
    return f"Google Drive API error (HTTP {status})."


def list_documents(service=None, folder_id: str | None = None) -> list[dict]:
    """Metadata for every Google Doc directly in the folder (handles pagination)."""
    folder_id = folder_id or settings.google_drive_folder_id
    if not folder_id:
        raise DriveError("GOOGLE_DRIVE_FOLDER_ID is not set.")
    service = service or build_service()

    docs: list[dict] = []
    page_token = None
    try:
        while True:
            resp = (
                service.files()
                .list(
                    q=(
                        f"'{folder_id}' in parents and trashed=false and "
                        f"(mimeType='{GOOGLE_DOC_MIME}' or mimeType='{DOCX_MIME}')"
                    ),
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                    pageSize=100,
                    pageToken=page_token,
                )
                .execute()
            )
            for f in resp.get("files", []):
                docs.append(
                    {
                        "id": f["id"],
                        "title": f.get("name", "(untitled)"),
                        "mime_type": f.get("mimeType", GOOGLE_DOC_MIME),
                        "modified_time": f.get("modifiedTime", ""),
                        "url": f.get("webViewLink")
                        or f"https://docs.google.com/document/d/{f['id']}/edit",
                    }
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        logger.warning("Drive list failed: status=%s", getattr(exc.resp, "status", None))
        raise DriveError(_http_message(exc)) from exc
    return docs


def extract_docx_text(data: bytes) -> str:
    from docx import Document

    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DriveError("Could not read this Word file (corrupt or unsupported).") from exc
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def read_document_text(doc_id: str, service=None, mime_type: str = GOOGLE_DOC_MIME) -> str:
    service = service or build_service()
    try:
        if mime_type == DOCX_MIME:
            return extract_docx_text(service.files().get_media(fileId=doc_id).execute())
        data = service.files().export(fileId=doc_id, mimeType="text/plain").execute()
    except HttpError as exc:
        logger.warning("Drive read failed for %s: status=%s", doc_id, getattr(exc.resp, "status", None))
        raise DriveError(_http_message(exc)) from exc
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return data or ""


def fetch_documents(with_text: bool = True) -> list[dict]:
    """List docs and (optionally) attach text. A single unreadable doc is skipped with an error note."""
    service = build_service()
    docs = list_documents(service)
    if with_text:
        for d in docs:
            try:
                d["text"] = read_document_text(d["id"], service, d["mime_type"])
            except DriveError as exc:
                d["text"] = ""
                d["error"] = str(exc)
    return docs
