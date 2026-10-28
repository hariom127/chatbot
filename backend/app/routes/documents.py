import logging

from fastapi import APIRouter, HTTPException

from app.services import google_drive, indexer
from app.services.google_drive import DriveError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents")
def list_documents(include_text: bool = False):
    """Lists docs in the configured folder (use include_text=true to preview extracted text)."""
    try:
        docs = google_drive.fetch_documents(with_text=include_text)
    except DriveError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"count": len(docs), "documents": docs}


@router.post("/sync")
def sync():
    try:
        docs = google_drive.fetch_documents(with_text=True)
        return indexer.sync_documents(docs)
    except DriveError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("Sync failed")
        raise HTTPException(status_code=500, detail="Sync failed. Check the backend logs.")


@router.get("/sync/status")
def sync_status():
    return indexer.get_status()
