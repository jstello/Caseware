from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from sse_starlette.sse import EventSourceResponse

from .schemas import JsonRunRequest
from .service import get_service


def create_app() -> FastAPI:
    app = FastAPI(
        title="Invoice Agent",
        version="0.1.0",
        description="Local invoice-processing agent with FastAPI, Google ADK, and a custom SSE contract.",
    )

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs/stream")
    async def run_stream(request: Request) -> EventSourceResponse:
        service = get_service()
        run_id = service.new_run_id()
        run_dir = service.create_run_dir(run_id)

        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            payload = JsonRunRequest.model_validate(await request.json())
            prepared_input = await service.prepare_folder_input(payload.folder_path)
            prompt = payload.prompt
        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            prompt = form.get("prompt")
            uploads: list[UploadFile] = [
                value
                for _, value in form.multi_items()
                if isinstance(value, StarletteUploadFile)
            ]
            if not uploads:
                raise HTTPException(status_code=422, detail="At least one invoice image is required.")
            prepared_input = await service.prepare_upload_input(run_dir=run_dir, uploads=uploads)
        else:
            raise HTTPException(
                status_code=415,
                detail="Use application/json with folder_path or multipart/form-data with invoice image files.",
            )

        return EventSourceResponse(
            service.run_stream(
                run_id=run_id,
                run_dir=run_dir,
                prepared_input=prepared_input,
                prompt=prompt if isinstance(prompt, str) else None,
            )
        )

    return app


app = create_app()
