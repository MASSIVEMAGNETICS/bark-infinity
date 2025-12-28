"""FastAPI server that exposes Bark Infinity generation endpoints for the React UI."""

from __future__ import annotations

import base64
import io
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import numpy as np

from bark_infinity import api, config, error_handling, generation
from bark_infinity.api import process_history_prompt
from bark_infinity.generation import SAMPLE_RATE, set_seed

error_handling.set_global_exception_logger()

app = FastAPI(title="Bark Infinity React API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_history_prompt(history_prompt: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Attempt to resolve the history prompt path and provide a friendly name."""
    if not history_prompt:
        return None, None

    resolved = process_history_prompt(history_prompt=history_prompt)
    if resolved:
        name = Path(resolved).stem
        return resolved, name
    # Fall back to whatever the user provided so the error message is meaningful.
    return history_prompt, Path(history_prompt).stem if history_prompt else None


def _audio_array_to_wav_bytes(audio_array) -> bytes:
    """Convert a numpy float waveform into a WAV byte sequence."""
    if audio_array is None:
        raise ValueError("No audio returned from Bark Infinity")

    # Ensure we always work with a 1-D array
    audio = np.array(audio_array)
    if audio.ndim > 1:
        audio = audio.squeeze()

    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(int16_audio.tobytes())

    return buffer.getvalue()


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    history_prompt: Optional[str] = Field(default=None)
    text_temp: float = Field(default=0.7, ge=0.0, le=1.5)
    waveform_temp: float = Field(default=0.7, ge=0.0, le=1.5)
    seed: Optional[int] = Field(default=None, ge=0)


class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    filename: str
    duration_seconds: float
    history_prompt_path: Optional[str]
    history_prompt_name: Optional[str]


class PromptInfo(BaseModel):
    name: str
    path: str


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/prompts", response_model=List[PromptInfo])
def list_prompts() -> List[PromptInfo]:
    prompts: List[PromptInfo] = []
    seen_paths: set[str] = set()

    for directory in config.VALID_HISTORY_PROMPT_DIRS:
        if not directory:
            continue
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            continue
        for candidate in sorted(directory_path.glob("*.npz")):
            resolved_path = str(candidate.resolve())
            if resolved_path in seen_paths:
                continue
            prompts.append(PromptInfo(name=candidate.stem, path=resolved_path))
            seen_paths.add(resolved_path)

    return prompts


@app.post("/api/generate", response_model=GenerateResponse)
def generate_audio(request: GenerateRequest) -> GenerateResponse:
    resolved_history_prompt, history_name = _resolve_history_prompt(request.history_prompt)

    if request.seed is not None:
        set_seed(int(request.seed))

    try:
        audio_array = api.generate_audio(
            text=request.text,
            history_prompt=resolved_history_prompt,
            text_temp=request.text_temp,
            waveform_temp=request.waveform_temp,
            silent=True,
        )
    except Exception as exc:  # pragma: no cover - FastAPI will convert to HTTP error
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wav_bytes = _audio_array_to_wav_bytes(audio_array)
    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    duration_seconds = len(audio_array) / float(SAMPLE_RATE)

    filename = f"bark-react-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.wav"

    return GenerateResponse(
        audio_base64=audio_base64,
        sample_rate=SAMPLE_RATE,
        filename=filename,
        duration_seconds=duration_seconds,
        history_prompt_path=resolved_history_prompt,
        history_prompt_name=history_name,
    )


@app.on_event("startup")
def configure_bark_defaults() -> None:
    # Ensure Bark runs with the same defaults as the CLI/Gradio experience.
    generation.OFFLOAD_CPU = os.environ.get("SUNO_OFFLOAD_CPU", "1") not in {"0", "false", "False"}
    generation.USE_SMALL_MODELS = os.environ.get("SUNO_USE_SMALL_MODELS", "0") in {"1", "true", "True"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("bark_react_server:app", host="0.0.0.0", port=8000, reload=False)
