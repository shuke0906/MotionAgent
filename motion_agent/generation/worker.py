"""Minimal Phase 7 generation worker and queue facade."""

from __future__ import annotations

import queue
import time
from collections.abc import Callable

import torch

from motion_agent.common.errors import WorkerUnavailableError
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.normal_generator import PredictModel, generate_motion
from motion_agent.generation.schemas import GenerationRequest, GenerationResult, GPUGenerationJob, WorkerHealth
from motion_agent.state.schemas import GEMAdapterContext


class InProcessGenerationQueue:
    def __init__(self, maxsize: int = 8) -> None:
        self._queue: queue.Queue[GPUGenerationJob] = queue.Queue(maxsize=maxsize)

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def submit(self, job: GPUGenerationJob) -> None:
        if self._queue.full():
            raise WorkerUnavailableError("generation queue is full")
        self._queue.put(job.model_copy(update={"queued_at": time.time()}))

    def get_nowait(self) -> GPUGenerationJob:
        try:
            return self._queue.get_nowait()
        except queue.Empty as exc:
            raise WorkerUnavailableError("generation queue is empty") from exc


class GEMGenerationWorker:
    def __init__(
        self,
        *,
        worker_id: str,
        model_factory: Callable[[], PredictModel],
        checkpoint_version: str,
        adapter_context: GEMAdapterContext | None = None,
        candidate_store: CandidateStore | None = None,
        device: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._model_factory = model_factory
        self._model: PredictModel | None = None
        self.checkpoint_version = checkpoint_version
        self.adapter_context = adapter_context
        self.candidate_store = candidate_store
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_job_id: str | None = None
        self.last_success_at: float | None = None
        self.last_failure_at: float | None = None
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.last_latency_ms: float | None = None
        self.peak_vram_mb: float | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is None:
            self._model = self._model_factory()

    def run(self, request: GenerationRequest) -> GenerationResult:
        self.load_model()
        if self._model is None:
            raise WorkerUnavailableError("GEM model is not loaded")
        job_id = request.generation_id
        self.current_job_id = job_id
        start = time.perf_counter()
        try:
            result = generate_motion(
                request,
                model=self._model,
                checkpoint_version=self.checkpoint_version,
                adapter_context=self.adapter_context,
                candidate_store=self.candidate_store,
            )
            self.jobs_completed += 1
            self.last_success_at = time.time()
            self.peak_vram_mb = _peak_memory_mb()
            return result
        except Exception:
            self.jobs_failed += 1
            self.last_failure_at = time.time()
            raise
        finally:
            self.current_job_id = None
            self.last_latency_ms = (time.perf_counter() - start) * 1000

    def health(self, *, queue_depth: int = 0) -> WorkerHealth:
        return WorkerHealth(
            worker_id=self.worker_id,
            available=self.current_job_id is None,
            model_loaded=self.model_loaded,
            current_job_id=self.current_job_id,
            queue_depth=queue_depth,
            last_success_at=self.last_success_at,
            last_failure_at=self.last_failure_at,
            jobs_completed=self.jobs_completed,
            jobs_failed=self.jobs_failed,
            device=self.device,
            gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            peak_vram_mb=self.peak_vram_mb,
            last_latency_ms=self.last_latency_ms,
        )


def _peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
