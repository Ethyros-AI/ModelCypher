import asyncio
import uuid
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Set, Any
from contextlib import asynccontextmanager

class ResourceIntensiveOperation(str, Enum):
    RAG_INDEXING = "RAG Indexing"
    RAG_QUERY = "RAG Query"
    MODEL_INFERENCE = "Model Inference"

@dataclass
class TrainingSessionInfo:
    job_id: str
    start_time: float
    duration: float

class ResourceError(Exception):
    pass

class TrainingResourceGuard:
    """
    Singleton enforcing exclusive GPU access across training, RAG, and inference.
    Replicates Swift's Actor isolation using asyncio.Lock.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingResourceGuard, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._lock = asyncio.Lock()
        self._active_training_job_id: Optional[str] = None
        self._training_start_time: Optional[float] = None
        
        # Simple set of owners for inference (e.g. "user", "session_id")
        self._active_inference_owners: Set[str] = set()
        self._max_concurrent_inference_owners = 2
        
        self._initialized = True

    @property
    def is_training_active(self) -> bool:
        return self._active_training_job_id is not None

    async def get_current_training_session(self) -> Optional[TrainingSessionInfo]:
        async with self._lock:
            if not self._active_training_job_id or not self._training_start_time:
                return None
            
            duration = time.time() - self._training_start_time
            return TrainingSessionInfo(
                job_id=self._active_training_job_id,
                start_time=self._training_start_time,
                duration=duration
            )

    async def begin_training(self, job_id: str):
        async with self._lock:
            if self._active_inference_owners:
                raise ResourceError(f"Inference in progress by: {self._active_inference_owners}")
            
            if self._active_training_job_id:
                raise ResourceError(f"Training job {self._active_training_job_id} is already running.")
            
            self._active_training_job_id = job_id
            self._training_start_time = time.time()

    async def end_training(self, job_id: str):
        async with self._lock:
            if self._active_training_job_id == job_id:
                self._active_training_job_id = None
                self._training_start_time = None

    @asynccontextmanager
    async def training_session(self, job_id: str):
        """Context manager for safe training session resource usage."""
        await self.begin_training(job_id)
        try:
            yield
        finally:
            await self.end_training(job_id)

    async def request_resource_access(self, operation: ResourceIntensiveOperation):
        async with self._lock:
            if self._active_training_job_id:
                raise ResourceError(f"{operation.value} Unavailable: Training is currently using the GPU.")

    async def begin_inference(self, owner: str):
        async with self._lock:
            if self._active_training_job_id:
                raise ResourceError(f"Training job {self._active_training_job_id} is active. Inference blocked.")
            
            if owner in self._active_inference_owners:
                return # Already registered
                
            if len(self._active_inference_owners) >= self._max_concurrent_inference_owners:
                 raise ResourceError("Maximum concurrent inference sessions reached.")
            
            self._active_inference_owners.add(owner)

    async def end_inference(self, owner: str):
        async with self._lock:
            if owner in self._active_inference_owners:
                self._active_inference_owners.remove(owner)

    @asynccontextmanager
    async def inference_session(self, owner: str):
        await self.begin_inference(owner)
        try:
            yield
        finally:
            await self.end_inference(owner)
