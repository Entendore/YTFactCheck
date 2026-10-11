#!/usr/bin/env python3
# ==============================================================================
# llama-cpp-smart-gpu — Single-file Smart GPU LLM Inference Engine
# ==============================================================================

# Usage:
#   python llama_smart_gpu.py -m /path/to/model.gguf
#   python llama_smart_gpu.py --info
#   python llama_smart_gpu.py --list-models
#   python llama_smart_gpu.py --benchmark -m /path/to/model.gguf
# ==============================================================================

__version__ = "2.2.0"
__all__ = [
    "GPUConfig", "PerformanceConfig", "InferenceConfig", "ServerConfig",
    "LoggingConfig", "SmartGPUConfig", "SmartGPUConfigBuilder",
    "load_config", "save_config", "apply_cli_overrides",
    "VRAMPressure", "GPUMemoryInfo", "LayerOffloadPlan", "MemoryAlert",
    "VRAMEstimator", "GPUDetector", "SmartGPUManager",
    "GenerationMetrics", "WindowStats", "PerformanceReport", "HealthStatus",
    "RollingWindow", "PerformanceMonitor",
    "ModelInfo", "GenerationResult", "SlotState", "LlamaEngine",
    "ModelStatus", "ModelEntry", "ModelRecommendation",
    "ModelClassifier", "ModelManager",
    "RequestPriority", "RequestStatus", "InferenceRequest", "Session",
    "SessionManager", "InferencePipeline",
    "create_app", "mount_gui",
    "main", "setup_logging", "get_system_info", "format_bytes",
    "format_duration", "parse_args",
    "LlamaSmartGPUError", "ModelNotFoundError", "ModelLoadError",
    "GPUOutOfMemoryError", "InferenceError", "ConfigValidationError",
    "DownloadError", "SessionExpiredError",
    "EngineProtocol", "GPUManagerProtocol",
    "retry", "CircuitBreaker", "CircuitState", "TokenBucketRateLimiter",
    "EventBus", "EventType", "Event", "LazyProperty", "Singleton",
    "ttl_cache", "timed", "log_calls", "timer", "suppressed_exceptions",
    "ModelLifecycle", "ModelLifecycleState", "SmartGPUConfigBuilder",
    "ReadWriteLock", "ServiceContainer", "deprecated", "rate_limit",
    "SlotContext", "ModelContext",
]

# --- Standard Library Imports (consolidated) ---

import os
import re
import sys
import json
import time
import math
import uuid
import abc
import hashlib
import shutil
import logging
import argparse
import platform
import threading
import subprocess
import urllib.request
import urllib.error
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List, Literal,
    Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast,
)
from typing_extensions import Protocol, runtime_checkable
from functools import wraps, cached_property
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from logging.handlers import RotatingFileHandler
from queue import PriorityQueue

logger = logging.getLogger(__name__)

# --- Type Aliases ---

Seconds = float
Milliseconds = float
Megabytes = float
TokenCount = int
LayerCount = int
Percentage = float
ModelPath = str
SessionId = str
RequestId = str
ModelName = str
KVCacheType = Literal["f16", "q8_0", "q4_0", "q4_1"]
FinishReason = Literal["stop", "length", "stop_sequence"]
JSONDict = Dict[str, Any]
MessageList = List[Dict[str, str]]
EventCallback = Callable[..., None]
TokenCallback = Callable[[str], None]
ErrorCallback = Callable[[Exception], None]
MetricCallback = Callable[[str, float, float, float], None]
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ==============================================================================
# EXCEPTION HIERARCHY
# ==============================================================================

class LlamaSmartGPUError(Exception):
    """Base exception for all llama-cpp-smart-gpu errors.

    Provides structured error context with error code, component name,
    and optional details dict for machine-readable error reporting.
    """

    def __init__(self, message: str = "", code: str = "UNKNOWN",
                 component: str = "", details: Optional[JSONDict] = None):
        self.code = code
        self.component = component
        self.details = details or {}
        full_msg = f"[{code}]"
        if component:
            full_msg += f" [{component}]"
        full_msg += f" {message}"
        super().__init__(full_msg)


class ModelNotFoundError(LlamaSmartGPUError):
    """Raised when a requested model cannot be found on disk or in the registry."""

    def __init__(self, model_name: str, searched_paths: Optional[List[str]] = None):
        self.model_name = model_name
        self.searched_paths = searched_paths or []
        super().__init__(
            message=f"Model not found: {model_name}",
            code="MODEL_NOT_FOUND", component="ModelManager",
            details={"model_name": model_name, "searched_paths": searched_paths},
        )


class ModelLoadError(LlamaSmartGPUError):
    """Raised when a model fails to load (corrupt file, incompatible format, OOM)."""

    def __init__(self, model_name: str, reason: str = ""):
        self.model_name = model_name
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            code="MODEL_LOAD_FAILED", component="Engine",
            details={"model_name": model_name, "reason": reason},
        )


class GPUOutOfMemoryError(LlamaSmartGPUError):
    """Raised when GPU VRAM is insufficient for the requested operation."""

    def __init__(self, required_mb: float, available_mb: float):
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            message=f"GPU OOM: need {required_mb:.0f} MB, have {available_mb:.0f} MB",
            code="GPU_OUT_OF_MEMORY", component="GPUManager",
            details={"required_mb": required_mb, "available_mb": available_mb},
        )


class InferenceError(LlamaSmartGPUError):
    """Raised when text generation fails (timeout, cancellation, internal error)."""

    def __init__(self, reason: str = "", request_id: str = ""):
        super().__init__(
            message=f"Inference failed: {reason}",
            code="INFERENCE_ERROR", component="InferencePipeline",
            details={"reason": reason, "request_id": request_id},
        )


class ConfigValidationError(LlamaSmartGPUError):
    """Raised when a configuration value is invalid or out of range."""

    def __init__(self, field_path: str, value: Any, reason: str = ""):
        super().__init__(
            message=f"Invalid config '{field_path}': {reason} (got {value!r})",
            code="CONFIG_VALIDATION_ERROR", component="Config",
            details={"field_path": field_path, "value": str(value), "reason": reason},
        )


class DownloadError(LlamaSmartGPUError):
    """Raised when a model download fails or is cancelled."""

    def __init__(self, url: str, reason: str = ""):
        super().__init__(
            message=f"Download failed for {url}: {reason}",
            code="DOWNLOAD_ERROR", component="ModelManager",
            details={"url": url, "reason": reason},
        )


class SessionExpiredError(LlamaSmartGPUError):
    """Raised when attempting to use an expired conversation session."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session expired: {session_id}",
            code="SESSION_EXPIRED", component="SessionManager",
            details={"session_id": session_id},
        )


# ==============================================================================
# ABSTRACT BASE CLASSES (Protocol-based interfaces)
# ==============================================================================

@runtime_checkable
class EngineProtocol(Protocol):
    """Protocol defining the interface for any LLM inference engine.

    Implementations must provide model loading, text generation,
    and telemetry.  Enables dependency injection and test mocking.
    """
    @property
    def is_loaded(self) -> bool: ...
    def load_model(self, model_path: str) -> "ModelInfo": ...
    def unload_model(self) -> None: ...
    def generate(self, prompt: str, **kwargs) -> "GenerationResult": ...
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]: ...
    def chat(self, messages: MessageList, **kwargs) -> "GenerationResult": ...
    def chat_stream(self, messages: MessageList, **kwargs) -> Iterator[Dict]: ...
    def benchmark(self, prompt_tokens: int = 128, gen_tokens: int = 128) -> JSONDict: ...
    def get_telemetry(self) -> JSONDict: ...


@runtime_checkable
class GPUManagerProtocol(Protocol):
    """Protocol defining the interface for GPU memory management."""
    def initialize(self) -> "GPUMemoryInfo": ...
    def compute_offload_plan(self, n_params_b: float, n_layers: int,
                             quant: str = "q4_k_m", n_ctx: int = 4096,
                             n_heads: int = 32) -> "LayerOffloadPlan": ...
    def get_available_vram(self) -> float: ...
    def get_memory_info(self) -> "GPUMemoryInfo": ...
    def get_pressure(self) -> "VRAMPressure": ...
    def start_monitoring(self) -> None: ...
    def stop_monitoring(self) -> None: ...


# ==============================================================================
# RETRY DECORATOR (exponential backoff with jitter)
# ==============================================================================

def retry(max_attempts: int = 3, base_delay: Seconds = 1.0,
          max_delay: Seconds = 30.0, exponential_base: float = 2.0,
          jitter: bool = True,
          retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
          on_retry: Optional[Callable[[int, Exception, Seconds], None]] = None,
          ) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff and optional jitter.

    Args:
        max_attempts: Maximum number of call attempts (including the first).
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff.
        jitter: If True, adds random jitter (0..base_delay) to avoid thundering herd.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        on_retry: Optional callback(attempt, exception, delay) fired on each retry.
    """
    import random

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random() * 0.5
                    if on_retry:
                        on_retry(attempt, exc, delay)
                    logging.getLogger(func.__module__).warning(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt, max_attempts, func.__qualname__, delay, exc,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        wrapper._retry_config = {  # type: ignore[attr-defined]
            "max_attempts": max_attempts, "base_delay": base_delay, "max_delay": max_delay,
        }
        return wrapper  # type: ignore[return-value]
    return decorator


# ==============================================================================
# CIRCUIT BREAKER
# ==============================================================================

class CircuitState(Enum):
    """States of a circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for protecting GPU-critical operations.

    Tracks consecutive failures and opens the circuit when a threshold
    is exceeded.  After a cooldown, transitions to half-open to probe
    for recovery.

    Example::

        gpu_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        with gpu_cb:
            result = engine.generate(prompt)
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: Seconds = 60.0,
                 half_open_max_calls: int = 1, name: str = "default"):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker '%s' -> HALF_OPEN", self._name)
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def record_success(self):
        with self._lock:
            self._success_count += 1
            self._failure_count = 0
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker '%s' -> CLOSED (recovered)", self._name)

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold and self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker '%s' -> OPEN after %d failures",
                               self._name, self._failure_count)

    def __enter__(self):
        state = self.state
        if state == CircuitState.OPEN:
            raise LlamaSmartGPUError(
                message=f"Circuit breaker '{self._name}' is OPEN (too many failures)",
                code="CIRCUIT_OPEN", component="CircuitBreaker",
            )
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                self._half_open_calls += 1
                if self._half_open_calls > self._half_open_max_calls:
                    raise LlamaSmartGPUError(
                        message=f"Circuit breaker '{self._name}' HALF_OPEN limit reached",
                        code="CIRCUIT_HALF_OPEN_LIMIT", component="CircuitBreaker",
                    )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False


# ==============================================================================
# TOKEN BUCKET RATE LIMITER
# ==============================================================================

class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter.

    Controls operation rate using a classic token bucket algorithm.
    Tokens are added at a fixed refill rate up to a maximum capacity.

    Example::

        limiter = TokenBucketRateLimiter(rate=10.0, capacity=20)
        if limiter.acquire():
            handle_request()
    """

    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def acquire(self, tokens: int = 1, timeout: Optional[Seconds] = None) -> bool:
        """Attempt to acquire tokens, optionally blocking until available."""
        deadline = (time.monotonic() + timeout) if timeout else None
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                if deadline is None:
                    return False
                wait_time = min((tokens - self._tokens) / self._rate,
                                deadline - time.monotonic())
                if wait_time <= 0:
                    return False
            time.sleep(max(wait_time, 0.001))

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ==============================================================================
# EVENT BUS (Observer Pattern)
# ==============================================================================

class EventType(Enum):
    """System event types for the event bus."""
    MODEL_DISCOVERED = "model.discovered"
    MODEL_LOADING = "model.loading"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADING = "model.unloading"
    MODEL_UNLOADED = "model.unloaded"
    MODEL_VALIDATED = "model.validated"
    MODEL_DOWNLOAD_STARTED = "model.download.started"
    MODEL_DOWNLOAD_PROGRESS = "model.download.progress"
    MODEL_DOWNLOAD_COMPLETED = "model.download.completed"
    MODEL_DOWNLOAD_FAILED = "model.download.failed"
    GPU_INITIALIZED = "gpu.initialized"
    GPU_PRESSURE_CHANGE = "gpu.pressure_change"
    GPU_MEMORY_WARNING = "gpu.memory_warning"
    GPU_LAYER_ADJUSTMENT = "gpu.layer_adjustment"
    INFERENCE_STARTED = "inference.started"
    INFERENCE_COMPLETED = "inference.completed"
    INFERENCE_FAILED = "inference.failed"
    PERFORMANCE_REGRESSION = "performance.regression"
    SERVER_STARTING = "server.starting"
    SERVER_STARTED = "server.started"
    SERVER_SHUTDOWN = "server.shutdown"


@dataclass(slots=True)
class Event:
    """Immutable event object emitted by the event bus."""
    type: EventType
    data: JSONDict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class EventBus:
    """Central event bus implementing the Observer pattern.

    Components subscribe to event types and receive callbacks when events
    are emitted.  Errors in individual callbacks are caught and logged
    to prevent one failing handler from disrupting others.

    Example::

        bus = EventBus()
        bus.subscribe(EventType.MODEL_LOADED, lambda e: print(e.data))
        bus.emit(EventType.MODEL_LOADED, {"model": "llama-7b.gguf"})
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[EventCallback]] = {}
        self._global_handlers: List[EventCallback] = []
        self._history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        with self._lock:
            self._global_handlers.append(callback)

    def unsubscribe(self, event_type: EventType, callback: EventCallback) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type, [])
            if callback in handlers:
                handlers.remove(callback)

    def emit(self, event_type: EventType, data: Optional[JSONDict] = None,
             source: str = "") -> None:
        event = Event(type=event_type, data=data or {}, source=source)
        self._history.append(event)
        with self._lock:
            handlers = list(self._subscribers.get(event_type, []))
            global_handlers = list(self._global_handlers)
        for callback in handlers + global_handlers:
            try:
                callback(event)
            except Exception as exc:
                logging.getLogger(__name__).error("Event handler error for %s: %s",
                                                   event_type.value, exc)

    def get_history(self, event_type: Optional[EventType] = None,
                    limit: int = 100) -> List[Event]:
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        self._history.clear()


_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance (lazy singleton)."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


# ==============================================================================
# LAZY PROPERTY DESCRIPTOR
# ==============================================================================

class LazyProperty:
    """Descriptor that computes a property once and caches the result.

    Thread-safe via double-checked locking.

    Example::

        class MyClass:
            @LazyProperty
            def expensive_data(self):
                return compute_expensive_thing()
    """

    def __init__(self, factory: Callable[..., Any]):
        self._factory = factory
        self._attr_name: Optional[str] = None

    def __set_name__(self, owner: Type, name: str):
        self._attr_name = f"_lazy_{name}"

    def __get__(self, obj: Any, owner: Optional[Type] = None):
        if obj is None:
            return self
        attr = self._attr_name or f"_lazy_{self._factory.__name__}"
        value = getattr(obj, attr, None)
        if value is None:
            with threading.Lock():
                value = getattr(obj, attr, None)
                if value is None:
                    value = self._factory(obj)
                    setattr(obj, attr, value)
        return value


# ==============================================================================
# SINGLETON METACLASS
# ==============================================================================

class Singleton(type):
    """Thread-safe singleton metaclass with double-checked locking.

    Example::

        class Database(metaclass=Singleton):
            def __init__(self):
                self.connection = create_connection()
    """
    _instances: Dict[Type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def reset_instance(cls, target_cls: Type) -> None:
        """Reset a singleton (useful for testing)."""
        with cls._lock:
            cls._instances.pop(target_cls, None)


# ==============================================================================
# TTL CACHE DECORATOR
# ==============================================================================

def ttl_cache(ttl: Seconds = 60.0, maxsize: int = 128):
    """Decorator that caches results with a time-to-live (thread-safe).

    Example::

        @ttl_cache(ttl=30.0)
        def get_gpu_info():
            return GPUDetector.detect()
    """
    def decorator(func: F) -> F:
        cache: OrderedDict = OrderedDict()
        timestamps: Dict[Any, float] = {}
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args):
            key = args
            with lock:
                if key in cache:
                    if time.time() - timestamps[key] < ttl:
                        cache.move_to_end(key)
                        return cache[key]
                    del cache[key]
                    del timestamps[key]
                result = func(*args)
                cache[key] = result
                timestamps[key] = time.time()
                while len(cache) > maxsize:
                    oldest = next(iter(cache))
                    del cache[oldest]
                    del timestamps[oldest]
                return result

        wrapper.cache_clear = lambda: (cache.clear(), timestamps.clear())  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]
    return decorator


# ==============================================================================
# TIMING & LOGGING DECORATORS
# ==============================================================================

@dataclass(slots=True)
class TimingResult:
    """Result from the @timed decorator or timer() context manager."""
    elapsed_ms: Milliseconds
    function_name: str
    args_repr: str = ""


def timed(func: Optional[F] = None, *, log_level: str = "DEBUG") -> Callable:
    """Decorator that measures and logs function execution time.

    Usable with or without parentheses::

        @timed
        def load_model(path): ...

        @timed(log_level="INFO")
        def benchmark(): ...
    """
    def decorator(fn: F) -> F:
        _log_level = getattr(logging, log_level.upper(), logging.DEBUG)
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                args_str = ", ".join(
                    [repr(a) for a in args[:3]] +
                    [f"{k}={v!r}" for k, v in list(kwargs.items())[:3]])
                logging.getLogger(fn.__module__).log(
                    _log_level, "%s took %.1f ms (%s)",
                    fn.__qualname__, elapsed, args_str)
        return wrapper  # type: ignore[return-value]
    if func is not None:
        return decorator(func)
    return decorator


def log_calls(func: Optional[F] = None, *, log_level: str = "DEBUG",
              log_result: bool = False) -> Callable:
    """Decorator that logs function entry, exit, and optional return value."""
    def decorator(fn: F) -> F:
        _log_level = getattr(logging, log_level.upper(), logging.DEBUG)
        _logger = logging.getLogger(fn.__module__)
        @wraps(fn)
        def wrapper(*args, **kwargs):
            _logger.log(_log_level, "-> %s(%s)", fn.__qualname__,
                        ", ".join(repr(a) for a in args[:5]))
            try:
                result = fn(*args, **kwargs)
                if log_result:
                    _logger.log(_log_level, "<- %s -> %s", fn.__qualname__,
                                repr(result)[:200])
                else:
                    _logger.log(_log_level, "<- %s [ok]", fn.__qualname__)
                return result
            except Exception as exc:
                _logger.log(_log_level, "<- %s [ERROR] %s", fn.__qualname__, exc)
                raise
        return wrapper  # type: ignore[return-value]
    if func is not None:
        return decorator(func)
    return decorator


# ==============================================================================
# CONTEXT MANAGORS
# ==============================================================================

@contextmanager
def timer(operation_name: str = "operation") -> Iterator[TimingResult]:
    """Context manager that measures wall-clock time of a code block.

    Example::

        with timer("model_load") as t:
            engine.load_model(path)
        print(f"Loaded in {t.elapsed_ms:.0f} ms")
    """
    result = TimingResult(elapsed_ms=0.0, function_name=operation_name)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000


@contextmanager
def suppressed_exceptions(*exception_types: Type[Exception],
                          log_level: str = "WARNING") -> Iterator[None]:
    """Context manager that suppresses specified exceptions and logs them.

    Example::

        with suppressed_exceptions(OSError):
            os.remove(temp_file)
    """
    try:
        yield
    except exception_types as exc:
        logging.getLogger(__name__).log(
            getattr(logging, log_level.upper(), logging.WARNING),
            "Suppressed %s: %s", type(exc).__name__, exc)


# ==============================================================================
# STATE MACHINE - Model Lifecycle
# ==============================================================================

class ModelLifecycleState(Enum):
    """States in the model lifecycle state machine."""
    UNLOADED = "unloaded"
    DISCOVERING = "discovering"
    DISCOVERED = "discovered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DOWNLOADING = "downloading"
    DOWNLOAD_FAILED = "download_failed"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        return self in (ModelLifecycleState.LOADED, ModelLifecycleState.ERROR)

    def __str__(self) -> str:
        return f"ModelLifecycleState.{self.name}"


_MODEL_TRANSITIONS: Dict[Tuple[ModelLifecycleState, EventType], ModelLifecycleState] = {
    (ModelLifecycleState.UNLOADED, EventType.MODEL_DISCOVERED): ModelLifecycleState.DISCOVERED,
    (ModelLifecycleState.DISCOVERED, EventType.MODEL_DOWNLOAD_STARTED): ModelLifecycleState.DOWNLOADING,
    (ModelLifecycleState.DOWNLOADING, EventType.MODEL_DOWNLOAD_COMPLETED): ModelLifecycleState.DISCOVERED,
    (ModelLifecycleState.DOWNLOADING, EventType.MODEL_DOWNLOAD_FAILED): ModelLifecycleState.DOWNLOAD_FAILED,
    (ModelLifecycleState.DOWNLOAD_FAILED, EventType.MODEL_DOWNLOAD_STARTED): ModelLifecycleState.DOWNLOADING,
    (ModelLifecycleState.DISCOVERED, EventType.MODEL_VALIDATED): ModelLifecycleState.VALIDATED,
    (ModelLifecycleState.VALIDATED, EventType.MODEL_LOADING): ModelLifecycleState.LOADING,
    (ModelLifecycleState.LOADING, EventType.MODEL_LOADED): ModelLifecycleState.LOADED,
    (ModelLifecycleState.LOADED, EventType.MODEL_UNLOADING): ModelLifecycleState.UNLOADING,
    (ModelLifecycleState.UNLOADING, EventType.MODEL_UNLOADED): ModelLifecycleState.UNLOADED,
    (ModelLifecycleState.LOADING, EventType.INFERENCE_FAILED): ModelLifecycleState.ERROR,
    (ModelLifecycleState.ERROR, EventType.MODEL_LOADING): ModelLifecycleState.LOADING,
    (ModelLifecycleState.ERROR, EventType.MODEL_UNLOADING): ModelLifecycleState.UNLOADING,
}


class ModelLifecycle:
    """State machine that tracks a model's lifecycle.

    Enforces valid state transitions and emits events on the global
    event bus.  Invalid transitions raise LlamaSmartGPUError.

    Example::

        lc = ModelLifecycle("llama-7b.gguf")
        lc.transition(EventType.MODEL_LOADING)
        lc.transition(EventType.MODEL_LOADED)
        assert lc.state == ModelLifecycleState.LOADED
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._state = ModelLifecycleState.UNLOADED
        self._history: List[Tuple[ModelLifecycleState, float]] = [(self._state, time.time())]
        self._transition_count = 0

    @property
    def state(self) -> ModelLifecycleState:
        return self._state

    @property
    def history(self) -> List[Tuple[ModelLifecycleState, float]]:
        return list(self._history)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def can_transition(self, event: EventType) -> bool:
        return (self._state, event) in _MODEL_TRANSITIONS

    def transition(self, event: EventType) -> ModelLifecycleState:
        key = (self._state, event)
        new_state = _MODEL_TRANSITIONS.get(key)
        if new_state is None:
            raise LlamaSmartGPUError(
                message=f"Invalid transition for '{self.model_name}': "
                        f"{self._state.value} + {event.value}",
                code="INVALID_STATE_TRANSITION", component="ModelLifecycle",
                details={"model": self.model_name, "current": self._state.value,
                         "event": event.value},
            )
        old = self._state
        self._state = new_state
        self._transition_count += 1
        self._history.append((new_state, time.time()))
        logging.getLogger(__name__).info(
            "Model '%s': %s -> %s (%s)", self.model_name,
            old.value, new_state.value, event.value)
        try:
            get_event_bus().emit(event, {
                "model": self.model_name, "old_state": old.value,
                "new_state": new_state.value,
            }, source="ModelLifecycle")
        except Exception:
            pass
        return new_state

    def force_state(self, state: ModelLifecycleState) -> None:
        self._state = state
        self._history.append((state, time.time()))

    def time_in_state(self) -> Seconds:
        _, entered_at = self._history[-1]
        return time.time() - entered_at


# ==============================================================================
# BUILDER PATTERN - SmartGPUConfigBuilder
# ==============================================================================

class SmartGPUConfigBuilder:
    """Fluent builder for constructing SmartGPUConfig with validation.

    Example::

        config = (
            SmartGPUConfigBuilder()
            .with_gpu(vram_budget_mb=8192, n_gpu_layers=-1)
            .with_performance(n_ctx=8192, n_batch=1024)
            .with_inference(temperature=0.8, top_p=0.9)
            .with_server(host="127.0.0.1", port=8000)
            .with_model_dir("~/models")
            .build()
        )
    """

    def __init__(self):
        self._gpu: Dict[str, Any] = {}
        self._performance: Dict[str, Any] = {}
        self._inference: Dict[str, Any] = {}
        self._server: Dict[str, Any] = {}
        self._logging_cfg: Dict[str, Any] = {}
        self._model_paths: List[str] = []
        self._model_dir = os.path.expanduser("~/.cache/llama-smart-gpu/models")
        self._chat_template = ""

    def with_gpu(self, **kwargs) -> "SmartGPUConfigBuilder":
        if "vram_target_fraction" in kwargs:
            v = kwargs["vram_target_fraction"]
            if not 0.0 < v <= 1.0:
                raise ConfigValidationError("gpu.vram_target_fraction", v, "must be in (0.0, 1.0]")
        self._gpu.update(kwargs)
        return self

    def with_performance(self, **kwargs) -> "SmartGPUConfigBuilder":
        if "n_ctx" in kwargs and kwargs["n_ctx"] < 1:
            raise ConfigValidationError("performance.n_ctx", kwargs["n_ctx"], "must be >= 1")
        if "n_batch" in kwargs and kwargs["n_batch"] < 1:
            raise ConfigValidationError("performance.n_batch", kwargs["n_batch"], "must be >= 1")
        self._performance.update(kwargs)
        return self

    def with_inference(self, **kwargs) -> "SmartGPUConfigBuilder":
        if "temperature" in kwargs and not 0.0 <= kwargs["temperature"] <= 2.0:
            raise ConfigValidationError("inference.temperature", kwargs["temperature"], "must be in [0.0, 2.0]")
        if "top_p" in kwargs and not 0.0 <= kwargs["top_p"] <= 1.0:
            raise ConfigValidationError("inference.top_p", kwargs["top_p"], "must be in [0.0, 1.0]")
        self._inference.update(kwargs)
        return self

    def with_server(self, **kwargs) -> "SmartGPUConfigBuilder":
        if "port" in kwargs and not 1 <= kwargs["port"] <= 65535:
            raise ConfigValidationError("server.port", kwargs["port"], "must be in [1, 65535]")
        self._server.update(kwargs)
        return self

    def with_logging(self, **kwargs) -> "SmartGPUConfigBuilder":
        if "level" in kwargs:
            kwargs["level"] = kwargs["level"].upper()
        self._logging_cfg.update(kwargs)
        return self

    def with_model_dir(self, path: str) -> "SmartGPUConfigBuilder":
        self._model_dir = os.path.expanduser(path)
        return self

    def with_model_path(self, path: str) -> "SmartGPUConfigBuilder":
        self._model_paths.append(os.path.expanduser(path))
        return self

    def with_chat_template(self, template: str) -> "SmartGPUConfigBuilder":
        self._chat_template = template
        return self

    def build(self) -> SmartGPUConfig:
        return SmartGPUConfig(
            gpu=GPUConfig(**self._gpu) if self._gpu else GPUConfig(),
            performance=PerformanceConfig(**self._performance) if self._performance else PerformanceConfig(),
            inference=InferenceConfig(**self._inference) if self._inference else InferenceConfig(),
            server=ServerConfig(**self._server) if self._server else ServerConfig(),
            logging=LoggingConfig(**self._logging_cfg) if self._logging_cfg else LoggingConfig(),
            model_paths=self._model_paths,
            model_dir=self._model_dir,
            chat_template=self._chat_template,
        )




# ==============================================================================
# READ-WRITE LOCK
# ==============================================================================

class ReadWriteLock:
    """A readers-writer lock allowing concurrent reads but exclusive writes.

    Multiple threads can hold the read lock simultaneously, but only one
    thread can hold the write lock (and no readers while writing).

    Example::

        rwl = ReadWriteLock()
        with rwl.read():     # multiple concurrent readers ok
            data = shared_resource
        with rwl.write():    # exclusive access
            shared_resource += 1
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._readers = threading.Condition(self._lock)
        self._reader_count = 0
        self._writer = False

    def read(self):
        """Acquire read lock (shared). Context manager."""
        return _ReadLockContext(self)

    def write(self):
        """Acquire write lock (exclusive). Context manager."""
        return _WriteLockContext(self)


class _ReadLockContext:
    __slots__ = ("_rwl",)

    def __init__(self, rwl: ReadWriteLock):
        self._rwl = rwl

    def __enter__(self):
        with self._rwl._lock:
            while self._rwl._writer:
                self._rwl._readers.wait()
            self._rwl._reader_count += 1
        return self

    def __exit__(self, *args):
        with self._rwl._lock:
            self._rwl._reader_count -= 1
            if self._rwl._reader_count == 0:
                self._rwl._readers.notify_all()
        return False


class _WriteLockContext:
    __slots__ = ("_rwl",)

    def __init__(self, rwl: ReadWriteLock):
        self._rwl = rwl

    def __enter__(self):
        with self._rwl._lock:
            while self._rwl._writer or self._rwl._reader_count > 0:
                self._rwl._readers.wait()
            self._rwl._writer = True
        return self

    def __exit__(self, *args):
        with self._rwl._lock:
            self._rwl._writer = False
            self._rwl._readers.notify_all()
        return False


# ==============================================================================
# SERVICE CONTAINER (Dependency Injection)
# ==============================================================================

class ServiceContainer:
    """Simple dependency injection container for shared application components.

    Allows components to be registered and retrieved by type or name.
    Supports lazy initialization via factory functions.

    Example::

        container = ServiceContainer()
        container.register(SmartGPUManager, gpu_manager)
        container.register("perf", PerformanceMonitor)

        gpu = container.resolve(SmartGPUManager)
        perf = container.resolve("perf")
    """

    def __init__(self):
        self._services: Dict[Any, Any] = {}
        self._factories: Dict[Any, Callable[..., Any]] = {}
        self._lock = threading.Lock()

    def register(self, key: Any, instance: Any) -> None:
        """Register a concrete instance."""
        with self._lock:
            self._services[key] = instance
            self._factories.pop(key, None)

    def register_factory(self, key: Any, factory: Callable[..., Any]) -> None:
        """Register a factory for lazy initialization."""
        with self._lock:
            self._factories[key] = factory
            self._services.pop(key, None)

    def resolve(self, key: Any) -> Any:
        """Resolve a service by key. Creates via factory if needed."""
        with self._lock:
            if key in self._services:
                return self._services[key]
            if key in self._factories:
                instance = self._factories[key]()
                self._services[key] = instance
                return instance
        raise KeyError(f"Service not registered: {key!r}")

    def has(self, key: Any) -> bool:
        with self._lock:
            return key in self._services or key in self._factories

    def reset(self) -> None:
        """Clear all registrations (useful for testing)."""
        with self._lock:
            self._services.clear()
            self._factories.clear()


# ==============================================================================
# @deprecated DECORATOR
# ==============================================================================

def deprecated(reason: str = "", version: str = "") -> Callable[[F], F]:
    """Mark a function or method as deprecated.

    Emits a DeprecationWarning on each call.

    Example::

        @deprecated("Use new_api() instead", version="3.0")
        def old_api(): ...
    """
    def decorator(fn: F) -> F:
        msg = f"{fn.__qualname__} is deprecated"
        if version:
            msg += f" (since v{version})"
        if reason:
            msg += f": {reason}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            import warnings
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# ==============================================================================
# @rate_limit DECORATOR
# ==============================================================================

def rate_limit(limiter: TokenBucketRateLimiter, tokens: int = 1,
               timeout: Optional[Seconds] = None):
    """Decorator that applies token-bucket rate limiting to a function.

    Example::

        api_limiter = TokenBucketRateLimiter(rate=30, capacity=60)

        @rate_limit(api_limiter, timeout=5.0)
        def handle_request():
            ...
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not limiter.acquire(tokens=tokens, timeout=timeout):
                raise LlamaSmartGPUError(
                    message="Rate limit exceeded",
                    code="RATE_LIMITED", component="RateLimiter",
                )
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# ==============================================================================
# INFERENCE SLOT CONTEXT MANAGER
# ==============================================================================

class SlotContext:
    """Context manager that acquires an inference slot and releases it on exit.

    Example::

        with SlotContext(engine, session_id="abc") as slot:
            # slot.id is available
            result = engine.generate(prompt)
        # slot automatically released
    """

    def __init__(self, engine: "LlamaEngine", session_id: Optional[str] = None):
        self._engine = engine
        self._session_id = session_id
        self.slot: Optional[SlotState] = None

    def __enter__(self) -> "SlotContext":
        self.slot = self._engine.acquire_slot(self._session_id)
        if self.slot is None:
            raise LlamaSmartGPUError(
                message="No inference slots available (all busy)",
                code="NO_SLOTS", component="SlotContext",
            )
        return self

    def __exit__(self, *args):
        if self.slot is not None:
            self._engine.release_slot(self.slot.id)
            self.slot = None
        return False


# ==============================================================================
# MODEL CONTEXT MANAGER (temporary model loading)
# ==============================================================================

class ModelContext:
    """Context manager for temporarily loading a model.

    Loads the model on enter, unloads on exit. Useful for batch processing
    or one-shot scripts where you want guaranteed cleanup.

    Example::

        with ModelContext(engine, "/path/to/model.gguf") as info:
            result = engine.generate("Hello")
        # model is automatically unloaded
    """

    def __init__(self, engine: "LlamaEngine", model_path: str, **load_kwargs):
        self._engine = engine
        self._model_path = model_path
        self._load_kwargs = load_kwargs
        self.info: Optional[ModelInfo] = None
        self._was_loaded = engine.is_loaded

    def __enter__(self) -> "ModelContext":
        self.info = self._engine.load_model(self._model_path, **self._load_kwargs)
        return self

    def __exit__(self, *args):
        self._engine.unload_model()
        return False

# ==============================================================================
# END PATTERNS INFRASTRUCTURE
# ==============================================================================



# ==============================================================================
# SECTION: config.py
# ==============================================================================

"""
Configuration management for llama.cpp smart GPU inference engine.
Supports YAML, JSON, ENV, and CLI overrides with layered priority.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# ─── Defaults ───────────────────────────────────────────────────────────────

_DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/llama-smart-gpu/models")
_DEFAULT_CONFIG_PATH = os.path.expanduser("~/.cache/llama-smart-gpu/config.json")


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class GPUConfig:
    """GPU and VRAM management settings."""
    enabled: bool = True
    device_index: int = 0
    # VRAM budget in MB; 0 = auto-detect from hardware
    vram_budget_mb: int = 0
    # Reserve this much VRAM (MB) for system / other processes
    vram_reserve_mb: int = 512
    # Target fraction of VRAM to use (0.0–1.0) when auto-detecting
    vram_target_fraction: float = 0.90
    # Number of GPU layers to offload; -1 = auto-calculate from VRAM
    n_gpu_layers: int = -1
    # Split tensors across GPU for large models
    tensor_split: Optional[List[float]] = None
    # Use memory-mapped file for model weights (reduces RAM)
    use_mmap: bool = True
    # Use memory-locked allocation (may improve perf, uses more RAM)
    use_mlock: bool = False
    # Enable CUDA graphs for faster inference on repeated patterns
    use_cublas: bool = True


@dataclass(slots=True)
class PerformanceConfig:
    """Performance tuning knobs."""
    # Number of threads for CPU fallback / batch processing
    n_threads: int = 0  # 0 = auto (physical cores)
    n_threads_batch: int = 0  # 0 = same as n_threads
    # Batch size for prompt evaluation
    n_batch: int = 512
    # Context window size (tokens)
    n_ctx: int = 4096
    # Rope scaling type: 0=none, 1=linear, 2=ytz
    rope_scaling_type: int = 0
    # Rope freq base (0 = model default)
    rope_freq_base: float = 0.0
    # Cache type for KV: "f16", "q8_0", "q4_0", "q4_1"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    # Enable continuous batching / slot-based inference
    continuous_batching: bool = True
    # Max concurrent slots for continuous batching
    n_slots: int = 4
    # Mlock the KV cache to avoid swap
    cache_mlock: bool = False
    # Quantize KV cache to save VRAM
    quantize_kv_cache: bool = False


@dataclass
class InferenceConfig:
    """Inference generation parameters (defaults, overridable per-request)."""
    n_predict: int = 512
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Stop sequences
    stop: List[str] = field(default_factory=lambda: [])
    # Seed for reproducibility; -1 = random
    seed: int = -1
    # Streaming mode default
    stream: bool = True

    def __post_init__(self):
        """Validate inference parameter ranges at construction time."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigValidationError("inference.temperature", self.temperature,
                                          "must be in [0.0, 2.0]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ConfigValidationError("inference.top_p", self.top_p,
                                          "must be in [0.0, 1.0]")
        if self.n_predict < 1:
            raise ConfigValidationError("inference.n_predict", self.n_predict,
                                          "must be >= 1")
        if self.top_k < 0:
            raise ConfigValidationError("inference.top_k", self.top_k,
                                          "must be >= 0")


@dataclass(slots=True)
class ServerConfig:
    """API server settings."""
    host: str = "0.0.0.0"
    port: int = 8080
    # Max request body size in MB
    max_body_size_mb: int = 100
    # API key for authentication (empty = no auth)
    api_key: str = ""
    # CORS allowed origins
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    # Request timeout in seconds
    timeout_seconds: int = 300
    # Enable metrics endpoint at /metrics
    enable_metrics: bool = True


@dataclass(slots=True)
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None
    # Rotate log file at this size (MB)
    max_file_size_mb: int = 50
    # Keep this many rotated log files
    max_backups: int = 5


@dataclass(slots=True)
class SmartGPUConfig:
    """Top-level configuration combining all sub-configs."""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Model paths (can be multiple; first that loads wins)
    model_paths: List[str] = field(default_factory=list)
    # Directory to search for models
    model_dir: str = _DEFAULT_MODEL_DIR
    # Chat template override (empty = auto-detect from model)
    chat_template: str = ""


# ─── Helpers ────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _env_overrides() -> dict:
    """Read configuration overrides from environment variables.

    Convention: LLAMA_GPU_VRAM_BUDGET_MB=8192 → gpu.vram_budget_mb = 8192
    """
    prefix = "LLAMA_"
    overrides: Dict[str, Any] = {}
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith(prefix):
            continue
        key = raw_key[len(prefix):].lower()
        parts = key.split("_", 1)
        if len(parts) == 1:
            # Top-level key
            overrides[parts[0]] = _auto_cast(raw_val)
        else:
            # Nested key e.g. gpu_vram_budget_mb → {"gpu": {"vram_budget_mb": ...}}
            section, subkey = parts
            overrides.setdefault(section, {})[subkey] = _auto_cast(raw_val)
    return overrides


def _auto_cast(value: str) -> Any:
    """Attempt to cast a string to int / float / bool, falling back to str."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Handle comma-separated lists
    if "," in value:
        return [_auto_cast(v.strip()) for v in value.split(",")]
    return value


def _dict_to_config(data: dict) -> SmartGPUConfig:
    """Convert a nested dict into a SmartGPUConfig dataclass tree."""
    gpu = GPUConfig(**data.get("gpu", {}))
    perf = PerformanceConfig(**data.get("performance", {}))
    inf = InferenceConfig(**data.get("inference", {}))
    srv = ServerConfig(**data.get("server", {}))
    log = LoggingConfig(**data.get("logging", {}))
    return SmartGPUConfig(
        gpu=gpu,
        performance=perf,
        inference=inf,
        server=srv,
        logging=log,
        model_paths=data.get("model_paths", []),
        model_dir=data.get("model_dir", _DEFAULT_MODEL_DIR),
        chat_template=data.get("chat_template", ""),
    )


# ─── Public API ─────────────────────────────────────────────────────────────

def load_config(path: Optional[str] = None) -> SmartGPUConfig:
    """Load configuration with layered priority:

    1. Built-in defaults
    2. Config file (JSON)
    3. Environment variables (LLAMA_*)
    4. CLI overrides (not handled here; caller merges)
    """
    # 1. Start from defaults
    config_dict = asdict(SmartGPUConfig())

    # 2. Merge file config
    config_path = path or os.environ.get("LLAMA_CONFIG", _DEFAULT_CONFIG_PATH)
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                file_data = json.load(f)
            config_dict = _deep_merge(config_dict, file_data)
            logger.info("Loaded config from %s", config_file)
        except Exception as exc:
            logger.warning("Failed to load config file %s: %s", config_file, exc)

    # 3. Merge environment overrides
    env_data = _env_overrides()
    if env_data:
        config_dict = _deep_merge(config_dict, env_data)
        logger.debug("Applied %d env overrides", len(env_data))

    return _dict_to_config(config_dict)


def save_config(cfg: SmartGPUConfig, path: Optional[str] = None) -> str:
    """Persist current config to a JSON file."""
    config_path = path or os.environ.get("LLAMA_CONFIG", _DEFAULT_CONFIG_PATH)
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    logger.info("Config saved to %s", config_file)
    return str(config_file)


def apply_cli_overrides(cfg: SmartGPUConfig, overrides: Dict[str, Any]) -> SmartGPUConfig:
    """Apply flat CLI overrides like {"gpu.n_gpu_layers": 40}."""
    current = asdict(cfg)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = current
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return _dict_to_config(current)


# ==============================================================================
# SECTION: gpu_manager.py
# ==============================================================================

"""
Smart GPU RAM / VRAM Manager for llama.cpp inference.

Responsibilities:
  - Detect available GPU hardware and VRAM
  - Calculate optimal layer offloading strategy
  - Monitor real-time VRAM usage
  - Dynamically adjust GPU layers when VRAM pressure detected
  - Quantize KV cache on-the-fly when approaching limits
  - Provide detailed memory telemetry
"""


# ─── Enums & Data ───────────────────────────────────────────────────────────

class VRAMPressure(Enum):
    """VRAM pressure levels for adaptive strategies."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def description(self) -> str:
        _d = {VRAMPressure.LOW: "Normal operation, plenty of VRAM headroom",
              VRAMPressure.MODERATE: "Approaching capacity, monitor closely",
              VRAMPressure.HIGH: "High pressure, consider reducing GPU layers",
              VRAMPressure.CRITICAL: "Critical pressure, immediate action needed"}
        return _d[self]

    @property
    def severity(self) -> int:
        return {VRAMPressure.LOW: 0, VRAMPressure.MODERATE: 1,
                VRAMPressure.HIGH: 2, VRAMPressure.CRITICAL: 3}[self]

    def __str__(self) -> str:
        return f"VRAMPressure.{self.name}"


@dataclass(slots=True)
class GPUMemoryInfo:
    """Snapshot of GPU memory state."""

    def __repr__(self) -> str:
        return (f"GPUMemoryInfo(gpu={self.gpu_name!r}, vram={self.used_vram_mb:.0f}/{self.total_vram_mb:.0f} MB, "
                f"ram={self.used_ram_mb:.0f}/{self.total_ram_mb:.0f} MB)")
    total_vram_mb: float = 0.0
    used_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    total_ram_mb: float = 0.0
    used_ram_mb: float = 0.0
    free_ram_mb: float = 0.0
    gpu_name: str = "unknown"
    gpu_utilization_pct: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_usage_w: float = 0.0
    timestamp: float = 0.0


@dataclass
class LayerOffloadPlan:
    """Plan for which layers go on GPU vs CPU."""
    total_layers: int = 0
    gpu_layers: int = 0
    cpu_layers: int = 0
    estimated_vram_mb: float = 0.0
    estimated_ram_mb: float = 0.0
    kv_cache_vram_mb: float = 0.0
    confidence: float = 1.0  # 0-1 how confident the estimate is
    strategy: str = "balanced"


@dataclass(slots=True)
class MemoryAlert:
    """Alert when memory pressure crosses thresholds."""
    level: VRAMPressure
    message: str
    vram_used_pct: float
    timestamp: float


# ─── VRAM Estimator ─────────────────────────────────────────────────────────

class VRAMEstimator:
    """Estimates VRAM requirements for a model based on its parameters.

    Uses heuristics derived from llama.cpp memory characteristics:
      - FP16 weights: 2 bytes per parameter
      - Q4_0 quant: ~0.5 bytes per parameter (varies by quant)
      - KV cache: 2 bytes per token per layer per head (FP16)
      - Overhead: ~5-10% for CUDA kernels, workspace, etc.
    """

    # Bytes per parameter for common quantization formats
    QUANT_BYTES = {
        "f16": 2.0,
        "q8_0": 1.06,
        "q5_1": 0.75,
        "q5_0": 0.69,
        "q4_1": 0.65,
        "q4_0": 0.56,
        "q3_k_s": 0.44,
        "q3_k_m": 0.49,
        "q4_k_m": 0.63,
        "q5_k_m": 0.77,
        "q6_k": 0.85,
        "q8_0": 1.06,
    }

    # Typical layer distribution for transformer models
    ATTENTION_PARAMS_FRACTION = 0.33
    FEEDFORWARD_PARAMS_FRACTION = 0.67

    @classmethod
    def estimate_model_vram(cls, n_params_b: float, n_layers: int,
                            quant: str = "q4_k_m",
                            n_gpu_layers: int = -1,
                            n_ctx: int = 4096,
                            n_heads: int = 32,
                            n_kv_heads: Optional[int] = None) -> float:
        """Estimate VRAM needed in MB for a given model configuration.

        Args:
            n_params_b: Total parameters in billions
            n_layers: Number of transformer layers
            quant: Quantization format name
            n_gpu_layers: Layers to put on GPU (-1 = all)
            n_ctx: Context window size (tokens)
            n_heads: Number of attention heads
            n_kv_heads: Number of KV heads (for GQA; defaults to n_heads)

        Returns:
            Estimated VRAM in MB
        """
        bytes_per_param = cls.QUANT_BYTES.get(quant.lower(), 0.63)

        # Model weights on GPU
        if n_gpu_layers == -1:
            gpu_layer_fraction = 1.0
        else:
            gpu_layer_fraction = min(n_gpu_layers / max(n_layers, 1), 1.0)

        weights_mb = n_params_b * 1e9 * bytes_per_param * gpu_layer_fraction / (1024 ** 2)

        # KV cache (per-token, on GPU)
        effective_kv_heads = n_kv_heads or n_heads
        head_dim = (n_params_b * 1e9 * cls.ATTENTION_PARAMS_FRACTION) / (n_layers * n_heads * 2)  # rough
        kv_per_token_bytes = 2 * n_layers * effective_kv_heads * max(head_dim, 128) * 2  # K+V, fp16
        kv_cache_mb = n_ctx * kv_per_token_bytes / (1024 ** 2)

        # CUDA overhead (~200MB base + workspace)
        cuda_overhead_mb = 200 + (weights_mb * 0.08)

        total = weights_mb + kv_cache_mb + cuda_overhead_mb
        logger.debug(
            "VRAM estimate: weights=%.0fMB, kv_cache=%.0fMB, overhead=%.0fMB, total=%.0fMB",
            weights_mb, kv_cache_mb, cuda_overhead_mb, total,
        )
        return total

    @classmethod
    def compute_optimal_gpu_layers(cls, n_params_b: float, n_layers: int,
                                   available_vram_mb: float,
                                   quant: str = "q4_k_m",
                                   n_ctx: int = 4096,
                                   n_heads: int = 32,
                                   reserve_mb: float = 512) -> LayerOffloadPlan:
        """Binary-search for the maximum number of GPU layers that fit in VRAM.

        Starts from all-layers and reduces until the estimate fits within
        (available_vram - reserve).
        """
        target_vram = available_vram_mb - reserve_mb
        if target_vram <= 0:
            return LayerOffloadPlan(
                total_layers=n_layers, gpu_layers=0, cpu_layers=n_layers,
                estimated_vram_mb=0, estimated_ram_mb=n_params_b * 1e9 * 0.63 / (1024**2),
                kv_cache_vram_mb=0, confidence=1.0, strategy="cpu_only",
            )

        # Check if all layers fit
        full_vram = cls.estimate_model_vram(n_params_b, n_layers, quant, n_layers, n_ctx, n_heads)
        if full_vram <= target_vram:
            return LayerOffloadPlan(
                total_layers=n_layers, gpu_layers=n_layers, cpu_layers=0,
                estimated_vram_mb=full_vram, estimated_ram_mb=0,
                kv_cache_vram_mb=0, confidence=0.9, strategy="full_gpu",
            )

        # Binary search for optimal split
        lo, hi = 0, n_layers
        best = 0
        best_vram = 0.0
        while lo <= hi:
            mid = (lo + hi) // 2
            vram = cls.estimate_model_vram(n_params_b, n_layers, quant, mid, n_ctx, n_heads)
            if vram <= target_vram:
                best = mid
                best_vram = vram
                lo = mid + 1
            else:
                hi = mid - 1

        # Compute CPU-side RAM for remaining layers
        bytes_per_param = cls.QUANT_BYTES.get(quant.lower(), 0.63)
        cpu_ram_mb = n_params_b * 1e9 * bytes_per_param * (1 - best / max(n_layers, 1)) / (1024 ** 2)

        strategy = "partial_offload" if best > 0 else "cpu_only"
        if best == n_layers:
            strategy = "full_gpu"

        return LayerOffloadPlan(
            total_layers=n_layers, gpu_layers=best, cpu_layers=n_layers - best,
            estimated_vram_mb=best_vram, estimated_ram_mb=cpu_ram_mb,
            kv_cache_vram_mb=0, confidence=0.8, strategy=strategy,
        )


# ─── GPU Detector ───────────────────────────────────────────────────────────

class GPUDetector:
    """Detect GPU hardware and query memory information.

    Tries multiple backends:
      1. pynvml (NVIDIA NVML) — most accurate
      2. llama-cpp-python internal — fallback
      3. /proc/meminfo + nvidia-smi CLI — last resort
    """

    @retry(max_attempts=2, base_delay=0.5, retryable_exceptions=(OSError, RuntimeError))
    @staticmethod
    def detect() -> GPUMemoryInfo:
        """Return current GPU and system memory state."""
        info = GPUMemoryInfo(timestamp=time.time())

        # Try pynvml first
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info.total_vram_mb = mem.total / (1024 ** 2)
            info.used_vram_mb = mem.used / (1024 ** 2)
            info.free_vram_mb = mem.free / (1024 ** 2)
            info.gpu_name = pynvml.nvmlDeviceGetName(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info.gpu_utilization_pct = util.gpu
            except Exception:
                pass
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                info.gpu_temperature_c = temp
            except Exception:
                pass
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                info.gpu_power_usage_w = power / 1000.0
            except Exception:
                pass
            pynvml.nvmlShutdown()
            logger.debug("GPU detected via pynvml: %s (%.0f MB VRAM)", info.gpu_name, info.total_vram_mb)
        except ImportError:
            logger.debug("pynvml not available, trying nvidia-smi fallback")
            info = GPUDetector._nvidia_smi_fallback()
        except Exception as exc:
            logger.warning("pynvml failed: %s, trying fallback", exc)
            info = GPUDetector._nvidia_smi_fallback()

        # System RAM (always available on Linux)
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info.total_ram_mb = int(line.split()[1]) / 1024
                    elif line.startswith("MemAvailable:"):
                        info.free_ram_mb = int(line.split()[1]) / 1024
            info.used_ram_mb = info.total_ram_mb - info.free_ram_mb
        except Exception:
            pass

        return info

    @staticmethod
    def _nvidia_smi_fallback() -> GPUMemoryInfo:
        """Parse nvidia-smi output as a fallback GPU detection method."""
        info = GPUMemoryInfo(timestamp=time.time())
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    info.gpu_name = parts[0]
                    info.total_vram_mb = float(parts[1])
                    info.used_vram_mb = float(parts[2])
                    info.free_vram_mb = float(parts[3])
                if len(parts) >= 5:
                    info.gpu_utilization_pct = float(parts[4])
                if len(parts) >= 6:
                    info.gpu_temperature_c = float(parts[5])
                if len(parts) >= 7:
                    info.gpu_power_usage_w = float(parts[6])
        except Exception as exc:
            logger.warning("nvidia-smi fallback failed: %s", exc)
        return info

    @staticmethod
    def list_gpus() -> List[Dict]:
        """List all available GPUs with their details."""
        gpus = []
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({
                    "index": i,
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "total_vram_mb": mem.total / (1024 ** 2),
                    "free_vram_mb": mem.free / (1024 ** 2),
                })
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return gpus


# ─── Smart GPU Manager ──────────────────────────────────────────────────────

class SmartGPUManager:
    """High-level manager that orchestrates GPU memory decisions.

    Features:
      - Periodic VRAM monitoring with alerts
      - Automatic layer offload adjustment under pressure
      - KV cache quantization trigger
      - Callback system for engine adaptation
    """

    def __init__(self, vram_budget_mb: float = 0, vram_reserve_mb: float = 512,
                 vram_target_fraction: float = 0.90,
                 monitor_interval_seconds: float = 5.0):
        self._budget_mb = vram_budget_mb
        self._reserve_mb = vram_reserve_mb
        self._target_fraction = vram_target_fraction
        self._monitor_interval = monitor_interval_seconds

        self._detector = GPUDetector()
        self._estimator = VRAMEstimator()

        self._current_info = GPUMemoryInfo()
        self._alerts: List[MemoryAlert] = []
        self._callbacks: Dict[VRAMPressure, List[Callable]] = {
            level: [] for level in VRAMPressure
        }
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Current offload plan
        self._current_plan: Optional[LayerOffloadPlan] = None

        # Performance stats
        self._stats = {
            "total_monitor_cycles": 0,
            "pressure_transitions": 0,
            "layer_adjustments": 0,
            "kv_quantize_events": 0,
            "last_pressure": VRAMPressure.LOW,
        }

    # ── Public API ──────────────────────────────────────────────────────

    @log_calls(log_result=True)
    def initialize(self) -> GPUMemoryInfo:
        """Detect hardware and set baseline. Call once at startup."""
        info = self._detector.detect()
        self._current_info = info

        # If budget not explicitly set, auto-detect from hardware
        if self._budget_mb <= 0 and info.total_vram_mb > 0:
            self._budget_mb = info.total_vram_mb * self._target_fraction
            logger.info(
                "Auto-detected VRAM budget: %.0f MB (total=%.0f MB, target=%.0f%%)",
                self._budget_mb, info.total_vram_mb, self._target_fraction * 100,
            )

        logger.info(
            "GPU initialized: %s | VRAM: %.0f/%.0f MB | RAM: %.0f/%.0f MB",
            info.gpu_name, info.used_vram_mb, info.total_vram_mb,
            info.used_ram_mb, info.total_ram_mb,
        )
        return info

    def compute_offload_plan(self, n_params_b: float, n_layers: int,
                             quant: str = "q4_k_m",
                             n_ctx: int = 4096,
                             n_heads: int = 32) -> LayerOffloadPlan:
        """Compute optimal GPU layer offloading plan for a model."""
        available = self._budget_mb - self._reserve_mb if self._budget_mb > 0 else 0
        plan = self._estimator.compute_optimal_gpu_layers(
            n_params_b, n_layers, max(available, 0), quant, n_ctx, n_heads, self._reserve_mb,
        )
        self._current_plan = plan
        logger.info(
            "Offload plan: %d/%d layers on GPU | est. VRAM=%.0f MB | strategy=%s",
            plan.gpu_layers, plan.total_layers, plan.estimated_vram_mb, plan.strategy,
        )
        return plan

    def get_available_vram(self) -> float:
        """Get currently free VRAM in MB."""
        info = self._detector.detect()
        return info.free_vram_mb

    def get_memory_info(self) -> GPUMemoryInfo:
        """Get a fresh snapshot of GPU and system memory."""
        self._current_info = self._detector.detect()
        return self._current_info

    def get_pressure(self) -> VRAMPressure:
        """Classify current VRAM pressure level."""
        info = self.get_memory_info()
        if info.total_vram_mb <= 0:
            return VRAMPressure.LOW
        used_pct = (info.used_vram_mb / info.total_vram_mb) * 100
        if used_pct < 50:
            return VRAMPressure.LOW
        elif used_pct < 75:
            return VRAMPressure.MODERATE
        elif used_pct < 90:
            return VRAMPressure.HIGH
        else:
            return VRAMPressure.CRITICAL

    def should_quantize_kv(self) -> bool:
        """Check if KV cache quantization should be enabled based on pressure."""
        pressure = self.get_pressure()
        return pressure in (VRAMPressure.HIGH, VRAMPressure.CRITICAL)

    def recommend_n_gpu_layers(self, current_layers: int) -> int:
        """Dynamically recommend GPU layer count based on current VRAM state.

        If pressure is critical, reduce layers. If pressure is low, suggest
        increasing layers for better performance.
        """
        pressure = self.get_pressure()
        if pressure == VRAMPressure.CRITICAL:
            new_layers = max(current_layers - 4, 0)
            if new_layers != current_layers:
                logger.warning(
                    "VRAM CRITICAL — reducing GPU layers %d → %d",
                    current_layers, new_layers,
                )
                with self._lock:
                    self._stats["layer_adjustments"] += 1
            return new_layers
        elif pressure == VRAMPressure.HIGH:
            new_layers = max(current_layers - 2, 0)
            if new_layers != current_layers:
                logger.info(
                    "VRAM HIGH — reducing GPU layers %d → %d",
                    current_layers, new_layers,
                )
                with self._lock:
                    self._stats["layer_adjustments"] += 1
            return new_layers
        elif pressure == VRAMPressure.LOW and self._current_plan:
            # Can we afford more layers?
            if current_layers < self._current_plan.gpu_layers:
                logger.info(
                    "VRAM LOW — suggesting GPU layers %d → %d",
                    current_layers, self._current_plan.gpu_layers,
                )
                return self._current_plan.gpu_layers
        return current_layers

    # ── Monitoring ───────────────────────────────────────────────────────

    def start_monitoring(self):
        """Start background VRAM monitoring thread."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="vram-monitor",
        )
        self._monitor_thread.start()
        logger.info("VRAM monitoring started (interval=%.1fs)", self._monitor_interval)

    def stop_monitoring(self):
        """Stop the background monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("VRAM monitoring stopped")

    def register_callback(self, pressure_level: VRAMPressure, callback: Callable):
        """Register a callback to fire when a pressure level is reached."""
        self._callbacks[pressure_level].append(callback)

    def get_stats(self) -> Dict:
        """Return monitoring statistics."""
        with self._lock:
            stats = self._stats.copy()
        stats["current_vram_budget_mb"] = self._budget_mb
        stats["current_pressure"] = self.get_pressure().value
        if self._current_plan:
            stats["gpu_layers"] = self._current_plan.gpu_layers
            stats["cpu_layers"] = self._current_plan.cpu_layers
            stats["strategy"] = self._current_plan.strategy
        return stats

    # ── Internal ─────────────────────────────────────────────────────────

    def _monitor_loop(self):
        """Background loop that checks VRAM and fires alerts."""
        while self._running:
            try:
                pressure = self.get_pressure()
                info = self._current_info

                with self._lock:
                    self._stats["total_monitor_cycles"] += 1
                    prev = self._stats["last_pressure"]
                    if pressure != prev:
                        self._stats["pressure_transitions"] += 1
                        self._stats["last_pressure"] = pressure

                        alert = MemoryAlert(
                            level=pressure,
                            message=f"VRAM pressure changed: {prev.value} → {pressure.value} "
                                    f"(used={info.used_vram_mb:.0f}/{info.total_vram_mb:.0f} MB)",
                            vram_used_pct=(info.used_vram_mb / max(info.total_vram_mb, 1)) * 100,
                            timestamp=time.time(),
                        )
                        self._alerts.append(alert)
                        logger.info(alert.message)

                    # Fire callbacks
                    for cb in self._callbacks.get(pressure, []):
                        try:
                            cb(pressure, info)
                        except Exception as exc:
                            logger.error("Pressure callback error: %s", exc)

            except Exception as exc:
                logger.error("Monitor loop error: %s", exc)

            time.sleep(self._monitor_interval)

    def get_alerts(self, limit: int = 50) -> List[MemoryAlert]:
        """Return recent memory alerts."""
        return self._alerts[-limit:]


# ==============================================================================
# SECTION: performance.py
# ==============================================================================

"""
Performance Monitoring and Metrics System.

Provides:
  - Real-time throughput and latency tracking
  - Rolling window statistics (p50, p90, p99)
  - Generation timing breakdown (prompt eval vs generation)
  - GPU utilization and VRAM efficiency metrics
  - Prometheus-compatible /metrics endpoint output
  - Periodic health reports
  - Performance regression detection
"""


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class GenerationMetrics:
    """Metrics from a single generation call."""
    timestamp: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_eval_tps: float = 0.0  # tokens/s for prompt processing
    finish_reason: str = ""
    slot_id: int = 0


@dataclass(slots=True)
class WindowStats:
    """Statistics computed over a rolling window."""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    std_dev: float = 0.0
    sum_val: float = 0.0


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    uptime_seconds: float = 0.0
    total_generations: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0

    # Throughput
    avg_tokens_per_second: float = 0.0
    avg_prompt_eval_tps: float = 0.0

    # Latency (generation only)
    latency_gen_ms: WindowStats = field(default_factory=WindowStats)
    latency_total_ms: WindowStats = field(default_factory=WindowStats)

    # Token throughput over windows
    throughput_tps: WindowStats = field(default_factory=WindowStats)

    # GPU metrics
    gpu_name: str = ""
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_utilization_pct: float = 0.0
    gpu_utilization_pct: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_w: float = 0.0

    # Efficiency
    vram_per_token_mb: float = 0.0  # VRAM efficiency
    avg_context_usage_pct: float = 0.0

    # Health
    health_status: str = "healthy"
    warnings: List[str] = field(default_factory=list)


class HealthStatus(Enum):
    """System health status based on aggregated metrics."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

    @property
    def severity(self) -> int:
        return {HealthStatus.HEALTHY: 0, HealthStatus.DEGRADED: 1, HealthStatus.CRITICAL: 2}[self]

    def __gt__(self, other): return self.severity > other.severity

    def __str__(self): return f"HealthStatus.{self.name}"


# ─── Rolling Window Stats ───────────────────────────────────────────────────

class RollingWindow:
    """Efficient rolling window for numeric values.

    Maintains a fixed-size deque and computes statistics in O(n log n)
    for percentiles (sorted copy) and O(n) for others.
    """

    def __init__(self, max_size: int = 1000):
        self._data: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, value: float):
        with self._lock:
            self._data.append(value)

    def stats(self) -> WindowStats:
        with self._lock:
            data = list(self._data)

        if not data:
            return WindowStats()

        data.sort()
        n = len(data)
        s = WindowStats(
            count=n,
            min_val=data[0],
            max_val=data[-1],
            sum_val=sum(data),
            median=data[n // 2],
            p90=data[int(n * 0.90)] if n > 1 else data[0],
            p95=data[int(n * 0.95)] if n > 1 else data[0],
            p99=data[int(n * 0.99)] if n > 1 else data[0],
        )
        s.mean = s.sum_val / n
        if n > 1:
            variance = sum((x - s.mean) ** 2 for x in data) / (n - 1)
            s.std_dev = math.sqrt(variance)
        return s

    def clear(self):
        with self._lock:
            self._data.clear()


# ─── Performance Monitor ────────────────────────────────────────────────────

class PerformanceMonitor:
    """Central performance monitoring system.

    Collects generation metrics, computes rolling statistics,
    generates reports, and exposes Prometheus-compatible metrics.
    """

    def __init__(self, window_size: int = 1000, report_interval: float = 60.0):
        self._window_size = window_size
        self._report_interval = report_interval

        # Rolling windows
        self._gen_latency = RollingWindow(window_size)       # generation ms
        self._total_latency = RollingWindow(window_size)     # total ms
        self._throughput = RollingWindow(window_size)        # tokens/s
        self._prompt_eval_tps = RollingWindow(window_size)   # prompt eval tokens/s
        self._prompt_tokens = RollingWindow(window_size)      # per-request prompt tokens
        self._completion_tokens = RollingWindow(window_size)  # per-request completion tokens

        # Counters
        self._total_generations = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._start_time = time.time()

        # Recent metrics
        self._recent: deque = deque(maxlen=100)

        # Callbacks for performance alerts
        self._alert_callbacks: List[Callable] = []

        # Background reporting
        self._running = False
        self._report_thread: Optional[threading.Thread] = None

        # Regression detection
        self._baseline_tps: Optional[float] = None
        self._regression_threshold = 0.7  # alert if <70% of baseline

    # ── Recording ────────────────────────────────────────────────────────

    def record_generation(self, metrics: GenerationMetrics):
        """Record metrics from a completed generation."""
        self._total_generations += 1
        self._total_tokens += metrics.completion_tokens
        self._total_prompt_tokens += metrics.prompt_tokens

        if metrics.generation_ms > 0:
            self._gen_latency.add(metrics.generation_ms)
        if metrics.total_ms > 0:
            self._total_latency.add(metrics.total_ms)
        if metrics.tokens_per_second > 0:
            self._throughput.add(metrics.tokens_per_second)
        if metrics.prompt_eval_tps > 0:
            self._prompt_eval_tps.add(metrics.prompt_eval_tps)
        if metrics.prompt_tokens > 0:
            self._prompt_tokens.add(metrics.prompt_tokens)
        if metrics.completion_tokens > 0:
            self._completion_tokens.add(metrics.completion_tokens)

        self._recent.append(metrics)

        # Check for performance regression
        self._check_regression(metrics.tokens_per_second)

    def record_from_result(self, result, prompt_tokens: int = 0):
        """Convenience: record from a GenerationResult + engine telemetry."""
        metrics = GenerationMetrics(
            timestamp=time.time(),
            prompt_tokens=result.prompt_tokens or prompt_tokens,
            completion_tokens=result.tokens_generated,
            generation_ms=result.generation_ms,
            total_ms=result.generation_ms,  # approximate
            tokens_per_second=result.tokens_per_second,
            prompt_eval_tps=0,
            finish_reason=result.finish_reason,
        )
        self.record_generation(metrics)

    # ── Reports ──────────────────────────────────────────────────────────

    def get_report(self, gpu_info=None) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        uptime = time.time() - self._start_time

        throughput_stats = self._throughput.stats()
        prompt_eval_stats = self._prompt_eval_tps.stats()

        report = PerformanceReport(
            uptime_seconds=uptime,
            total_generations=self._total_generations,
            total_tokens_generated=self._total_tokens,
            total_prompt_tokens=self._total_prompt_tokens,
            avg_tokens_per_second=throughput_stats.mean,
            avg_prompt_eval_tps=prompt_eval_stats.mean,
            latency_gen_ms=self._gen_latency.stats(),
            latency_total_ms=self._total_latency.stats(),
            throughput_tps=throughput_stats,
        )

        # GPU info if available
        if gpu_info:
            report.gpu_name = gpu_info.gpu_name
            report.vram_used_mb = gpu_info.used_vram_mb
            report.vram_total_mb = gpu_info.total_vram_mb
            report.vram_utilization_pct = (
                (gpu_info.used_vram_mb / gpu_info.total_vram_mb * 100)
                if gpu_info.total_vram_mb > 0 else 0
            )
            report.gpu_utilization_pct = gpu_info.gpu_utilization_pct
            report.gpu_temperature_c = gpu_info.gpu_temperature_c
            report.gpu_power_w = gpu_info.gpu_power_usage_w

            # VRAM efficiency
            if self._total_tokens > 0 and gpu_info.used_vram_mb > 0:
                report.vram_per_token_mb = gpu_info.used_vram_mb / max(self._total_tokens, 1)

        # Health assessment
        report.health_status, report.warnings = self._assess_health(report)

        return report

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary dict for API responses."""
        tps_stats = self._throughput.stats()
        lat_stats = self._gen_latency.stats()
        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_generations": self._total_generations,
            "total_tokens": self._total_tokens,
            "avg_tps": round(tps_stats.mean, 2),
            "p50_latency_ms": round(lat_stats.median, 1),
            "p99_latency_ms": round(lat_stats.p99, 1),
        }

    # ── Prometheus Metrics ───────────────────────────────────────────────

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible text metrics for /metrics endpoint."""
        tps_stats = self._throughput.stats()
        lat_stats = self._gen_latency.stats()
        uptime = time.time() - self._start_time

        lines = [
            f"# HELP llama_generations_total Total number of generation calls",
            f"# TYPE llama_generations_total counter",
            f"llama_generations_total {self._total_generations}",
            f"",
            f"# HELP llama_tokens_generated_total Total tokens generated",
            f"# TYPE llama_tokens_generated_total counter",
            f"llama_tokens_generated_total {self._total_tokens}",
            f"",
            f"# HELP llama_prompt_tokens_total Total prompt tokens processed",
            f"# TYPE llama_prompt_tokens_total counter",
            f"llama_prompt_tokens_total {self._total_prompt_tokens}",
            f"",
            f"# HELP llama_throughput_tokens_per_second Rolling average tokens/s",
            f"# TYPE llama_throughput_tokens_per_second gauge",
            f"llama_throughput_tokens_per_second {tps_stats.mean:.2f}",
            f"",
            f"# HELP llama_latency_generation_ms Generation latency stats",
            f"# TYPE llama_latency_generation_ms summary",
            f'llama_latency_generation_ms{{quantile="0.5"}} {lat_stats.median:.1f}',
            f'llama_latency_generation_ms{{quantile="0.9"}} {lat_stats.p90:.1f}',
            f'llama_latency_generation_ms{{quantile="0.95"}} {lat_stats.p95:.1f}',
            f'llama_latency_generation_ms{{quantile="0.99"}} {lat_stats.p99:.1f}',
            f"llama_latency_generation_ms_sum {lat_stats.sum_val:.1f}",
            f"llama_latency_generation_ms_count {lat_stats.count}",
            f"",
            f"# HELP llama_uptime_seconds Server uptime in seconds",
            f"# TYPE llama_uptime_seconds gauge",
            f"llama_uptime_seconds {uptime:.0f}",
        ]
        return "\n".join(lines)

    # ── Regression Detection ─────────────────────────────────────────────

    def set_baseline(self, tps: float):
        """Set a baseline throughput for regression detection."""
        self._baseline_tps = tps
        logger.info("Performance baseline set: %.1f tok/s", tps)

    def auto_baseline(self):
        """Set baseline from current rolling average."""
        stats = self._throughput.stats()
        if stats.count >= 10:
            self.set_baseline(stats.mean)

    def _check_regression(self, current_tps: float):
        """Check if current performance has regressed from baseline."""
        if self._baseline_tps is None or self._baseline_tps <= 0:
            return
        ratio = current_tps / self._baseline_tps
        if ratio < self._regression_threshold:
            for cb in self._alert_callbacks:
                try:
                    cb("regression", current_tps, self._baseline_tps, ratio)
                except Exception:
                    pass

    def register_alert_callback(self, callback: Callable):
        """Register a callback for performance alerts."""
        self._alert_callbacks.append(callback)

    # ── Background Reporting ─────────────────────────────────────────────

    def start_reporting(self, callback: Optional[Callable] = None):
        """Start periodic background reporting."""
        self._running = True
        self._report_callback = callback
        self._report_thread = threading.Thread(
            target=self._report_loop, daemon=True, name="perf-monitor",
        )
        self._report_thread.start()

    def stop_reporting(self):
        """Stop background reporting."""
        self._running = False
        if self._report_thread:
            self._report_thread.join(timeout=5)

    # ── Health ───────────────────────────────────────────────────────────

    def _assess_health(self, report: PerformanceReport) -> tuple:
        """Assess overall system health based on metrics."""
        status = HealthStatus.HEALTHY
        warnings = []

        # Check latency
        if report.latency_gen_ms.p99 > 5000:
            status = HealthStatus.DEGRADED
            warnings.append(f"High p99 latency: {report.latency_gen_ms.p99:.0f} ms")
        elif report.latency_gen_ms.p99 > 10000:
            status = HealthStatus.CRITICAL

        # Check throughput
        if report.avg_tokens_per_second > 0 and self._baseline_tps:
            ratio = report.avg_tokens_per_second / self._baseline_tps
            if ratio < 0.7:
                status = HealthStatus.CRITICAL
                warnings.append(f"Throughput regression: {ratio:.0%} of baseline")
            elif ratio < 0.85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append(f"Throughput below baseline: {ratio:.0%}")

        # Check GPU temperature
        if report.gpu_temperature_c > 85:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            warnings.append(f"High GPU temperature: {report.gpu_temperature_c:.0f} C")

        # Check VRAM
        if report.vram_utilization_pct > 95:
            status = HealthStatus.CRITICAL
            warnings.append(f"VRAM nearly full: {report.vram_utilization_pct:.0f}%")

        return status.value, warnings

    # ── Internal ─────────────────────────────────────────────────────────

    def _report_loop(self):
        """Background loop for periodic reporting."""
        while self._running:
            try:
                if self._report_callback:
                    report = self.get_report()
                    self._report_callback(report)
            except Exception as exc:
                logger.error("Report callback error: %s", exc)
            time.sleep(self._report_interval)


# ==============================================================================
# SECTION: engine.py
# ==============================================================================

"""
llama.cpp Engine Wrapper — Core inference engine abstraction.

Provides a clean interface over llama-cpp-python with:
  - Smart GPU layer management
  - KV cache type selection
  - Continuous batching support
  - Automatic context recycling
  - Thread-safe generation
  - Warm-up and benchmarking
"""


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class ModelInfo:
    """Metadata about a loaded model."""

    def __repr__(self) -> str:
        return (f"ModelInfo(path={self.model_path!r}, params={self.n_params_b:.1f}B, "
                f"layers={self.n_layers}, quant={self.quantization!r})")
    model_path: str = ""
    n_params_b: float = 0.0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    n_embd: int = 0
    vocab_size: int = 0
    context_length: int = 0
    quantization: str = "unknown"
    chat_template: str = ""


@dataclass(slots=True)
class GenerationResult:
    """Result from a single generation call."""

    def __repr__(self) -> str:
        return (f"GenerationResult(tokens={self.tokens_generated}, tps={self.tokens_per_second:.1f}, "
                f"reason={self.finish_reason!r})")
    text: str = ""
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    prompt_eval_ms: float = 0.0
    generation_ms: float = 0.0
    finish_reason: str = "stop"  # "stop", "length", "stop_sequence"
    slot_id: int = 0


@dataclass(slots=True)
class SlotState:
    """State of an inference slot for continuous batching."""
    id: int = 0
    busy: bool = False
    session_id: str = ""
    tokens_generated: int = 0
    n_ctx_used: int = 0
    started_at: float = 0.0


# ─── Engine ─────────────────────────────────────────────────────────────────

class LlamaEngine:
    """Core inference engine wrapping llama-cpp-python.

    This is the main workhorse that handles:
      - Model loading with smart GPU offloading
      - Single and streaming generation
      - Continuous batching via slots
      - Automatic warm-up and benchmarking
      - Thread-safe operations
    """

    def __init__(self, config: SmartGPUConfig, gpu_manager: SmartGPUManager):
        self._config = config
        self._gpu_config = config.gpu
        self._perf_config = config.performance
        self._inf_config = config.inference
        self._gpu_manager = gpu_manager

        self._model = None
        self._model_info = ModelInfo()
        self._loaded = False
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()

        # Slot management for continuous batching
        self._slots: List[SlotState] = [
            SlotState(id=i) for i in range(self._perf_config.n_slots)
        ]
        self._slot_lock = threading.Lock()

        # Telemetry
        self._total_generations = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._load_time_ms = 0.0
        self._warmup_done = False

        # Circuit breaker for generation operations
        self._generation_circuit = CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0, name="generation")

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def slots(self) -> List[SlotState]:
        return self._slots

    # ── Loading ──────────────────────────────────────────────────────────

    @timed(log_level="INFO")
    def load_model(self, model_path: str) -> ModelInfo:
        """Load a GGUF model with smart GPU layer offloading.

        The method:
          1. Probes the model metadata to determine layer count
          2. Asks SmartGPUManager for an offload plan
          3. Loads with optimal settings
          4. Warms up the model
        """
        if self._loaded:
            self.unload_model()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model: %s", model_path)
        load_start = time.perf_counter()

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with: "
                "pip install llama-cpp-python[server]  (or with CUBLAS=1 for GPU)"
            )

        # Determine GPU layers
        n_gpu_layers = self._gpu_config.n_gpu_layers
        if n_gpu_layers == -1:
            # Auto-detect: first probe the model to count layers, then compute plan
            n_gpu_layers = self._auto_detect_gpu_layers(model_path)

        # Determine thread count
        n_threads = self._perf_config.n_threads or os.cpu_count() or 4
        n_threads_batch = self._perf_config.n_threads_batch or n_threads

        # Build llama-cpp-python kwargs
        kwargs = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": self._perf_config.n_ctx,
            "n_batch": self._perf_config.n_batch,
            "n_threads": n_threads,
            "n_threads_batch": n_threads_batch,
            "use_mmap": self._gpu_config.use_mmap,
            "use_mlock": self._gpu_config.use_mlock,
            "rope_scaling_type": self._perf_config.rope_scaling_type or None,
            "rope_freq_base": self._perf_config.rope_freq_base or None,
            "verbose": False,
        }

        # KV cache types (llama-cpp-python >= 0.2.80)
        try:
            kwargs["cache_type_k"] = self._perf_config.cache_type_k
            kwargs["cache_type_v"] = self._perf_config.cache_type_v
        except TypeError:
            logger.debug("cache_type_k/v not supported by this llama-cpp-python version")

        # Tensor split for multi-GPU
        if self._gpu_config.tensor_split:
            kwargs["tensor_split"] = self._gpu_config.tensor_split

        logger.info(
            "Loading with: n_gpu_layers=%d, n_ctx=%d, n_batch=%d, n_threads=%d, "
            "cache_k=%s, cache_v=%s, mmap=%s",
            n_gpu_layers, self._perf_config.n_ctx, self._perf_config.n_batch,
            n_threads, self._perf_config.cache_type_k, self._perf_config.cache_type_v,
            self._gpu_config.use_mmap,
        )

        self._model = Llama(**kwargs)

        load_end = time.perf_counter()
        self._load_time_ms = (load_end - load_start) * 1000
        self._loaded = True

        # Extract model metadata
        self._extract_model_info(model_path, n_gpu_layers)

        logger.info(
            "Model loaded in %.1f ms | %d layers | %s quant | %.2B params",
            self._load_time_ms, self._model_info.n_layers,
            self._model_info.quantization, self._model_info.n_params_b,
        )

        # Warm up
        self._warmup()

        # Start GPU monitoring
        self._gpu_manager.start_monitoring()

        # Emit model loaded event
        try:
            get_event_bus().emit(EventType.MODEL_LOADED, {
                "model_path": model_path, "load_time_ms": self._load_time_ms,
                "n_layers": self._model_info.n_layers, "n_params_b": self._model_info.n_params_b,
            }, source="LlamaEngine")
        except Exception:
            pass

        return self._model_info

    def unload_model(self):
        """Unload the current model and free resources."""
        if self._model is not None:
            logger.info("Unloading model")
            del self._model
            self._model = None
        self._loaded = False
        self._warmup_done = False
        self._gpu_manager.stop_monitoring()

    # ── Generation ───────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from a prompt. Thread-safe.

        Kwargs override InferenceConfig defaults for this call.
        """
        if not self._loaded:
            raise ModelLoadError(model_path="", reason="No model currently loaded")

        params = self._build_params(kwargs)
        gen_start = time.perf_counter()

        with self._generation_circuit:
            with self._generation_lock:
                try:
                    result = self._model(
                        prompt,
                        max_tokens=params["n_predict"],
                        temperature=params["temperature"],
                        top_k=params["top_k"],
                        top_p=params["top_p"],
                        min_p=params.get("min_p", 0.05),
                        repeat_penalty=params["repeat_penalty"],
                        frequency_penalty=params.get("frequency_penalty", 0.0),
                        presence_penalty=params.get("presence_penalty", 0.0),
                        stop=params.get("stop", []),
                        seed=params.get("seed", -1) or -1,
                        echo=False,
                    )
                except LlamaSmartGPUError:
                    raise
                except Exception as exc:
                    logger.error("Generation failed: %s", exc)
                    raise

        gen_end = time.perf_counter()
        gen_ms = (gen_end - gen_start) * 1000

        # Parse result
        choice = result.get("choices", [{}])[0]
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "stop")
        usage = result.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / (gen_ms / 1000) if gen_ms > 0 else 0

        # Update telemetry
        self._total_generations += 1
        self._total_tokens += completion_tokens
        self._total_prompt_tokens += prompt_tokens

        return GenerationResult(
            text=text,
            tokens_generated=completion_tokens,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            prompt_eval_ms=0,  # llama-cpp-python doesn't expose this directly
            generation_ms=gen_ms,
            finish_reason=finish_reason,
        )

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generation tokens one at a time. Thread-safe."""
        if not self._loaded:
            raise ModelLoadError(model_path="", reason="No model currently loaded")

        params = self._build_params(kwargs)

        with self._generation_lock:
            try:
                for chunk in self._model(
                    prompt,
                    max_tokens=params["n_predict"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    min_p=params.get("min_p", 0.05),
                    repeat_penalty=params["repeat_penalty"],
                    frequency_penalty=params.get("frequency_penalty", 0.0),
                    presence_penalty=params.get("presence_penalty", 0.0),
                    stop=params.get("stop", []),
                    seed=params.get("seed", -1) or -1,
                    stream=True,
                    echo=False,
                ):
                    delta = chunk.get("choices", [{}])[0].get("text", "")
                    if delta:
                        self._total_tokens += 1
                        yield delta
            except Exception as exc:
                logger.error("Stream generation failed: %s", exc)
                raise

        self._total_generations += 1

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> GenerationResult:
        """Chat-style generation using the model's built-in chat template."""
        if not self._loaded:
            raise ModelLoadError(model_path="", reason="No model currently loaded")

        params = self._build_params(kwargs)
        gen_start = time.perf_counter()

        with self._generation_lock:
            try:
                result = self._model.create_chat_completion(
                    messages,
                    max_tokens=params["n_predict"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    frequency_penalty=params.get("frequency_penalty", 0.0),
                    presence_penalty=params.get("presence_penalty", 0.0),
                    stop=params.get("stop", []),
                    seed=params.get("seed", -1) or -1,
                )
            except Exception as exc:
                logger.error("Chat generation failed: %s", exc)
                raise

        gen_end = time.perf_counter()
        gen_ms = (gen_end - gen_start) * 1000

        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")
        usage = result.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / (gen_ms / 1000) if gen_ms > 0 else 0

        self._total_generations += 1
        self._total_tokens += completion_tokens
        self._total_prompt_tokens += prompt_tokens

        return GenerationResult(
            text=text,
            tokens_generated=completion_tokens,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            prompt_eval_ms=0,
            generation_ms=gen_ms,
            finish_reason=finish_reason,
        )

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[Dict]:
        """Stream chat completion chunks. Yields delta dicts."""
        if not self._loaded:
            raise ModelLoadError(model_path="", reason="No model currently loaded")

        params = self._build_params(kwargs)

        with self._generation_lock:
            try:
                for chunk in self._model.create_chat_completion(
                    messages,
                    max_tokens=params["n_predict"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    stop=params.get("stop", []),
                    seed=params.get("seed", -1) or -1,
                    stream=True,
                ):
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        self._total_tokens += 1
                    yield delta
            except Exception as exc:
                logger.error("Chat stream failed: %s", exc)
                raise

        self._total_generations += 1

    # ── Slot Management ──────────────────────────────────────────────────

    def acquire_slot(self, session_id: Optional[str] = None) -> Optional[SlotState]:
        """Acquire an available inference slot. Returns None if all busy."""
        sid = session_id or str(uuid.uuid4())
        with self._slot_lock:
            for slot in self._slots:
                if not slot.busy:
                    slot.busy = True
                    slot.session_id = sid
                    slot.started_at = time.time()
                    slot.tokens_generated = 0
                    slot.n_ctx_used = 0
                    return slot
        return None

    def release_slot(self, slot_id: int):
        """Release a slot back to the pool."""
        with self._slot_lock:
            for slot in self._slots:
                if slot.id == slot_id:
                    slot.busy = False
                    slot.session_id = ""
                    slot.tokens_generated = 0
                    slot.n_ctx_used = 0
                    break

    # ── Benchmarking ─────────────────────────────────────────────────────

    @timed(log_level="INFO")
    def benchmark(self, prompt_tokens: int = 128, gen_tokens: int = 128) -> Dict[str, Any]:
        """Run a quick benchmark and return performance metrics."""
        if not self._loaded:
            raise ModelLoadError(model_path="", reason="No model currently loaded")

        logger.info("Running benchmark (prompt=%d, gen=%d tokens)...", prompt_tokens, gen_tokens)

        # Generate a test prompt
        test_prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10 + 1)

        start = time.perf_counter()
        result = self.generate(test_prompt, n_predict=gen_tokens, temperature=0.0)
        total_ms = (time.perf_counter() - start) * 1000

        bench = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.tokens_generated,
            "total_time_ms": total_ms,
            "tokens_per_second": result.tokens_per_second,
            "generation_time_ms": result.generation_ms,
            "model": self._model_info.model_path,
            "gpu_layers": self._model_info.n_layers,  # placeholder
            "n_ctx": self._perf_config.n_ctx,
        }
        logger.info(
            "Benchmark: %.1f tok/s | prompt=%d | completion=%d | total=%.0f ms",
            result.tokens_per_second, result.prompt_tokens,
            result.tokens_generated, total_ms,
        )
        return bench

    # ── Telemetry ────────────────────────────────────────────────────────

    def get_telemetry(self) -> Dict[str, Any]:
        """Get engine telemetry data."""
        gpu_info = self._gpu_manager.get_memory_info()
        gpu_stats = self._gpu_manager.get_stats()

        return {
            "model_loaded": self._loaded,
            "model_path": self._model_info.model_path,
            "total_generations": self._total_generations,
            "total_tokens_generated": self._total_tokens,
            "total_prompt_tokens": self._total_prompt_tokens,
            "load_time_ms": self._load_time_ms,
            "warmup_done": self._warmup_done,
            "gpu_name": gpu_info.gpu_name,
            "vram_used_mb": gpu_info.used_vram_mb,
            "vram_total_mb": gpu_info.total_vram_mb,
            "vram_free_mb": gpu_info.free_vram_mb,
            "ram_used_mb": gpu_info.used_ram_mb,
            "ram_total_mb": gpu_info.total_ram_mb,
            "gpu_utilization_pct": gpu_info.gpu_utilization_pct,
            "gpu_temperature_c": gpu_info.gpu_temperature_c,
            "gpu_power_w": gpu_info.gpu_power_usage_w,
            "gpu_pressure": gpu_stats.get("current_pressure", "unknown"),
            "gpu_strategy": gpu_stats.get("strategy", "none"),
            "slots_total": len(self._slots),
            "slots_busy": sum(1 for s in self._slots if s.busy),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _auto_detect_gpu_layers(self, model_path: str) -> int:
        """Probe model metadata and compute optimal GPU layers."""
        try:
            from llama_cpp import Llama
            # Quick load with 0 GPU layers just to read metadata
            probe = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=1, verbose=False)
            metadata = probe.metadata()
            n_layers = int(metadata.get("llama.layer_count", 0))
            n_params = int(metadata.get("llama.parameter_count", 0))
            n_heads = int(metadata.get("llama.attention.head_count", 32))
            del probe

            if n_layers > 0 and n_params > 0:
                plan = self._gpu_manager.compute_offload_plan(
                    n_params / 1e9, n_layers,
                    quant="q4_k_m",  # approximate
                    n_ctx=self._perf_config.n_ctx,
                    n_heads=n_heads,
                )
                return plan.gpu_layers
        except Exception as exc:
            logger.warning("Auto-detect GPU layers failed, using fallback: %s", exc)

        # Fallback: try loading with all layers, let llama.cpp handle it
        return 99

    def _extract_model_info(self, model_path: str, n_gpu_layers: int):
        """Extract model metadata from loaded model."""
        self._model_info.model_path = model_path
        try:
            metadata = self._model.metadata()
            self._model_info.n_layers = int(metadata.get("llama.layer_count", 0))
            self._model_info.n_params_b = int(metadata.get("llama.parameter_count", 0)) / 1e9
            self._model_info.n_heads = int(metadata.get("llama.attention.head_count", 32))
            self._model_info.n_kv_heads = int(metadata.get("llama.attention.head_count_kv", self._model_info.n_heads))
            self._model_info.n_embd = int(metadata.get("llama.embedding_length", 0))
            self._model_info.vocab_size = int(metadata.get("llama.vocab_size", 0))
            self._model_info.context_length = int(metadata.get("llama.context_length", self._perf_config.n_ctx))
            self._model_info.quantization = metadata.get("general.quantization_version", "unknown")
            self._model_info.chat_template = metadata.get("tokenizer.chat_template", "")
        except Exception as exc:
            logger.warning("Could not extract full model metadata: %s", exc)

    def _build_params(self, overrides: Dict) -> Dict:
        """Merge inference defaults with per-request overrides."""
        params = {
            "n_predict": self._inf_config.n_predict,
            "temperature": self._inf_config.temperature,
            "top_k": self._inf_config.top_k,
            "top_p": self._inf_config.top_p,
            "min_p": self._inf_config.min_p,
            "repeat_penalty": self._inf_config.repeat_penalty,
            "frequency_penalty": self._inf_config.frequency_penalty,
            "presence_penalty": self._inf_config.presence_penalty,
            "stop": self._inf_config.stop,
            "seed": self._inf_config.seed,
            "stream": self._inf_config.stream,
        }
        params.update(overrides)
        return params

    def _warmup(self):
        """Run a short warm-up generation to prime CUDA kernels and caches."""
        if self._warmup_done:
            return
        logger.info("Warming up model...")
        try:
            self._model("Hello", max_tokens=8, temperature=0.0, verbose=False)
            self._warmup_done = True
            logger.info("Warm-up complete")
        except Exception as exc:
            logger.warning("Warm-up failed (non-fatal): %s", exc)
            self._warmup_done = True  # Don't retry on every call


# ==============================================================================
# SECTION: model_manager.py
# ==============================================================================

"""
Model Manager — Professional model lifecycle, registry, and deployment system.

Features:
  - Model Registry with versioning, tags, aliases, and rich metadata
  - Automatic model discovery from configured directories
  - Model metadata extraction and persistent caching
  - Hot-swap model loading with pre-flight checks
  - VRAM-aware model selection and recommendations
  - HuggingFace GGUF download with progress, resume, and queue
  - Full file integrity verification (MD5 quick + SHA256 deep)
  - Model health validation (probe load, smoke test)
  - Model comparison and ranking
  - Quantization format analysis and VRAM projections
  - Model aliasing and tagging for easy reference
  - Download queue with priority and concurrency control
  - Model search and filter with multi-criteria queries
  - Auto-config generation per model (optimal n_ctx, n_batch, cache types)
  - Model lifecycle events and hooks
  - Disk space management with cleanup of stale downloads
"""


# ─── Enums ──────────────────────────────────────────────────────────────────

class ModelStatus(Enum):
    """Model lifecycle status."""
    DISCOVERED = "discovered"       # Found on disk, not yet validated
    VALIDATED = "validated"         # Metadata extracted, integrity checked
    READY = "ready"                 # Ready to load
    LOADED = "loaded"               # Currently loaded in engine
    CORRUPT = "corrupt"             # Failed integrity check
    DOWNLOADING = "downloading"     # Download in progress
    DOWNLOAD_PAUSED = "paused"      # Download paused
    ERROR = "error"                 # Unrecoverable error

    @property
    def is_terminal(self) -> bool:
        return self in (ModelStatus.LOADED, ModelStatus.ERROR, ModelStatus.CORRUPT)

    def __str__(self) -> str:
        return f"ModelStatus.{self.name}"


class DownloadStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelFamily(Enum):
    """Known model families for auto-classification."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    DEEPSEEK = "deepseek"
    COMMAND_R = "command-r"
    FALCON = "falcon"
    STABLELM = "stablelm"
    UNKNOWN = "unknown"


# ─── Rich Data Classes ──────────────────────────────────────────────────────

@dataclass
class QuantizationInfo:
    """Detailed quantization analysis."""
    format_name: str = "unknown"          # e.g., "Q4_K_M"
    bits_per_weight: float = 0.0          # Effective bits per weight
    compression_ratio: float = 0.0        # vs FP16
    vram_per_billion_params_mb: float = 0.0
    quality_tier: str = "unknown"         # "highest", "high", "medium", "low"
    recommended_use: str = "general"      # "chat", "code", "reasoning", "general"
    perplexity_delta_vs_f16: Optional[float] = None


@dataclass
class ModelRecommendation:
    """Recommendation for a specific use case."""
    model_name: str = ""
    reason: str = ""
    confidence: float = 0.0
    suggested_n_ctx: int = 4096
    suggested_n_batch: int = 512
    suggested_cache_k: str = "f16"
    suggested_cache_v: str = "f16"
    suggested_n_gpu_layers: int = -1
    estimated_vram_mb: float = 0.0
    estimated_tps_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class DownloadTask:
    """Represents a model download in progress or completed."""
    id: str = ""
    url: str = ""
    destination: str = ""
    filename: str = ""
    total_bytes: int = 0
    downloaded_bytes: int = 0
    status: DownloadStatus = DownloadStatus.PENDING
    priority: int = 2                    # Lower = higher priority
    progress_pct: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Resume support
    supports_resume: bool = False
    # Integrity
    expected_sha256: str = ""
    verified: bool = False


@dataclass
class ModelEntry:
    """Comprehensive model entry with rich metadata and lifecycle tracking."""
    # Identity
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    family: str = "unknown"
    version: str = "1.0.0"

    # File info
    path: str = ""
    size_mb: float = 0.0
    size_gb: float = 0.0
    file_hash_md5: str = ""          # Quick hash (first 1MB)
    file_hash_sha256: str = ""       # Full file hash (computed on demand)
    mtime: float = 0.0
    created_at: float = 0.0

    # Architecture
    n_params_b: float = 0.0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    n_embd: int = 0
    vocab_size: int = 0
    context_length: int = 0
    rope_freq_base: float = 0.0
    architecture: str = "llama"

    # Quantization
    quantization: str = "unknown"
    quant_info: Dict[str, Any] = field(default_factory=dict)
    bits_per_weight: float = 0.0

    # Templates and formatting
    chat_template: str = ""
    bos_token: str = ""
    eos_token: str = ""
    special_tokens: Dict[str, str] = field(default_factory=dict)

    # VRAM estimates
    estimated_vram_full_mb: float = 0.0
    estimated_vram_min_mb: float = 0.0     # CPU-only RAM (0 GPU layers)
    estimated_vram_optimal_mb: float = 0.0  # Best performance/VRAM tradeoff
    fits_in_vram: bool = False
    vram_fit_score: float = 0.0            # 0-1, how well it fits
    recommended_gpu_layers: int = -1

    # Lifecycle
    status: str = "discovered"
    loaded: bool = False
    last_loaded_at: float = 0.0
    load_count: int = 0
    last_validated_at: float = 0.0
    validation_errors: List[str] = field(default_factory=list)

    # Source / provenance
    source_url: str = ""
    source_repo: str = ""              # e.g., "TheBloke/Llama-2-7B-GGUF"
    license: str = ""
    description: str = ""

    # Performance tracking
    avg_tps: float = 0.0
    best_tps: float = 0.0
    total_tokens_generated: int = 0
    total_generations: int = 0

    # Auto-config
    auto_config: Dict[str, Any] = field(default_factory=dict)


# ─── Quantization Knowledge Base ────────────────────────────────────────────

QUANT_TABLE: Dict[str, QuantizationInfo] = {
    "f16":       QuantizationInfo("F16",       16.0, 1.0,  32768, "highest", "reasoning"),
    "q8_0":      QuantizationInfo("Q8_0",       8.5, 0.53, 17326, "high",    "chat"),
    "q6_k":      QuantizationInfo("Q6_K",       6.59,0.41, 13456, "high",    "chat"),
    "q5_k_m":    QuantizationInfo("Q5_K_M",     5.69,0.36, 11614, "high",    "chat"),
    "q5_k_s":    QuantizationInfo("Q5_K_S",     5.54,0.35, 11290, "medium",  "general"),
    "q5_1":      QuantizationInfo("Q5_1",       5.75,0.36, 11776, "medium",  "general"),
    "q5_0":      QuantizationInfo("Q5_0",       5.54,0.35, 11290, "medium",  "general"),
    "q4_k_m":    QuantizationInfo("Q4_K_M",     4.80,0.30,  9728, "medium",  "general"),
    "q4_k_s":    QuantizationInfo("Q4_K_S",     4.59,0.29,  9312, "medium",  "general"),
    "q4_1":      QuantizationInfo("Q4_1",       5.00,0.31, 10240, "low",     "general"),
    "q4_0":      QuantizationInfo("Q4_0",       4.50,0.28,  9216, "low",     "code"),
    "q3_k_m":    QuantizationInfo("Q3_K_M",     3.78,0.24,  7680, "low",     "code"),
    "q3_k_s":    QuantizationInfo("Q3_K_S",     3.40,0.21,  6912, "low",     "code"),
    "q2_k":      QuantizationInfo("Q2_K",       2.63,0.16,  5376, "low",     "code"),
    "iq3_m":     QuantizationInfo("IQ3_M",      3.45,0.22,  7040, "low",     "code"),
    "iq3_s":     QuantizationInfo("IQ3_S",      3.20,0.20,  6528, "low",     "code"),
}

# Family detection patterns (match against model filename)
FAMILY_PATTERNS: List[Tuple[ModelFamily, re.Pattern]] = [
    (ModelFamily.LLAMA,     re.compile(r"(llama|llama-?2|llama-?3|vicuna|nous-hermes|openhermes|wizardlm|orca)", re.I)),
    (ModelFamily.MISTRAL,   re.compile(r"(mistral|zephyr|open-mistral)", re.I)),
    (ModelFamily.MIXTRAL,   re.compile(r"(mixtral)", re.I)),
    (ModelFamily.QWEN,      re.compile(r"(qwen|qwen2)", re.I)),
    (ModelFamily.GEMMA,     re.compile(r"(gemma)", re.I)),
    (ModelFamily.PHI,       re.compile(r"(phi-?2|phi-?3|phicode)", re.I)),
    (ModelFamily.DEEPSEEK,  re.compile(r"(deepseek|deepseek-?coder)", re.I)),
    (ModelFamily.COMMAND_R, re.compile(r"(command-?r|c4ai-?command)", re.I)),
    (ModelFamily.FALCON,    re.compile(r"(falcon)", re.I)),
    (ModelFamily.STABLELM,  re.compile(r"(stablelm|stable-?code)", re.I)),
]

# Quantization detection from filename
QUANT_PATTERN = re.compile(
    r"\.(q[0-9]_[kmso]?|iq[0-9]_[xs]?|f16|f32|bf16)", re.I
)


# ─── Model Classifier ───────────────────────────────────────────────────────

class ModelClassifier:
    """Classify models by family, quantization, and capabilities from filenames and metadata."""

    @staticmethod
    def detect_family(name: str) -> str:
        """Detect model family from name/path string."""
        lower = name.lower()
        for family, pattern in FAMILY_PATTERNS:
            if pattern.search(lower):
                return family.value
        return ModelFamily.UNKNOWN.value

    @staticmethod
    def detect_quantization(name: str) -> str:
        """Detect quantization format from filename."""
        match = QUANT_PATTERN.search(name.lower())
        if match:
            return match.group(1).upper()
        return "unknown"

    @staticmethod
    def detect_params_from_name(name: str) -> float:
        """Estimate parameter count from model name (e.g., '7b' → 7.0)."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", name)
        if match:
            return float(match.group(1))
        return 0.0

    @staticmethod
    def get_quant_info(quant_name: str) -> QuantizationInfo:
        """Look up quantization details from the knowledge base."""
        key = quant_name.lower().replace("-", "_").replace(".", "_")
        if key in QUANT_TABLE:
            return QUANT_TABLE[key]
        # Try partial match
        for k, v in QUANT_TABLE.items():
            if k in key or key in k:
                return v
        return QuantizationInfo(quant_name, 4.8, 0.30, 9728, "medium", "general")

    @staticmethod
    def suggest_auto_config(entry: "ModelEntry", available_vram_mb: float,
                            n_ctx: int = 4096) -> Dict[str, Any]:
        """Generate optimal configuration suggestions for a model.

        Considers VRAM budget, model size, quantization, and known best practices
        for the model family to produce a configuration that balances performance
        and memory usage.
        """
        config = {
            "n_ctx": n_ctx,
            "n_batch": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
        }

        # Adjust n_batch based on VRAM budget
        if available_vram_mb < 4096:
            config["n_batch"] = 256
        elif available_vram_mb > 24576:
            config["n_batch"] = 1024

        # KV cache quantization for tight VRAM
        model_vram = entry.estimated_vram_full_mb
        if model_vram > 0:
            vram_ratio = model_vram / max(available_vram_mb, 1)
            if vram_ratio > 0.85:
                config["cache_type_k"] = "q8_0"
                config["cache_type_v"] = "q8_0"
            elif vram_ratio > 0.70:
                config["cache_type_k"] = "q8_0"
                config["cache_type_v"] = "f16"

        # GPU layers recommendation
        if available_vram_mb > 0 and entry.n_layers > 0:
            plan = VRAMEstimator.compute_optimal_gpu_layers(
                entry.n_params_b, entry.n_layers,
                available_vram_mb, entry.quantization,
                n_ctx=n_ctx, n_heads=entry.n_heads,
            )
            config["n_gpu_layers"] = plan.gpu_layers

        # Family-specific tuning
        family = entry.family
        if family in ("mistral", "mixtral"):
            config["n_batch"] = min(config["n_batch"], 512)
        elif family == "qwen":
            config["n_batch"] = min(config["n_batch"], 512)

        # Context length recommendations based on model capability
        if entry.context_length > 0:
            config["max_supported_ctx"] = entry.context_length
            # Don't suggest more context than the model supports
            if n_ctx > entry.context_length:
                config["n_ctx"] = entry.context_length

        return config


# ─── Model Validator ────────────────────────────────────────────────────────

class ModelValidator:
    """Validate model files for integrity and loadability.

    Provides three levels of validation:
      - Quick: MD5 of first 1MB (fast, detects obviously wrong files)
      - Standard: Full SHA256 (thorough, detects corruption anywhere)
      - Deep: Quick load test with n_ctx=1 (verifies GGUF structure)
    """

    @staticmethod
    def quick_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
        """MD5 of first chunk_size bytes — fast integrity check."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(chunk_size))
        return h.hexdigest()

    @staticmethod
    def full_hash(path: Path) -> str:
        """SHA256 of entire file — thorough integrity check."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)  # 8MB chunks
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def validate_quick(entry: ModelEntry) -> Tuple[bool, List[str]]:
        """Quick validation: check file exists and quick hash matches."""
        errors = []
        path = Path(entry.path)
        if not path.exists():
            errors.append(f"File not found: {entry.path}")
            return False, errors
        if path.stat().st_size < 1024:
            errors.append("File too small to be a valid GGUF model")
            return False, errors
        current_hash = ModelValidator.quick_hash(path)
        if entry.file_hash_md5 and current_hash != entry.file_hash_md5:
            errors.append(f"Quick hash mismatch: expected {entry.file_hash_md5}, got {current_hash}")
            return False, errors
        return True, errors

    @staticmethod
    def validate_deep(path: str) -> Tuple[bool, List[str]]:
        """Deep validation: attempt to load model with n_ctx=1."""
        errors = []
        try:
            from llama_cpp import Llama
            probe = Llama(model_path=path, n_gpu_layers=0, n_ctx=1, verbose=False)
            # Try a tiny generation to verify the model works
            _ = probe("Hi", max_tokens=1, temperature=0.0)
            del probe
            return True, errors
        except ImportError:
            errors.append("llama-cpp-python not available for deep validation")
            return False, errors
        except Exception as exc:
            errors.append(f"Deep validation failed: {exc}")
            return False, errors


# ─── Model Discovery ────────────────────────────────────────────────────────

class ModelDiscovery:
    """Scan directories for GGUF model files and extract rich metadata.

    Features:
      - Recursive directory scanning
      - Persistent metadata cache with staleness detection
      - Automatic family and quantization classification
      - VRAM estimation for all discovered models
    """

    GGUF_EXTENSIONS = {".gguf"}

    def __init__(self, model_dirs: List[str], cache_file: Optional[str] = None):
        self._model_dirs = [Path(d) for d in model_dirs]
        self._cache_file = Path(cache_file) if cache_file else None
        self._cache: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()

        if self._cache_file and self._cache_file.exists():
            self._load_cache()

    def scan(self, force: bool = False) -> List[ModelEntry]:
        """Scan all model directories and return discovered models.

        Args:
            force: If True, re-probe all models even if cached.
        """
        found: Dict[str, ModelEntry] = {}

        for model_dir in self._model_dirs:
            if not model_dir.exists():
                logger.debug("Model directory does not exist: %s", model_dir)
                continue

            for path in model_dir.rglob("*"):
                if path.suffix.lower() in self.GGUF_EXTENSIONS:
                    entry = self._process_model_file(path, force=force)
                    if entry:
                        found[entry.name] = entry

        with self._lock:
            # Update cache with found models
            self._cache.update(found)
            # Remove entries whose files no longer exist
            stale = [
                name for name, entry in self._cache.items()
                if name not in found and not Path(entry.path).exists()
            ]
            for name in stale:
                del self._cache[name]

        self._save_cache()
        return list(self._cache.values())

    def get_models(self) -> List[ModelEntry]:
        """Return cached model list, scanning if empty."""
        with self._lock:
            if self._cache:
                return list(self._cache.values())
        return self.scan()

    def get_model(self, name: str) -> Optional[ModelEntry]:
        """Get a model by name or alias."""
        with self._lock:
            # Direct name match
            if name in self._cache:
                return self._cache[name]
            # Alias match
            for entry in self._cache.values():
                if name in entry.aliases:
                    return entry
        return None

    def find_by_path(self, path: str) -> Optional[ModelEntry]:
        """Find a model entry by its file path."""
        abs_path = str(Path(path).resolve())
        with self._lock:
            for entry in self._cache.values():
                if str(Path(entry.path).resolve()) == abs_path:
                    return entry
        return None

    def search(self, query: str, tags: Optional[List[str]] = None,
               family: Optional[str] = None, min_params: float = 0,
               max_params: float = float("inf"),
               fits_vram_only: bool = False,
               sort_by: str = "name") -> List[ModelEntry]:
        """Search and filter models with multi-criteria support.

        Args:
            query: Text search across name, description, family
            tags: Filter by tags (AND logic — must have all)
            family: Filter by model family
            min_params: Minimum parameter count in billions
            max_params: Maximum parameter count in billions
            fits_vram_only: Only return models that fit in VRAM
            sort_by: "name", "params", "size", "vram", "quality"
        """
        with self._lock:
            entries = list(self._cache.values())

        results = []
        for entry in entries:
            # Text query
            if query:
                q = query.lower()
                searchable = f"{entry.name} {entry.description} {entry.family} {' '.join(entry.tags)}".lower()
                if q not in searchable:
                    continue

            # Tags filter
            if tags:
                entry_tags = set(entry.tags)
                if not all(t in entry_tags for t in tags):
                    continue

            # Family filter
            if family and entry.family != family.lower():
                continue

            # Parameter range
            if entry.n_params_b < min_params or entry.n_params_b > max_params:
                continue

            # VRAM fit
            if fits_vram_only and not entry.fits_in_vram:
                continue

            results.append(entry)

        # Sort
        sort_key_map = {
            "name": lambda e: e.name,
            "params": lambda e: e.n_params_b,
            "size": lambda e: e.size_mb,
            "vram": lambda e: e.estimated_vram_full_mb,
            "quality": lambda e: QUANT_TABLE.get(e.quantization.lower(), QuantizationInfo()).bits_per_weight,
        }
        key_fn = sort_key_map.get(sort_by, lambda e: e.name)
        reverse = sort_by in ("params", "size", "vram", "quality")
        results.sort(key=key_fn, reverse=reverse)

        return results

    def _process_model_file(self, path: Path, force: bool = False) -> Optional[ModelEntry]:
        """Extract metadata from a GGUF file and classify it."""
        stat = path.stat()
        name = path.stem
        size_mb = stat.st_size / (1024 ** 2)
        size_gb = size_mb / 1024

        # Check cache (unless forced)
        if not force:
            with self._lock:
                cached = self._cache.get(name)
                if cached and cached.mtime == stat.st_mtime and cached.path == str(path):
                    return cached

        entry = ModelEntry(
            name=name,
            path=str(path),
            size_mb=size_mb,
            size_gb=size_gb,
            mtime=stat.st_mtime,
            created_at=stat.st_ctime,
            status=ModelStatus.DISCOVERED.value,
        )

        # Classify by family and quantization from filename
        entry.family = ModelClassifier.detect_family(name)
        entry.quantization = ModelClassifier.detect_quantization(name)

        # Estimate params from name if not yet probed
        entry.n_params_b = ModelClassifier.detect_params_from_name(name)

        # Quick hash for integrity
        try:
            entry.file_hash_md5 = ModelValidator.quick_hash(path)
        except Exception:
            pass

        # Extract GGUF metadata (slower, may fail)
        try:
            metadata = self._extract_gguf_metadata(path)
            if metadata:
                entry.n_params_b = metadata.get("n_params_b", entry.n_params_b)
                entry.n_layers = metadata.get("n_layers", 0)
                entry.n_heads = metadata.get("n_heads", 32)
                entry.n_kv_heads = metadata.get("n_kv_heads", entry.n_heads)
                entry.n_embd = metadata.get("n_embd", 0)
                entry.vocab_size = metadata.get("vocab_size", 0)
                entry.context_length = metadata.get("context_length", 4096)
                entry.rope_freq_base = metadata.get("rope_freq_base", 0)
                entry.architecture = metadata.get("architecture", "llama")
                entry.chat_template = metadata.get("chat_template", "")
                entry.bos_token = metadata.get("bos_token", "")
                entry.eos_token = metadata.get("eos_token", "")
                entry.special_tokens = metadata.get("special_tokens", {})
                # Use probed quant if more specific
                if metadata.get("quantization") and metadata["quantization"] != "unknown":
                    entry.quantization = metadata["quantization"]
                entry.status = ModelStatus.VALIDATED.value
                entry.last_validated_at = time.time()
        except Exception as exc:
            logger.debug("Could not extract metadata from %s: %s", path, exc)

        # Enrich quantization info
        qi = ModelClassifier.get_quant_info(entry.quantization)
        entry.bits_per_weight = qi.bits_per_weight
        entry.quant_info = {
            "format": qi.format_name,
            "bits_per_weight": qi.bits_per_weight,
            "compression_ratio": qi.compression_ratio,
            "quality_tier": qi.quality_tier,
            "recommended_use": qi.recommended_use,
        }

        # Auto-generate tags based on what we know
        entry.tags = self._auto_tag(entry)

        return entry

    def _extract_gguf_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract metadata from a GGUF file using llama-cpp-python probe."""
        try:
            from llama_cpp import Llama
            probe = Llama(model_path=str(path), n_gpu_layers=0, n_ctx=1, verbose=False)
            raw = probe.metadata()
            del probe

            return {
                "n_params_b": int(raw.get("llama.parameter_count", 0)) / 1e9,
                "n_layers": int(raw.get("llama.layer_count", 0)),
                "n_heads": int(raw.get("llama.attention.head_count", 32)),
                "n_kv_heads": int(raw.get("llama.attention.head_count_kv", 32)),
                "n_embd": int(raw.get("llama.embedding_length", 0)),
                "vocab_size": int(raw.get("llama.vocab_size", 0)),
                "context_length": int(raw.get("llama.context_length", 4096)),
                "rope_freq_base": float(raw.get("llama.rope.freq_base", 0)),
                "architecture": raw.get("general.architecture", "llama"),
                "quantization": raw.get("general.quantization_version", "unknown"),
                "chat_template": raw.get("tokenizer.chat_template", ""),
                "bos_token": raw.get("tokenizer.ggml.tokens", [""])[0] if raw.get("tokenizer.ggml.tokens") else "",
                "eos_token": raw.get("tokenizer.ggml.eos_token_id", ""),
                "special_tokens": {},
            }
        except Exception as exc:
            logger.debug("GGUF probe failed for %s: %s", path, exc)
            # Size-based estimation fallback
            size_bytes = path.stat().st_size
            est_params = size_bytes / 0.63
            return {
                "n_params_b": est_params / 1e9,
                "n_layers": int(est_params / 1e9 * 32 / 7),
                "quantization": "q4_k_m (estimated)",
            }

    @staticmethod
    def _auto_tag(entry: ModelEntry) -> List[str]:
        """Auto-generate tags for a model based on its properties."""
        tags = set()
        if entry.family != "unknown":
            tags.add(entry.family)
        if entry.n_params_b > 0:
            if entry.n_params_b <= 3:
                tags.add("small")
            elif entry.n_params_b <= 8:
                tags.add("medium")
            elif entry.n_params_b <= 14:
                tags.add("large")
            elif entry.n_params_b <= 70:
                tags.add("xl")
            else:
                tags.add("xxl")
        qi = ModelClassifier.get_quant_info(entry.quantization)
        if qi.quality_tier != "unknown":
            tags.add(f"quality-{qi.quality_tier}")
        if qi.recommended_use != "unknown":
            tags.add(qi.recommended_use)
        if entry.context_length >= 32768:
            tags.add("long-context")
        if entry.context_length >= 128000:
            tags.add("ultra-context")
        # Check for instruct/chat variants
        name_lower = entry.name.lower()
        if any(k in name_lower for k in ("instruct", "chat", "rlhf", "hermes")):
            tags.add("instruct")
        if any(k in name_lower for k in ("code", "coder", "coding")):
            tags.add("code")
        if any(k in name_lower for k in ("base", "pretrain")):
            tags.add("base")
        return sorted(tags)


    # ── Cache Persistence ────────────────────────────────────────────────

    def _load_cache(self):
        """Load model metadata cache from disk."""
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)
            for name, entry_data in data.items():
                try:
                    self._cache[name] = ModelEntry(**entry_data)
                except TypeError:
                    # Handle schema evolution — skip entries with unknown fields
                    logger.debug("Skipping cache entry with incompatible schema: %s", name)
            logger.debug("Loaded %d cached model entries", len(self._cache))
        except Exception as exc:
            logger.warning("Failed to load model cache: %s", exc)

    def _save_cache(self):
        """Save model metadata cache to disk."""
        if not self._cache_file:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            with self._lock:
                for name, entry in self._cache.items():
                    data[name] = asdict(entry)
            with open(self._cache_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to save model cache: %s", exc)


# ─── Download Manager ───────────────────────────────────────────────────────

class DownloadManager:
    """Manage model downloads with progress, resume, queue, and concurrency.

    Features:
      - Priority-based download queue
      - Progress tracking with speed and ETA
      - Resume partial downloads (if server supports Range headers)
      - SHA256 verification after download
      - Concurrent download limit
      - Callback system for progress updates
    """

    def __init__(self, model_dir: str, max_concurrent: int = 2):
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._max_concurrent = max_concurrent
        self._queue: PriorityQueue = PriorityQueue()
        self._active: Dict[str, DownloadTask] = {}
        self._completed: Dict[str, DownloadTask] = {}
        self._lock = threading.Lock()
        self._running = False
        self._workers: List[threading.Thread] = []
        self._callbacks: List[Callable] = []
        self._cancel_flags: Dict[str, threading.Event] = {}

    def start(self):
        """Start the download worker pool."""
        self._running = True
        for i in range(self._max_concurrent):
            t = threading.Thread(target=self._worker_loop, daemon=True,
                                 name=f"download-worker-{i}")
            t.start()
            self._workers.append(t)
        logger.info("Download manager started with %d workers", self._max_concurrent)

    def stop(self):
        """Stop all download workers."""
        self._running = False
        for _ in self._workers:
            self._queue.put((99, time.time(), None))  # Sentinel
        for t in self._workers:
            t.join(timeout=5)
        self._workers.clear()

    def submit(self, url: str, filename: Optional[str] = None,
               priority: int = 2, expected_sha256: str = "") -> DownloadTask:
        """Submit a download task. Returns a DownloadTask for tracking."""
        task_id = hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:12]
        fname = filename or url.split("/")[-1]
        dest = self._model_dir / fname

        task = DownloadTask(
            id=task_id,
            url=url,
            destination=str(dest),
            filename=fname,
            priority=priority,
            expected_sha256=expected_sha256,
        )

        with self._lock:
            self._active[task_id] = task

        self._queue.put((priority, task.created_at, task))
        logger.info("Download queued: %s → %s (priority=%d)", url, fname, priority)
        return task

    def cancel(self, task_id: str) -> bool:
        """Cancel a download task."""
        flag = self._cancel_flags.get(task_id)
        if flag:
            flag.set()
            with self._lock:
                task = self._active.get(task_id)
                if task:
                    task.status = DownloadStatus.CANCELLED
            return True
        return False

    def pause(self, task_id: str) -> bool:
        """Pause a download (sets cancel flag; resume will re-queue)."""
        return self.cancel(task_id)

    def get_task(self, task_id: str) -> Optional[DownloadTask]:
        """Get download task status."""
        with self._lock:
            if task_id in self._active:
                return self._active[task_id]
            if task_id in self._completed:
                return self._completed[task_id]
        return None

    def list_active(self) -> List[Dict[str, Any]]:
        """List active downloads."""
        with self._lock:
            return [asdict(t) for t in self._active.values() if t.status == DownloadStatus.ACTIVE]

    def list_completed(self) -> List[Dict[str, Any]]:
        """List completed downloads."""
        with self._lock:
            return [asdict(t) for t in self._completed.values()]

    def register_callback(self, callback: Callable):
        """Register a callback(task, event_type) for download events."""
        self._callbacks.append(callback)

    # ── Internal ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        while self._running:
            try:
                priority, _, task = self._queue.get(timeout=2.0)
                if task is None:
                    break
                self._execute_download(task)
            except Exception:
                continue

    def _execute_download(self, task: DownloadTask):
        """Execute a single download with resume support and progress tracking."""
        cancel = threading.Event()
        self._cancel_flags[task.id] = cancel

        task.status = DownloadStatus.ACTIVE
        task.started_at = time.time()
        dest = Path(task.destination)

        # Check for partial download (resume)
        resume_from = 0
        if dest.exists():
            resume_from = dest.stat().st_size
            # Check if server supports Range
            try:
                req = urllib.request.Request(task.url, method="HEAD")
                resp = urllib.request.urlopen(req, timeout=10)
                accept_ranges = resp.headers.get("Accept-Ranges", "")
                content_length = int(resp.headers.get("Content-Length", 0))
                if "bytes" in accept_ranges and resume_from < content_length:
                    task.downloaded_bytes = resume_from
                    task.total_bytes = content_length
                    task.supports_resume = True
                    logger.info("Resuming download from %d bytes", resume_from)
                else:
                    resume_from = 0
            except Exception:
                resume_from = 0

        try:
            headers = {}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

            req = urllib.request.Request(task.url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)

            # Get total size
            content_length = int(resp.headers.get("Content-Length", 0))
            if resume_from > 0 and content_length > 0:
                task.total_bytes = resume_from + content_length
            else:
                task.total_bytes = content_length

            # Download with progress
            mode = "ab" if resume_from > 0 else "wb"
            last_progress_time = time.time()
            last_bytes = task.downloaded_bytes

            with open(dest, mode) as f:
                while True:
                    if cancel.is_set():
                        task.status = DownloadStatus.CANCELLED
                        self._fire_callback(task, "cancelled")
                        return

                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break

                    f.write(chunk)
                    task.downloaded_bytes += len(chunk)

                    # Update progress
                    now = time.time()
                    elapsed = now - last_progress_time
                    if elapsed >= 1.0:  # Update every second
                        bytes_delta = task.downloaded_bytes - last_bytes
                        task.speed_mbps = (bytes_delta / elapsed) / (1024 * 1024)
                        if task.speed_mbps > 0 and task.total_bytes > 0:
                            remaining = (task.total_bytes - task.downloaded_bytes)
                            task.eta_seconds = remaining / (task.speed_mbps * 1024 * 1024)
                        task.progress_pct = (task.downloaded_bytes / max(task.total_bytes, 1)) * 100
                        last_progress_time = now
                        last_bytes = task.downloaded_bytes
                        self._fire_callback(task, "progress")

            # Download complete
            task.progress_pct = 100.0
            task.completed_at = time.time()
            task.speed_mbps = 0.0
            task.eta_seconds = 0.0

            # Verify SHA256 if provided
            if task.expected_sha256:
                task.verified = self._verify_file(dest, task.expected_sha256)
                if not task.verified:
                    task.status = DownloadStatus.FAILED
                    task.error = "SHA256 verification failed"
                    self._fire_callback(task, "verify_failed")
                    return

            task.status = DownloadStatus.COMPLETED
            with self._lock:
                self._completed[task.id] = task
                self._active.pop(task.id, None)
            self._fire_callback(task, "completed")
            logger.info("Download complete: %s (%.1f MB)", task.filename, dest.stat().st_size / (1024**2))

        except Exception as exc:
            task.status = DownloadStatus.FAILED
            task.error = str(exc)
            self._fire_callback(task, "failed")
            logger.error("Download failed: %s — %s", task.filename, exc)

    @staticmethod
    def _verify_file(path: Path, expected_sha256: str) -> bool:
        """Verify file SHA256 hash."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest() == expected_sha256

    def _fire_callback(self, task: DownloadTask, event: str):
        for cb in self._callbacks:
            try:
                cb(task, event)
            except Exception as exc:
                logger.error("Download callback error: %s", exc)


# ─── Model Manager ──────────────────────────────────────────────────────────

class ModelManager:
    """Professional model lifecycle manager.

    Orchestrates all model operations:
      - Discovery and registration
      - Validation and integrity checking
      - VRAM-aware selection and recommendations
      - Loading, unloading, and hot-swapping
      - Download management with queue
      - Aliasing, tagging, and search
      - Model comparison and ranking
      - Auto-config generation
      - Lifecycle events and hooks
    """

    def __init__(self, config: SmartGPUConfig, gpu_manager: SmartGPUManager):
        self._config = config
        self._gpu_manager = gpu_manager
        self._engine = None

        # Build model search directories
        model_dirs = [config.model_dir]
        for p in config.model_paths:
            parent = str(Path(p).parent)
            if parent not in model_dirs:
                model_dirs.append(parent)

        cache_dir = os.path.expanduser("~/.cache/llama-smart-gpu")
        cache_file = os.path.join(cache_dir, "model_cache.json")

        self._discovery = ModelDiscovery(model_dirs, cache_file)
        self._download_mgr = DownloadManager(config.model_dir)
        self._current_model: Optional[ModelEntry] = None
        self._lock = threading.Lock()

        # Lifecycle hooks
        self._hooks: Dict[str, List[Callable]] = {
            "pre_load": [], "post_load": [], "pre_unload": [],
            "post_unload": [], "on_error": [], "on_validate": [],
        }

        # Model performance history (persisted)
        self._perf_history_file = os.path.join(cache_dir, "perf_history.json")
        self._perf_history: Dict[str, Dict] = {}
        self._load_perf_history()

    # ── Engine Binding ───────────────────────────────────────────────────

    def set_engine(self, engine):
        """Set the LlamaEngine instance for model loading."""
        self._engine = engine

    # ── Lifecycle Hooks ──────────────────────────────────────────────────

    def register_hook(self, event: str, callback: Callable):
        """Register a lifecycle hook callback.

        Events: pre_load, post_load, pre_unload, post_unload, on_error, on_validate
        """
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _fire_hook(self, event: str, *args, **kwargs):
        for cb in self._hooks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception as exc:
                logger.error("Hook callback error (%s): %s", event, exc)

    # ── Discovery ────────────────────────────────────────────────────────

    def discover_models(self, force: bool = False) -> List[ModelEntry]:
        """Scan for available models and update VRAM estimates."""
        models = self._discovery.scan(force=force)
        gpu_info = self._gpu_manager.get_memory_info()
        available_vram = self._gpu_manager.get_available_vram()

        for model in models:
            if model.n_params_b > 0 and model.n_layers > 0:
                # Full offload estimate
                model.estimated_vram_full_mb = VRAMEstimator.estimate_model_vram(
                    model.n_params_b, model.n_layers, model.quantization,
                    n_gpu_layers=model.n_layers,
                    n_ctx=self._config.performance.n_ctx,
                    n_heads=model.n_heads,
                )
                # CPU-only estimate
                model.estimated_vram_min_mb = 0  # CPU only needs no VRAM

                # VRAM fit assessment
                model.fits_in_vram = model.estimated_vram_full_mb <= available_vram
                if available_vram > 0 and model.estimated_vram_full_mb > 0:
                    model.vram_fit_score = max(0, 1.0 - (model.estimated_vram_full_mb / available_vram))
                else:
                    model.vram_fit_score = 0.0

                # Compute optimal GPU layers
                plan = VRAMEstimator.compute_optimal_gpu_layers(
                    model.n_params_b, model.n_layers,
                    available_vram, model.quantization,
                    n_ctx=self._config.performance.n_ctx,
                    n_heads=model.n_heads,
                )
                model.recommended_gpu_layers = plan.gpu_layers
                model.estimated_vram_optimal_mb = plan.estimated_vram_mb

                # Generate auto-config
                model.auto_config = ModelClassifier.suggest_auto_config(
                    model, available_vram, self._config.performance.n_ctx
                )

                # Update status
                if model.status == ModelStatus.DISCOVERED.value:
                    model.status = ModelStatus.VALIDATED.value

                # Restore perf history
                hist = self._perf_history.get(model.name, {})
                model.avg_tps = hist.get("avg_tps", 0)
                model.best_tps = hist.get("best_tps", 0)
                model.total_generations = hist.get("total_generations", 0)

        return models

    # ── Query & Search ───────────────────────────────────────────────────

    def list_models(self) -> List[Dict[str, Any]]:
        """Return model list as dicts for API responses."""
        models = self._discovery.get_models()
        result = []
        for m in models:
            result.append({
                "name": m.name,
                "aliases": m.aliases,
                "tags": m.tags,
                "family": m.family,
                "path": m.path,
                "size_mb": round(m.size_mb, 1),
                "size_gb": round(m.size_gb, 2),
                "params_b": round(m.n_params_b, 2),
                "quantization": m.quantization,
                "bits_per_weight": m.bits_per_weight,
                "quality_tier": m.quant_info.get("quality_tier", "unknown"),
                "recommended_use": m.quant_info.get("recommended_use", "unknown"),
                "context_length": m.context_length,
                "n_layers": m.n_layers,
                "n_heads": m.n_heads,
                "architecture": m.architecture,
                "estimated_vram_full_mb": round(m.estimated_vram_full_mb, 0),
                "estimated_vram_optimal_mb": round(m.estimated_vram_optimal_mb, 0),
                "recommended_gpu_layers": m.recommended_gpu_layers,
                "fits_in_vram": m.fits_in_vram,
                "vram_fit_score": round(m.vram_fit_score, 2),
                "status": m.status,
                "loaded": m.loaded,
                "avg_tps": round(m.avg_tps, 1),
                "load_count": m.load_count,
                "has_chat_template": bool(m.chat_template),
                "auto_config": m.auto_config,
            })
        return result

    def search_models(self, query: str = "", tags: Optional[List[str]] = None,
                      family: Optional[str] = None, min_params: float = 0,
                      max_params: float = float("inf"),
                      fits_vram_only: bool = False,
                      sort_by: str = "name") -> List[Dict[str, Any]]:
        """Search and filter models with multi-criteria support."""
        results = self._discovery.search(
            query=query, tags=tags, family=family,
            min_params=min_params, max_params=max_params,
            fits_vram_only=fits_vram_only, sort_by=sort_by,
        )
        # Convert to API-friendly dicts
        return [
            {
                "name": m.name, "aliases": m.aliases, "tags": m.tags,
                "family": m.family, "path": m.path,
                "size_mb": round(m.size_mb, 1), "size_gb": round(m.size_gb, 2),
                "params_b": round(m.n_params_b, 2),
                "quantization": m.quantization,
                "bits_per_weight": m.bits_per_weight,
                "quality_tier": m.quant_info.get("quality_tier", "unknown"),
                "context_length": m.context_length, "n_layers": m.n_layers,
                "estimated_vram_full_mb": round(m.estimated_vram_full_mb, 0),
                "fits_in_vram": m.fits_in_vram,
                "vram_fit_score": round(m.vram_fit_score, 2),
                "status": m.status,
            }
            for m in results
        ]

    def get_model_detail(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information including validation status."""
        entry = self._discovery.get_model(name)
        if not entry:
            return None
        d = asdict(entry)
        # Round floats for API
        for key in ("size_mb", "size_gb", "n_params_b", "bits_per_weight",
                     "estimated_vram_full_mb", "estimated_vram_optimal_mb",
                     "estimated_vram_min_mb", "vram_fit_score", "avg_tps", "best_tps"):
            if key in d and isinstance(d[key], float):
                d[key] = round(d[key], 2)
        return d

    # ── Selection & Recommendation ───────────────────────────────────────

    def select_best_model(self, use_case: str = "general") -> Optional[ModelEntry]:
        """Auto-select the best model for a given use case.

        Considers VRAM budget, use case (chat/code/reasoning), quality tier,
        and available performance history to pick the optimal model.
        """
        models = self.discover_models()
        candidates = [m for m in models if m.n_params_b > 0]
        if not candidates:
            return None

        # Score each model
        scored = []
        for m in candidates:
            score = 0.0

            # VRAM fit is critical — prefer models that fit well
            if m.fits_in_vram:
                score += 40 + m.vram_fit_score * 20
            else:
                score -= 50  # Strongly penalize models that don't fit

            # Prefer larger models (better quality) among those that fit
            score += min(m.n_params_b, 70) * 0.5  # Cap at 70B

            # Quality tier bonus
            tier_scores = {"highest": 15, "high": 12, "medium": 8, "low": 3}
            score += tier_scores.get(m.quant_info.get("quality_tier", ""), 5)

            # Use case match
            recommended = m.quant_info.get("recommended_use", "general")
            if use_case == recommended:
                score += 10
            elif use_case == "general":
                score += 5  # General is fine for everything

            # Has chat template bonus for chat use
            if use_case in ("chat",) and m.chat_template:
                score += 8

            # Performance history bonus
            if m.avg_tps > 0:
                score += min(m.avg_tps / 10, 10)  # Up to 10 bonus points

            # Instruct variants preferred for chat
            if use_case == "chat" and "instruct" in m.tags:
                score += 5

            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def compare_models(self, names: List[str]) -> Dict[str, Any]:
        """Compare multiple models side-by-side."""
        models = []
        for name in names:
            entry = self._discovery.get_model(name)
            if entry:
                models.append(entry)

        if len(models) < 2:
            return {"error": "Need at least 2 models to compare"}

        comparison = {
            "models": [],
            "verdict": "",
        }

        best_score = -1
        best_name = ""

        for m in models:
            score = 0
            if m.fits_in_vram:
                score += 30
            score += min(m.n_params_b, 70) * 0.5
            tier_scores = {"highest": 15, "high": 12, "medium": 8, "low": 3}
            score += tier_scores.get(m.quant_info.get("quality_tier", ""), 5)
            if m.avg_tps > 0:
                score += min(m.avg_tps / 10, 10)

            comparison["models"].append({
                "name": m.name,
                "params_b": m.n_params_b,
                "quantization": m.quantization,
                "quality_tier": m.quant_info.get("quality_tier", ""),
                "size_mb": round(m.size_mb, 0),
                "estimated_vram_mb": round(m.estimated_vram_full_mb, 0),
                "fits_vram": m.fits_in_vram,
                "vram_fit_score": round(m.vram_fit_score, 2),
                "context_length": m.context_length,
                "avg_tps": round(m.avg_tps, 1),
                "has_chat_template": bool(m.chat_template),
                "recommended_gpu_layers": m.recommended_gpu_layers,
                "tags": m.tags,
                "score": round(score, 1),
            })

            if score > best_score:
                best_score = score
                best_name = m.name

        comparison["verdict"] = f"Recommended: {best_name} (score: {best_score:.1f})"
        return comparison

    def get_recommendations(self, use_case: str = "general") -> List[ModelRecommendation]:
        """Get ranked model recommendations for a use case."""
        models = self.discover_models()
        available_vram = self._gpu_manager.get_available_vram()
        recommendations = []

        for m in models:
            if m.n_params_b <= 0:
                continue

            qi = ModelClassifier.get_quant_info(m.quantization)
            plan = VRAMEstimator.compute_optimal_gpu_layers(
                m.n_params_b, m.n_layers, available_vram, m.quantization,
                n_ctx=self._config.performance.n_ctx, n_heads=m.n_heads,
            )
            auto_cfg = ModelClassifier.suggest_auto_config(m, available_vram, self._config.performance.n_ctx)

            rec = ModelRecommendation(
                model_name=m.name,
                reason=self._generate_recommendation_reason(m, use_case, plan),
                confidence=m.vram_fit_score if m.fits_in_vram else 0.3,
                suggested_n_ctx=auto_cfg.get("n_ctx", 4096),
                suggested_n_batch=auto_cfg.get("n_batch", 512),
                suggested_cache_k=auto_cfg.get("cache_type_k", "f16"),
                suggested_cache_v=auto_cfg.get("cache_type_v", "f16"),
                suggested_n_gpu_layers=plan.gpu_layers,
                estimated_vram_mb=plan.estimated_vram_mb,
            )
            recommendations.append(rec)

        # Sort by confidence then by estimated VRAM efficiency
        recommendations.sort(key=lambda r: (r.confidence, -r.estimated_vram_mb), reverse=True)
        return recommendations

    @staticmethod
    def _generate_recommendation_reason(entry: ModelEntry, use_case: str,
                                         plan) -> str:
        """Generate a human-readable reason for a recommendation."""
        reasons = []
        if entry.fits_in_vram:
            reasons.append(f"Fits in VRAM ({entry.estimated_vram_full_mb:.0f} MB)")
        else:
            reasons.append(f"Partial offload ({plan.gpu_layers}/{entry.n_layers} layers on GPU)")

        qi = ModelClassifier.get_quant_info(entry.quantization)
        if qi.quality_tier in ("highest", "high"):
            reasons.append(f"High quality ({qi.format_name})")
        if use_case == "chat" and entry.chat_template:
            reasons.append("Has chat template")
        if "instruct" in entry.tags:
            reasons.append("Instruct-tuned")
        if entry.context_length >= 32768:
            reasons.append(f"Long context ({entry.context_length} tokens)")
        return " | ".join(reasons) if reasons else "Available model"

    # ── Validation ───────────────────────────────────────────────────────

    def validate_model(self, name: str, deep: bool = False) -> Dict[str, Any]:
        """Validate a model's integrity.

        Args:
            name: Model name
            deep: If True, also attempt a probe load (slow)

        Returns:
            Dict with validation results
        """
        entry = self._discovery.get_model(name)
        if not entry:
            return {"valid": False, "errors": [f"Model not found: {name}"]}

        # Quick validation
        quick_ok, quick_errors = ModelValidator.validate_quick(entry)

        # Deep validation (optional)
        deep_ok = None
        deep_errors = []
        if deep:
            deep_ok, deep_errors = ModelValidator.validate_deep(entry.path)

        # Update entry status
        if quick_ok and (deep_ok is None or deep_ok):
            entry.status = ModelStatus.VALIDATED.value
            entry.last_validated_at = time.time()
            entry.validation_errors = []
        else:
            entry.status = ModelStatus.CORRUPT.value
            entry.validation_errors = quick_errors + deep_errors

        self._fire_hook("on_validate", entry, quick_ok, deep_ok)

        return {
            "name": name,
            "quick_check": {"passed": quick_ok, "errors": quick_errors},
            "deep_check": {"passed": deep_ok, "errors": deep_errors} if deep is not None else None,
            "status": entry.status,
        }

    def compute_full_hash(self, name: str) -> str:
        """Compute full SHA256 hash for a model (slow for large files)."""
        entry = self._discovery.get_model(name)
        if not entry:
            raise FileNotFoundError(f"Model not found: {name}")
        path = Path(entry.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {entry.path}")
        logger.info("Computing SHA256 for %s (this may take a while)...", name)
        sha = ModelValidator.full_hash(path)
        entry.file_hash_sha256 = sha
        logger.info("SHA256: %s", sha)
        return sha

    # ── Loading ──────────────────────────────────────────────────────────

    def load_model(self, name_or_path: str, n_gpu_layers: Optional[int] = None) -> ModelEntry:
        """Load a model by name, alias, or path.

        Performs pre-flight checks:
          - File exists
          - Model validates
          - VRAM sufficiency warning
        Then hot-swaps the model with lifecycle hooks.
        """
        if not self._engine:
            raise RuntimeError("Engine not set — call set_engine() first")

        # Resolve model entry
        entry = self._discovery.get_model(name_or_path)
        if not entry:
            path = Path(name_or_path)
            if path.exists():
                entry = self._discovery.find_by_path(str(path))
                if not entry:
                    entry = ModelEntry(
                        name=path.stem, path=str(path),
                        status=ModelStatus.DISCOVERED.value,
                    )
            else:
                raise FileNotFoundError(f"Model not found: {name_or_path}")

        # Pre-flight validation
        quick_ok, errors = ModelValidator.validate_quick(entry)
        if not quick_ok:
            self._fire_hook("on_error", entry, "validation_failed", errors)
            raise RuntimeError(f"Model validation failed: {errors}")

        # VRAM warning
        available = self._gpu_manager.get_available_vram()
        if entry.estimated_vram_full_mb > available:
            logger.warning(
                "Model may exceed VRAM: estimated %.0f MB but only %.0f MB available. "
                "Will use partial GPU offloading.",
                entry.estimated_vram_full_mb, available,
            )

        # Pre-load hook
        self._fire_hook("pre_load", entry)

        # Hot-swap
        if self._current_model and self._current_model.loaded:
            logger.info("Hot-swapping: unloading %s", self._current_model.name)
            self._fire_hook("pre_unload", self._current_model)
            self._engine.unload_model()
            self._current_model.loaded = False
            self._current_model.status = ModelStatus.READY.value
            self._fire_hook("post_unload", self._current_model)

        # Override GPU layers if specified
        if n_gpu_layers is not None:
            self._config.gpu.n_gpu_layers = n_gpu_layers
        elif entry.recommended_gpu_layers > 0:
            self._config.gpu.n_gpu_layers = entry.recommended_gpu_layers

        # Load
        with self._lock:
            self._engine.load_model(entry.path)
            entry.loaded = True
            entry.status = ModelStatus.LOADED.value
            entry.last_loaded_at = time.time()
            entry.load_count += 1
            self._current_model = entry

        # Post-load hook
        self._fire_hook("post_load", entry)

        logger.info(
            "Model loaded: %s (%.1fB params, %s, %d GPU layers)",
            entry.name, entry.n_params_b, entry.quantization,
            self._config.gpu.n_gpu_layers,
        )
        return entry

    def unload_model(self):
        """Unload the current model."""
        if self._current_model:
            self._fire_hook("pre_unload", self._current_model)
        if self._engine:
            self._engine.unload_model()
        with self._lock:
            if self._current_model:
                self._current_model.loaded = False
                self._current_model.status = ModelStatus.READY.value
                self._fire_hook("post_unload", self._current_model)
                self._current_model = None

    def get_current_model(self) -> Optional[ModelEntry]:
        """Return the currently loaded model entry."""
        return self._current_model

    # ── Aliases & Tags ───────────────────────────────────────────────────

    def set_alias(self, model_name: str, alias: str):
        """Add an alias for a model."""
        entry = self._discovery.get_model(model_name)
        if entry and alias not in entry.aliases:
            entry.aliases.append(alias)

    def remove_alias(self, model_name: str, alias: str):
        """Remove an alias from a model."""
        entry = self._discovery.get_model(model_name)
        if entry and alias in entry.aliases:
            entry.aliases.remove(alias)

    def add_tag(self, model_name: str, tag: str):
        """Add a tag to a model."""
        entry = self._discovery.get_model(model_name)
        if entry and tag not in entry.tags:
            entry.tags.append(tag)

    def remove_tag(self, model_name: str, tag: str):
        """Remove a tag from a model."""
        entry = self._discovery.get_model(model_name)
        if entry and tag in entry.tags:
            entry.tags.remove(tag)

    # ── Downloads ────────────────────────────────────────────────────────

    def download_model(self, url: str, name: Optional[str] = None,
                       priority: int = 2, expected_sha256: str = "") -> DownloadTask:
        """Submit a model download. Returns a DownloadTask for tracking."""
        return self._download_mgr.submit(url, name, priority, expected_sha256)

    def start_downloads(self):
        """Start the download worker pool."""
        self._download_mgr.start()

    def stop_downloads(self):
        """Stop the download workers."""
        self._download_mgr.stop()

    def cancel_download(self, task_id: str) -> bool:
        """Cancel a download."""
        return self._download_mgr.cancel(task_id)

    def get_download_status(self, task_id: str) -> Optional[Dict]:
        """Get download task status."""
        task = self._download_mgr.get_task(task_id)
        return asdict(task) if task else None

    def list_downloads(self) -> Dict[str, List[Dict]]:
        """List all active and completed downloads."""
        return {
            "active": self._download_mgr.list_active(),
            "completed": self._download_mgr.list_completed(),
        }

    # ── Performance Tracking ─────────────────────────────────────────────

    def record_performance(self, model_name: str, tps: float, tokens: int = 0):
        """Record performance data for a model."""
        entry = self._discovery.get_model(model_name)
        if not entry:
            return
        if tps > 0:
            if entry.avg_tps > 0:
                entry.avg_tps = (entry.avg_tps * 0.8) + (tps * 0.2)  # EMA
            else:
                entry.avg_tps = tps
            entry.best_tps = max(entry.best_tps, tps)
        if tokens > 0:
            entry.total_tokens_generated += tokens
            entry.total_generations += 1

        # Persist
        self._perf_history[model_name] = {
            "avg_tps": entry.avg_tps,
            "best_tps": entry.best_tps,
            "total_generations": entry.total_generations,
            "total_tokens": entry.total_tokens_generated,
        }
        self._save_perf_history()

    # ── Disk Management ──────────────────────────────────────────────────

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information for model directories."""
        total_size = 0
        model_count = 0
        for model_dir in self._discovery._model_dirs:
            if model_dir.exists():
                for f in model_dir.rglob("*.gguf"):
                    total_size += f.stat().st_size
                    model_count += 1

        # Check free disk space
        try:
            stat = shutil.disk_usage(self._config.model_dir)
            free_gb = stat.free / (1024 ** 3)
            total_gb = stat.total / (1024 ** 3)
            used_pct = (stat.used / stat.total) * 100
        except Exception:
            free_gb = total_gb = used_pct = 0

        return {
            "model_files_count": model_count,
            "total_model_size_gb": round(total_size / (1024 ** 3), 2),
            "disk_free_gb": round(free_gb, 2),
            "disk_total_gb": round(total_gb, 2),
            "disk_used_pct": round(used_pct, 1),
        }

    def cleanup_partial_downloads(self) -> List[str]:
        """Remove partial/incomplete download files."""
        cleaned = []
        model_dir = Path(self._config.model_dir)
        if not model_dir.exists():
            return cleaned

        for f in model_dir.glob("*.part"):
            try:
                f.unlink()
                cleaned.append(str(f))
                logger.info("Cleaned up partial download: %s", f)
            except Exception as exc:
                logger.error("Failed to clean up %s: %s", f, exc)

        for f in model_dir.glob("*.tmp"):
            try:
                f.unlink()
                cleaned.append(str(f))
                logger.info("Cleaned up temp file: %s", f)
            except Exception as exc:
                logger.error("Failed to clean up %s: %s", f, exc)

        return cleaned

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_perf_history(self):
        """Load model performance history from disk."""
        try:
            if os.path.exists(self._perf_history_file):
                with open(self._perf_history_file, "r") as f:
                    self._perf_history = json.load(f)
        except Exception as exc:
            logger.debug("Could not load perf history: %s", exc)

    def _save_perf_history(self):
        """Persist model performance history to disk."""
        try:
            Path(self._perf_history_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self._perf_history_file, "w") as f:
                json.dump(self._perf_history, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save perf history: %s", exc)


# ==============================================================================
# SECTION: inference.py
# ==============================================================================

"""
Inference Pipeline — High-level inference with batching, sessions, and smart routing.

Features:
  - Session-based conversations with context management
  - Automatic context window recycling (sliding window)
  - Request queuing with priority support
  - Continuous batching scheduler
  - Retry logic with backoff
  - Graceful degradation under VRAM pressure
"""


# ─── Enums ──────────────────────────────────────────────────────────────────

class RequestPriority(Enum):
    """Priority levels for inference request scheduling (lower = higher priority)."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

    def __str__(self): return f"RequestPriority.{self.name}"


class RequestStatus(Enum):
    """Lifecycle states for an inference request."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        return self in (RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED)

    def __str__(self): return f"RequestStatus.{self.name}"


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    """A single inference request in the pipeline."""

    def __repr__(self) -> str:
        return f"InferenceRequest(id={self.id!r}, status={self.status.value!r}, priority={self.priority.name})"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    prompt: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    priority: RequestPriority = RequestPriority.NORMAL
    status: RequestStatus = RequestStatus.QUEUED
    stream: bool = False
    # Override generation params
    params: Dict[str, Any] = field(default_factory=dict)
    # Result
    result: Optional[Any] = None
    error: Optional[str] = None
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    # Callbacks
    on_token: Optional[Callable] = None
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None


@dataclass
class Session:
    """Conversation session with persistent context."""

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, msgs={len(self.messages)}, tokens={self.total_tokens})"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    total_tokens: int = 0
    max_context_tokens: int = 4096
    system_prompt: str = ""


# ─── Session Manager ────────────────────────────────────────────────────────

class SessionManager:
    """Manage conversation sessions with context recycling.

    When a session's token count approaches the context window limit,
    older messages are pruned (keeping the system prompt and recent messages).
    """

    def __init__(self, max_context_tokens: int = 4096, max_sessions: int = 100,
                 session_ttl_seconds: float = 3600):
        self._max_ctx = max_context_tokens
        self._max_sessions = max_sessions
        self._ttl = session_ttl_seconds
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create_session(self, system_prompt: str = "", max_context: Optional[int] = None) -> Session:
        """Create a new conversation session."""
        session = Session(
            max_context_tokens=max_context or self._max_ctx,
            system_prompt=system_prompt,
        )
        with self._lock:
            # Evict expired sessions if at capacity
            if len(self._sessions) >= self._max_sessions:
                self._evict_expired()
            self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_used_at = time.time()
            return session

    def add_message(self, session_id: str, role: str, content: str) -> Optional[Session]:
        """Add a message to a session and check context limits."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            session.messages.append({"role": role, "content": content})
            session.last_used_at = time.time()
            # Rough token estimate (4 chars per token)
            session.total_tokens = sum(len(m["content"]) // 4 for m in session.messages)

            # Recycle context if approaching limit
            if session.total_tokens > session.max_context_tokens * 0.85:
                self._recycle_context(session)
            return session

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Get all messages for a session, prepending system prompt."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            msgs = []
            if session.system_prompt:
                msgs.append({"role": "system", "content": session.system_prompt})
            msgs.extend(session.messages)
            return msgs

    def delete_session(self, session_id: str):
        """Delete a session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        with self._lock:
            return [
                {
                    "id": s.id,
                    "messages": len(s.messages),
                    "total_tokens": s.total_tokens,
                    "last_used_at": s.last_used_at,
                }
                for s in self._sessions.values()
            ]

    def _recycle_context(self, session: Session):
        """Remove oldest messages to stay within context limit.

        Keeps system prompt and the most recent messages.
        """
        target_tokens = int(session.max_context_tokens * 0.6)
        while session.total_tokens > target_tokens and len(session.messages) > 2:
            # Remove oldest non-system message
            removed = session.messages.pop(0)
            session.total_tokens = sum(len(m["content"]) // 4 for m in session.messages)
            logger.debug("Recycled context for session %s: removed message", session.id)

    def _evict_expired(self):
        """Remove sessions that have exceeded their TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used_at > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug("Evicted %d expired sessions", len(expired))


# ─── Inference Pipeline ─────────────────────────────────────────────────────

class InferencePipeline:
    """High-level inference pipeline with queuing, batching, and session management.

    This is the primary interface for the API layer. It handles:
      - Request queuing with priority
      - Session management
      - VRAM-aware request scheduling
      - Streaming and non-streaming generation
      - Graceful degradation under memory pressure
    """

    def __init__(self, engine: LlamaEngine, gpu_manager: SmartGPUManager,
                 perf_monitor: PerformanceMonitor, inference_config: InferenceConfig):
        self._engine = engine
        self._gpu = gpu_manager
        self._perf = perf_monitor
        self._config = inference_config

        self._sessions = SessionManager(max_context_tokens=4096)
        self._request_queue: PriorityQueue = PriorityQueue()
        self._active_requests: Dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()

        # Worker threads
        self._running = False
        self._workers: List[threading.Thread] = []
        self._n_workers = 2  # Process up to 2 requests concurrently

        # Stats
        self._total_requests = 0
        self._total_failed = 0

        # Rate limiter: default 60 req/s
        self._rate_limiter = TokenBucketRateLimiter(rate=60.0, capacity=120)

    # ── Public API ───────────────────────────────────────────────────────

    def start(self, n_workers: int = 2):
        """Start the inference pipeline workers."""
        self._n_workers = n_workers
        self._running = True
        for i in range(n_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True,
                                 name=f"inference-worker-{i}")
            t.start()
            self._workers.append(t)
        logger.info("Inference pipeline started with %d workers", n_workers)

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        # Put sentinel items to unblock workers
        for _ in self._workers:
            self._request_queue.put((99, "", None))
        for t in self._workers:
            t.join(timeout=10)
        self._workers.clear()

    def submit(self, request: InferenceRequest) -> str:
        """Submit an inference request. Returns the request ID."""
        self._total_requests += 1
        with self._lock:
            self._active_requests[request.id] = request
        self._request_queue.put((request.priority.value, request.created_at, request))
        return request.id

    def generate(self, prompt: str, session_id: Optional[str] = None,
                 stream: bool = False, **kwargs) -> Any:
        """Convenience method: generate text from a prompt.

        For simple usage without explicit request objects.
        """
        req = InferenceRequest(
            session_id=session_id or "",
            prompt=prompt,
            stream=stream,
            params=kwargs,
        )

        if stream:
            return self._generate_stream_sync(req)
        else:
            return self._generate_sync(req)

    def chat(self, messages: List[Dict[str, str]], session_id: Optional[str] = None,
             stream: bool = False, **kwargs) -> Any:
        """Convenience method: chat-style generation."""
        req = InferenceRequest(
            session_id=session_id or "",
            messages=messages,
            stream=stream,
            params=kwargs,
        )

        if stream:
            return self._chat_stream_sync(req)
        else:
            return self._chat_sync(req)

    # ── Session Management ───────────────────────────────────────────────

    def create_session(self, system_prompt: str = "") -> Session:
        """Create a new conversation session."""
        return self._sessions.create_session(system_prompt, self._config.n_predict)

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Get messages from a session."""
        return self._sessions.get_messages(session_id)

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "running": self._running,
            "queue_size": self._request_queue.qsize(),
            "active_requests": len(self._active_requests),
            "workers": len(self._workers),
            "total_requests": self._total_requests,
            "total_failed": self._total_failed,
            "sessions": len(self._sessions.list_sessions()),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _generate_sync(self, request: InferenceRequest) -> Any:
        """Synchronous (non-streaming) text generation."""
        if not self._rate_limiter.acquire(timeout=5.0):
            raise InferenceError(reason="Rate limit exceeded", request_id=request.id)
        request.status = RequestStatus.RUNNING
        request.started_at = time.time()

        try:
            # Check VRAM pressure — degrade gracefully
            pressure = self._gpu.get_pressure()
            params = dict(request.params)
            if pressure == VRAMPressure.CRITICAL:
                # Reduce max tokens to avoid OOM
                params.setdefault("n_predict", min(self._config.n_predict, 128))
                logger.warning("VRAM CRITICAL — reducing generation length")

            result = self._engine.generate(request.prompt, **params)
            self._perf.record_from_result(result)

            request.result = result
            request.status = RequestStatus.COMPLETED
            request.completed_at = time.time()

            if request.on_complete:
                request.on_complete(result)

            return result

        except Exception as exc:
            request.status = RequestStatus.FAILED
            request.error = str(exc)
            self._total_failed += 1
            if request.on_error:
                request.on_error(exc)
            raise

    def _generate_stream_sync(self, request: InferenceRequest) -> Iterator[str]:
        """Streaming text generation."""
        request.status = RequestStatus.RUNNING
        request.started_at = time.time()

        try:
            for token in self._engine.generate_stream(request.prompt, **request.params):
                if request.on_token:
                    request.on_token(token)
                yield token

            request.status = RequestStatus.COMPLETED
            request.completed_at = time.time()

        except Exception as exc:
            request.status = RequestStatus.FAILED
            request.error = str(exc)
            self._total_failed += 1
            raise

    def _chat_sync(self, request: InferenceRequest) -> Any:
        """Synchronous chat generation."""
        request.status = RequestStatus.RUNNING
        request.started_at = time.time()

        try:
            result = self._engine.chat(request.messages, **request.params)
            self._perf.record_from_result(result)

            # Update session if applicable
            if request.session_id:
                self._sessions.add_message(request.session_id, "assistant", result.text)

            request.result = result
            request.status = RequestStatus.COMPLETED
            request.completed_at = time.time()
            return result

        except Exception as exc:
            request.status = RequestStatus.FAILED
            request.error = str(exc)
            self._total_failed += 1
            raise

    def _chat_stream_sync(self, request: InferenceRequest) -> Iterator[Dict]:
        """Streaming chat generation."""
        request.status = RequestStatus.RUNNING
        request.started_at = time.time()

        try:
            full_text = ""
            for delta in self._engine.chat_stream(request.messages, **request.params):
                if delta.get("content"):
                    full_text += delta["content"]
                yield delta

            # Update session
            if request.session_id and full_text:
                self._sessions.add_message(request.session_id, "assistant", full_text)

            request.status = RequestStatus.COMPLETED
            request.completed_at = time.time()

        except Exception as exc:
            request.status = RequestStatus.FAILED
            request.error = str(exc)
            self._total_failed += 1
            raise

    def _worker_loop(self):
        """Worker thread that processes requests from the queue."""
        while self._running:
            try:
                priority, _, request = self._request_queue.get(timeout=1.0)
                if request is None:
                    break

                # Process the request synchronously within the worker
                # (streaming requests are handled differently)
                try:
                    if request.messages:
                        self._chat_sync(request)
                    else:
                        self._generate_sync(request)
                except Exception:
                    pass  # Error already recorded in request

                # Clean up
                with self._lock:
                    self._active_requests.pop(request.id, None)

            except Exception:
                continue  # Timeout or empty queue, just retry


# ==============================================================================
# SECTION: api.py
# ==============================================================================

"""
FastAPI Server — REST API with OpenAI-compatible endpoints and model management.

Endpoints:
  POST /v1/chat/completions       — OpenAI-compatible chat
  POST /v1/completions            — OpenAI-compatible text completion
  POST /generate                   — Simple generation

  GET  /models                    — List available models (with search/filter)
  GET  /models/{name}             — Get detailed model info
  POST /models/load               — Load a model
  POST /models/unload             — Unload current model
  POST /models/validate/{name}    — Validate model integrity
  POST /models/hash/{name}        — Compute full SHA256 hash
  POST /models/alias              — Set/remove model alias
  POST /models/tag                — Add/remove model tag
  GET  /models/compare            — Compare models side-by-side
  GET  /models/recommendations    — Get model recommendations
  POST /models/download           — Submit a model download
  GET  /models/downloads          — List download status
  DELETE /models/downloads/{id}   — Cancel a download
  GET  /models/disk               — Disk usage info
  POST /models/cleanup            — Cleanup partial downloads

  GET  /health                    — Health check
  GET  /metrics                   — Prometheus metrics
  GET  /status                    — Full system status
  GET  /gpu                       — GPU memory info
  POST /sessions                  — Create chat session
  GET  /sessions                  — List sessions
  GET  /benchmark                 — Run quick benchmark
"""

from fastapi import FastAPI, HTTPException, Request, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field


# ─── Pydantic Models ────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    repeat_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    model_config = {"populate_by_name": True}

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    repeat_penalty: Optional[float] = None

    model_config = {"populate_by_name": True}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    session_id: Optional[str] = None

class LoadModelRequest(BaseModel):
    name: str
    n_gpu_layers: Optional[int] = None

class CreateSessionRequest(BaseModel):
    system_prompt: Optional[str] = ""

class AliasRequest(BaseModel):
    model_name: str
    alias: str
    action: str = "add"  # "add" or "remove"

class TagRequest(BaseModel):
    model_name: str
    tag: str
    action: str = "add"  # "add" or "remove"

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None
    priority: Optional[int] = 2
    expected_sha256: Optional[str] = ""

class CompareRequest(BaseModel):
    models: List[str]

class ValidateRequest(BaseModel):
    deep: bool = False


# ─── App Factory ────────────────────────────────────────────────────────────

def create_app(config: SmartGPUConfig, engine: LlamaEngine,
               gpu_manager: SmartGPUManager, model_manager: ModelManager,
               perf_monitor: PerformanceMonitor,
               pipeline: InferencePipeline) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup and shutdown lifecycle."""
        logger.info("Starting llama-cpp-smart-gpu server")
        model_manager.start_downloads()
        pipeline.start(n_workers=2)
        perf_monitor.start_reporting(
            callback=lambda r: logger.info(
                "Perf: %.1f tok/s | p50 latency=%.0f ms | health=%s",
                r.avg_tokens_per_second, r.latency_gen_ms.median, r.health_status,
            )
        )
        yield
        logger.info("Shutting down server")
        pipeline.stop()
        perf_monitor.stop_reporting()
        gpu_manager.stop_monitoring()
        model_manager.stop_downloads()
        engine.unload_model()

    app = FastAPI(
        title="llama-cpp-smart-gpu",
        description="Smart GPU LLM inference engine powered by llama.cpp",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references
    app.state.config = config
    app.state.engine = engine
    app.state.gpu = gpu_manager
    app.state.models = model_manager
    app.state.perf = perf_monitor
    app.state.pipeline = pipeline

    # Auth dependency
    async def verify_api_key(authorization: Optional[str] = Header(None)):
        if not config.server.api_key:
            return True
        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")
        key = authorization.replace("Bearer ", "")
        if key != config.server.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # OpenAI-Compatible Endpoints
    # ═══════════════════════════════════════════════════════════════════════

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, auth=Depends(verify_api_key)):
        """OpenAI-compatible chat completion endpoint."""
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        params = {}
        if req.temperature is not None:
            params["temperature"] = req.temperature
        if req.top_p is not None:
            params["top_p"] = req.top_p
        if req.top_k is not None:
            params["top_k"] = req.top_k
        if req.max_tokens is not None:
            params["n_predict"] = req.max_tokens
        if req.stop is not None:
            params["stop"] = req.stop
        if req.seed is not None:
            params["seed"] = req.seed
        if req.repeat_penalty is not None:
            params["repeat_penalty"] = req.repeat_penalty
        if req.frequency_penalty is not None:
            params["frequency_penalty"] = req.frequency_penalty
        if req.presence_penalty is not None:
            params["presence_penalty"] = req.presence_penalty

        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        if req.stream:
            async def stream_response():
                for delta in engine.chat_stream(messages, **params):
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model or "llama-cpp",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_response(), media_type="text/event-stream")

        result = engine.chat(messages, **params)
        perf_monitor.record_from_result(result)

        current_model = model_manager.get_current_model()
        model_name = current_model.name if current_model else req.model or "llama-cpp"

        # Record model performance
        if current_model and result.tokens_per_second > 0:
            model_manager.record_performance(current_model.name, result.tokens_per_second, result.tokens_generated)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "total_tokens": result.prompt_tokens + result.tokens_generated,
            },
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, auth=Depends(verify_api_key)):
        """OpenAI-compatible text completion endpoint."""
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        params = {}
        if req.temperature is not None:
            params["temperature"] = req.temperature
        if req.top_p is not None:
            params["top_p"] = req.top_p
        if req.top_k is not None:
            params["top_k"] = req.top_k
        if req.max_tokens is not None:
            params["n_predict"] = req.max_tokens
        if req.stop is not None:
            params["stop"] = req.stop
        if req.seed is not None:
            params["seed"] = req.seed
        if req.repeat_penalty is not None:
            params["repeat_penalty"] = req.repeat_penalty

        if req.stream:
            async def stream_response():
                for token in engine.generate_stream(req.prompt, **params):
                    chunk = {
                        "id": f"cmpl-{int(time.time())}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model or "llama-cpp",
                        "choices": [{"index": 0, "text": token, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_response(), media_type="text/event-stream")

        result = engine.generate(req.prompt, **params)
        perf_monitor.record_from_result(result)

        current_model = model_manager.get_current_model()
        if current_model and result.tokens_per_second > 0:
            model_manager.record_performance(current_model.name, result.tokens_per_second, result.tokens_generated)

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or "llama-cpp",
            "choices": [{"index": 0, "text": result.text, "finish_reason": result.finish_reason}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "total_tokens": result.prompt_tokens + result.tokens_generated,
            },
        }

    # ── Simple Generation ────────────────────────────────────────────────

    @app.post("/generate")
    async def simple_generate(req: GenerateRequest, auth=Depends(verify_api_key)):
        """Simple generation endpoint."""
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        params = {
            "n_predict": req.max_tokens or 512,
            "temperature": req.temperature or 0.7,
            "top_p": req.top_p or 0.95,
            "top_k": req.top_k or 40,
        }
        if req.stop:
            params["stop"] = req.stop

        if req.stream:
            async def stream_response():
                for token in engine.generate_stream(req.prompt, **params):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_response(), media_type="text/event-stream")

        result = engine.generate(req.prompt, **params)
        perf_monitor.record_from_result(result)
        return {
            "text": result.text,
            "tokens_generated": result.tokens_generated,
            "tokens_per_second": round(result.tokens_per_second, 2),
            "finish_reason": result.finish_reason,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Model Management
    # ═══════════════════════════════════════════════════════════════════════

    @app.get("/models")
    async def list_models(
        query: Optional[str] = None,
        tags: Optional[str] = None,
        family: Optional[str] = None,
        min_params: Optional[float] = None,
        max_params: Optional[float] = None,
        fits_vram: Optional[bool] = None,
        sort_by: str = "name",
        auth=Depends(verify_api_key),
    ):
        """List available models with search and filter support.

        Query parameters:
          query: Text search across name, family, description
          tags: Comma-separated tags to filter by
          family: Model family (llama, mistral, qwen, etc.)
          min_params: Minimum parameter count (billions)
          max_params: Maximum parameter count (billions)
          fits_vram: Only show models that fit in VRAM
          sort_by: Sort field (name, params, size, vram, quality)
        """
        if query or tags or family or min_params or max_params or fits_vram:
            tag_list = tags.split(",") if tags else None
            models = model_manager.search_models(
                query=query or "", tags=tag_list, family=family,
                min_params=min_params or 0,
                max_params=max_params or float("inf"),
                fits_vram_only=fits_vram or False,
                sort_by=sort_by,
            )
        else:
            models = model_manager.list_models()
        return {"models": models}

    @app.get("/models/{name}")
    async def get_model_detail(name: str, auth=Depends(verify_api_key)):
        """Get detailed information about a specific model."""
        detail = model_manager.get_model_detail(name)
        if not detail:
            raise HTTPException(status_code=404, detail=f"Model not found: {name}")
        return detail

    @app.post("/models/load")
    async def load_model(req: LoadModelRequest, auth=Depends(verify_api_key)):
        """Load a model by name with optional GPU layer override."""
        try:
            entry = model_manager.load_model(req.name, n_gpu_layers=req.n_gpu_layers)
            return {
                "status": "loaded",
                "model": entry.name,
                "path": entry.path,
                "params_b": entry.n_params_b,
                "quantization": entry.quantization,
                "recommended_gpu_layers": entry.recommended_gpu_layers,
                "estimated_vram_mb": round(entry.estimated_vram_optimal_mb, 0),
                "auto_config": entry.auto_config,
            }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model not found: {req.name}")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/models/unload")
    async def unload_model(auth=Depends(verify_api_key)):
        """Unload the current model."""
        model_manager.unload_model()
        return {"status": "unloaded"}

    @app.post("/models/validate/{name}")
    async def validate_model(name: str, req: ValidateRequest, auth=Depends(verify_api_key)):
        """Validate model integrity. Use deep=true for probe load test (slow)."""
        result = model_manager.validate_model(name, deep=req.deep)
        return result

    @app.post("/models/hash/{name}")
    async def compute_hash(name: str, auth=Depends(verify_api_key)):
        """Compute full SHA256 hash for a model (slow for large files)."""
        try:
            sha = model_manager.compute_full_hash(name)
            return {"model": name, "sha256": sha}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/models/alias")
    async def manage_alias(req: AliasRequest, auth=Depends(verify_api_key)):
        """Add or remove a model alias."""
        if req.action == "add":
            model_manager.set_alias(req.model_name, req.alias)
            return {"status": "alias_added", "model": req.model_name, "alias": req.alias}
        elif req.action == "remove":
            model_manager.remove_alias(req.model_name, req.alias)
            return {"status": "alias_removed", "model": req.model_name, "alias": req.alias}
        raise HTTPException(status_code=400, detail="action must be 'add' or 'remove'")

    @app.post("/models/tag")
    async def manage_tag(req: TagRequest, auth=Depends(verify_api_key)):
        """Add or remove a model tag."""
        if req.action == "add":
            model_manager.add_tag(req.model_name, req.tag)
            return {"status": "tag_added", "model": req.model_name, "tag": req.tag}
        elif req.action == "remove":
            model_manager.remove_tag(req.model_name, req.tag)
            return {"status": "tag_removed", "model": req.model_name, "tag": req.tag}
        raise HTTPException(status_code=400, detail="action must be 'add' or 'remove'")

    @app.post("/models/compare")
    async def compare_models(req: CompareRequest, auth=Depends(verify_api_key)):
        """Compare multiple models side-by-side."""
        if len(req.models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 model names")
        return model_manager.compare_models(req.models)

    @app.get("/models/recommendations")
    async def get_recommendations(
        use_case: str = "general",
        auth=Depends(verify_api_key),
    ):
        """Get ranked model recommendations for a use case."""
        recs = model_manager.get_recommendations(use_case)
        return {
            "use_case": use_case,
            "recommendations": [
                {
                    "model_name": r.model_name,
                    "reason": r.reason,
                    "confidence": round(r.confidence, 2),
                    "suggested_n_ctx": r.suggested_n_ctx,
                    "suggested_n_batch": r.suggested_n_batch,
                    "suggested_cache_k": r.suggested_cache_k,
                    "suggested_cache_v": r.suggested_cache_v,
                    "suggested_n_gpu_layers": r.suggested_n_gpu_layers,
                    "estimated_vram_mb": round(r.estimated_vram_mb, 0),
                }
                for r in recs
            ],
        }

    # ── Downloads ────────────────────────────────────────────────────────

    @app.post("/models/download")
    async def download_model(req: DownloadRequest, auth=Depends(verify_api_key)):
        """Submit a model download task."""
        task = model_manager.download_model(
            url=req.url,
            name=req.filename,
            priority=req.priority or 2,
            expected_sha256=req.expected_sha256 or "",
        )
        return {
            "task_id": task.id,
            "status": task.status.value,
            "filename": task.filename,
            "url": req.url,
        }

    @app.get("/models/downloads")
    async def list_downloads(auth=Depends(verify_api_key)):
        """List all active and completed downloads."""
        return model_manager.list_downloads()

    @app.get("/models/downloads/{task_id}")
    async def get_download_status(task_id: str, auth=Depends(verify_api_key)):
        """Get status of a specific download task."""
        status = model_manager.get_download_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Download task not found")
        return status

    @app.delete("/models/downloads/{task_id}")
    async def cancel_download(task_id: str, auth=Depends(verify_api_key)):
        """Cancel a download task."""
        if model_manager.cancel_download(task_id):
            return {"status": "cancelled", "task_id": task_id}
        raise HTTPException(status_code=404, detail="Download task not found")

    # ── Disk Management ──────────────────────────────────────────────────

    @app.get("/models/disk")
    async def disk_usage(auth=Depends(verify_api_key)):
        """Get disk usage information for model directories."""
        return model_manager.get_disk_usage()

    @app.post("/models/cleanup")
    async def cleanup_downloads(auth=Depends(verify_api_key)):
        """Cleanup partial and temporary download files."""
        cleaned = model_manager.cleanup_partial_downloads()
        return {"cleaned_files": cleaned, "count": len(cleaned)}

    # ═══════════════════════════════════════════════════════════════════════
    # Sessions
    # ═══════════════════════════════════════════════════════════════════════

    @app.post("/sessions")
    async def create_session(req: CreateSessionRequest, auth=Depends(verify_api_key)):
        """Create a new chat session."""
        session = pipeline.create_session(req.system_prompt)
        return {"session_id": session.id}

    @app.get("/sessions")
    async def list_sessions(auth=Depends(verify_api_key)):
        """List active sessions."""
        return {"sessions": pipeline._sessions.list_sessions()}

    # ═══════════════════════════════════════════════════════════════════════
    # Health & Metrics
    # ═══════════════════════════════════════════════════════════════════════

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        engine_loaded = engine.is_loaded
        gpu_info = gpu_manager.get_memory_info()
        pressure = gpu_manager.get_pressure()

        status = "ok" if engine_loaded else "no_model"
        if pressure.value in ("high", "critical"):
            status = "degraded"

        return {
            "status": status,
            "model_loaded": engine_loaded,
            "gpu_pressure": pressure.value,
            "vram_free_mb": round(gpu_info.free_vram_mb, 0),
        }

    @app.get("/metrics")
    async def metrics():
        """Prometheus-compatible metrics endpoint."""
        return PlainTextResponse(perf_monitor.get_prometheus_metrics())

    @app.get("/status")
    async def status(auth=Depends(verify_api_key)):
        """Full system status."""
        engine_tel = engine.get_telemetry()
        pipeline_stat = pipeline.get_status()
        gpu_info = gpu_manager.get_memory_info()
        perf_report = perf_monitor.get_report(gpu_info)
        gpu_stats = gpu_manager.get_stats()
        alerts = gpu_manager.get_alerts(limit=10)
        disk = model_manager.get_disk_usage()

        current_model = model_manager.get_current_model()

        return {
            "engine": engine_tel,
            "pipeline": pipeline_stat,
            "gpu": {
                "name": gpu_info.gpu_name,
                "vram_total_mb": round(gpu_info.total_vram_mb, 0),
                "vram_used_mb": round(gpu_info.used_vram_mb, 0),
                "vram_free_mb": round(gpu_info.free_vram_mb, 0),
                "utilization_pct": gpu_info.gpu_utilization_pct,
                "temperature_c": gpu_info.gpu_temperature_c,
                "power_w": gpu_info.gpu_power_usage_w,
                "pressure": gpu_stats.get("current_pressure"),
                "strategy": gpu_stats.get("strategy"),
            },
            "performance": {
                "avg_tps": round(perf_report.avg_tokens_per_second, 2),
                "p50_latency_ms": round(perf_report.latency_gen_ms.median, 1),
                "p99_latency_ms": round(perf_report.latency_gen_ms.p99, 1),
                "total_generations": perf_report.total_generations,
                "total_tokens": perf_report.total_tokens_generated,
                "health": perf_report.health_status,
                "warnings": perf_report.warnings,
            },
            "current_model": {
                "name": current_model.name if current_model else None,
                "params_b": current_model.n_params_b if current_model else 0,
                "quantization": current_model.quantization if current_model else None,
                "avg_tps": round(current_model.avg_tps, 1) if current_model and current_model.avg_tps > 0 else None,
                "load_count": current_model.load_count if current_model else 0,
            } if current_model else None,
            "disk": disk,
            "alerts": [
                {"level": a.level.value, "message": a.message, "time": a.timestamp}
                for a in alerts
            ],
        }

    @app.get("/gpu")
    async def gpu_info(auth=Depends(verify_api_key)):
        """Detailed GPU memory information."""
        info = gpu_manager.get_memory_info()
        stats = gpu_manager.get_stats()
        return {
            "gpu_name": info.gpu_name,
            "vram": {
                "total_mb": round(info.total_vram_mb, 0),
                "used_mb": round(info.used_vram_mb, 0),
                "free_mb": round(info.free_vram_mb, 0),
                "utilization_pct": round(
                    (info.used_vram_mb / max(info.total_vram_mb, 1)) * 100, 1
                ),
            },
            "ram": {
                "total_mb": round(info.total_ram_mb, 0),
                "used_mb": round(info.used_ram_mb, 0),
                "free_mb": round(info.free_ram_mb, 0),
            },
            "gpu_utilization_pct": info.gpu_utilization_pct,
            "temperature_c": info.gpu_temperature_c,
            "power_w": info.gpu_power_usage_w,
            "pressure": stats.get("current_pressure"),
            "monitor_cycles": stats.get("total_monitor_cycles", 0),
            "layer_adjustments": stats.get("layer_adjustments", 0),
        }

    # ── Benchmark ────────────────────────────────────────────────────────

    @app.get("/benchmark")
    async def benchmark(auth=Depends(verify_api_key)):
        """Run a quick benchmark."""
        if not engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")
        result = engine.benchmark()
        perf_monitor.set_baseline(result.get("tokens_per_second", 0))

        # Record model performance
        current_model = model_manager.get_current_model()
        if current_model:
            model_manager.record_performance(
                current_model.name,
                result.get("tokens_per_second", 0),
            )

        return result

    return app


# ==============================================================================
# SECTION: gui.py
# ==============================================================================

"""
Web GUI — Browser-based interface for model downloading and management.

Serves a modern single-page application at /gui/ that provides:
  - Dashboard with GPU status, system info, and performance metrics
  - Model browser with search, filter, sort, and VRAM fit analysis
  - HuggingFace GGUF model search and one-click download
  - Download queue with real-time progress bars
  - Model management: load, unload, validate, compare, alias, tag
  - Performance monitoring with live throughput graphs
  - Chat playground for testing loaded models
  - Settings panel for configuration
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


# ─── GUI HTML (Self-Contained SPA) ──────────────────────────────────────────

_GUI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llama-cpp-smart-gpu — Model Manager</title>
<style>
/* ── Reset & Variables ──────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-primary:#0f1117;--bg-secondary:#1a1d27;--bg-tertiary:#242837;
  --bg-hover:#2d3148;--bg-card:#1e2230;--bg-input:#161924;
  --text-primary:#e4e6f0;--text-secondary:#8b8fa8;--text-muted:#5c6080;
  --accent:#6c5ce7;--accent-light:#a29bfe;--accent-bg:rgba(108,92,231,0.12);
  --success:#00b894;--success-bg:rgba(0,184,148,0.12);
  --warning:#fdcb6e;--warning-bg:rgba(253,203,110,0.12);
  --danger:#e17055;--danger-bg:rgba(225,112,85,0.12);
  --info:#74b9ff;--info-bg:rgba(116,185,255,0.12);
  --border:#2d3148;--radius:8px;--radius-lg:12px;
  --shadow:0 2px 12px rgba(0,0,0,0.3);
  --font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono','Fira Code','Cascadia Code',monospace;
}
html,body{height:100%;background:var(--bg-primary);color:var(--text-primary);
  font-family:var(--font);font-size:14px;line-height:1.6;overflow:hidden}

/* ── Layout ─────────────────────────────────────────────────────── */
#app{display:flex;height:100vh;width:100vw}
#sidebar{width:220px;min-width:220px;background:var(--bg-secondary);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  padding:0;z-index:10}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}

/* ── Sidebar ────────────────────────────────────────────────────── */
.sidebar-logo{padding:20px 16px 12px;border-bottom:1px solid var(--border)}
.sidebar-logo h1{font-size:16px;font-weight:700;color:var(--accent-light);letter-spacing:-0.3px}
.sidebar-logo span{font-size:11px;color:var(--text-muted);display:block;margin-top:2px}
.sidebar-nav{flex:1;padding:8px 0;overflow-y:auto}
.nav-item{display:flex;align-items:center;gap:10px;padding:10px 16px;
  cursor:pointer;color:var(--text-secondary);transition:all 0.15s;
  border-left:3px solid transparent;font-size:13px;font-weight:500}
.nav-item:hover{background:var(--bg-hover);color:var(--text-primary)}
.nav-item.active{background:var(--accent-bg);color:var(--accent-light);
  border-left-color:var(--accent)}
.nav-item svg{width:18px;height:18px;flex-shrink:0}
.sidebar-footer{padding:12px 16px;border-top:1px solid var(--border);
  font-size:11px;color:var(--text-muted)}
.sidebar-footer .status-dot{width:8px;height:8px;border-radius:50%;
  display:inline-block;margin-right:6px;vertical-align:middle}
.sidebar-footer .status-dot.ok{background:var(--success)}
.sidebar-footer .status-dot.warn{background:var(--warning)}
.sidebar-footer .status-dot.err{background:var(--danger)}

/* ── Topbar ─────────────────────────────────────────────────────── */
#topbar{height:52px;min-height:52px;background:var(--bg-secondary);
  border-bottom:1px solid var(--border);display:flex;align-items:center;
  justify-content:space-between;padding:0 24px}
#topbar h2{font-size:15px;font-weight:600;color:var(--text-primary)}
.topbar-actions{display:flex;gap:8px;align-items:center}

/* ── Content ────────────────────────────────────────────────────── */
#content{flex:1;overflow-y:auto;padding:24px}

/* ── Cards ──────────────────────────────────────────────────────── */
.card{background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:20px;margin-bottom:16px;
  transition:border-color 0.2s}
.card:hover{border-color:var(--accent)}
.card-header{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:16px}
.card-title{font-size:14px;font-weight:600;color:var(--text-primary)}
.card-subtitle{font-size:12px;color:var(--text-muted);margin-top:2px}

/* ── Stat Grid ──────────────────────────────────────────────────── */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.stat-card{background:var(--bg-tertiary);border-radius:var(--radius);padding:16px;
  border:1px solid var(--border)}
.stat-label{font-size:11px;text-transform:uppercase;letter-spacing:0.5px;
  color:var(--text-muted);margin-bottom:4px}
.stat-value{font-size:22px;font-weight:700;font-family:var(--mono);
  color:var(--text-primary)}
.stat-detail{font-size:11px;color:var(--text-secondary);margin-top:4px}

/* ── Buttons ────────────────────────────────────────────────────── */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
  border-radius:var(--radius);border:1px solid var(--border);
  background:var(--bg-tertiary);color:var(--text-primary);cursor:pointer;
  font-size:12px;font-weight:500;transition:all 0.15s;font-family:var(--font)}
.btn:hover{background:var(--bg-hover);border-color:var(--accent)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn-primary:hover{background:var(--accent-light);border-color:var(--accent-light)}
.btn-success{background:var(--success);border-color:var(--success);color:#fff}
.btn-danger{background:var(--danger);border-color:var(--danger);color:#fff}
.btn-sm{padding:5px 10px;font-size:11px}
.btn:disabled{opacity:0.4;cursor:not-allowed}

/* ── Inputs ─────────────────────────────────────────────────────── */
.input,.select{background:var(--bg-input);border:1px solid var(--border);
  border-radius:var(--radius);padding:8px 12px;color:var(--text-primary);
  font-size:13px;font-family:var(--font);width:100%;transition:border-color 0.2s}
.input:focus,.select:focus{outline:none;border-color:var(--accent)}
.input::placeholder{color:var(--text-muted)}
.select{appearance:none;cursor:pointer}
.search-box{position:relative}
.search-box .input{padding-left:36px}
.search-box svg{position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:16px;height:16px;color:var(--text-muted)}

/* ── Progress Bar ───────────────────────────────────────────────── */
.progress-bar{height:6px;background:var(--bg-input);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;border-radius:3px;transition:width 0.3s ease;
  background:linear-gradient(90deg,var(--accent),var(--accent-light))}
.progress-fill.success{background:var(--success)}
.progress-fill.danger{background:var(--danger)}

/* ── Table ──────────────────────────────────────────────────────── */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:10px 12px;font-size:11px;text-transform:uppercase;
  letter-spacing:0.5px;color:var(--text-muted);border-bottom:1px solid var(--border);
  font-weight:600;white-space:nowrap}
td{padding:10px 12px;border-bottom:1px solid var(--border);font-size:13px;
  color:var(--text-secondary);white-space:nowrap}
tr:hover td{background:var(--bg-hover)}
tr.active td{background:var(--accent-bg)}

/* ── Badge ──────────────────────────────────────────────────────── */
.badge{display:inline-flex;align-items:center;padding:3px 8px;
  border-radius:4px;font-size:11px;font-weight:600}
.badge-success{background:var(--success-bg);color:var(--success)}
.badge-warning{background:var(--warning-bg);color:var(--warning)}
.badge-danger{background:var(--danger-bg);color:var(--danger)}
.badge-info{background:var(--info-bg);color:var(--info)}
.badge-accent{background:var(--accent-bg);color:var(--accent-light)}

/* ── Tags ───────────────────────────────────────────────────────── */
.tag{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;
  border-radius:4px;font-size:10px;font-weight:600;
  background:var(--bg-tertiary);border:1px solid var(--border);
  color:var(--text-secondary);margin:2px}
.tag-remove{cursor:pointer;color:var(--text-muted);font-size:12px}
.tag-remove:hover{color:var(--danger)}

/* ── Download Item ──────────────────────────────────────────────── */
.download-item{background:var(--bg-tertiary);border:1px solid var(--border);
  border-radius:var(--radius);padding:14px 16px;margin-bottom:10px}
.download-header{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px}
.download-name{font-size:13px;font-weight:600;color:var(--text-primary)}
.download-status{font-size:11px}
.download-stats{display:flex;gap:16px;font-size:11px;color:var(--text-muted);margin-top:6px}

/* ── Model Card Grid ────────────────────────────────────────────── */
.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.model-card{background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:16px;cursor:pointer;
  transition:all 0.2s}
.model-card:hover{border-color:var(--accent);transform:translateY(-1px);
  box-shadow:var(--shadow)}
.model-card.loaded{border-color:var(--success)}
.model-card-name{font-size:14px;font-weight:600;color:var(--text-primary);
  margin-bottom:6px;word-break:break-all}
.model-card-meta{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.model-card-detail{font-size:11px;color:var(--text-muted);margin-top:6px;
  display:flex;gap:16px}
.model-card-actions{display:flex;gap:6px;margin-top:10px}

/* ── Chat ───────────────────────────────────────────────────────── */
.chat-container{display:flex;flex-direction:column;height:calc(100vh - 140px)}
.chat-messages{flex:1;overflow-y:auto;padding:16px 0}
.chat-msg{margin-bottom:12px;max-width:80%}
.chat-msg.user{margin-left:auto}
.chat-msg-assistant{background:var(--bg-tertiary);border:1px solid var(--border);
  border-radius:var(--radius);padding:10px 14px;font-size:13px;color:var(--text-primary)}
.chat-msg-user{background:var(--accent-bg);border:1px solid var(--accent);
  border-radius:var(--radius);padding:10px 14px;font-size:13px;color:var(--text-primary)}
.chat-msg-label{font-size:10px;color:var(--text-muted);margin-bottom:4px;text-transform:uppercase}
.chat-input-bar{display:flex;gap:8px;padding-top:12px;border-top:1px solid var(--border)}
.chat-input-bar .input{flex:1}
.chat-input-bar .btn{white-space:nowrap}

/* ── Tabs ───────────────────────────────────────────────────────── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:16px}
.tab{padding:10px 20px;font-size:13px;font-weight:500;cursor:pointer;
  color:var(--text-secondary);border-bottom:2px solid transparent;
  transition:all 0.15s}
.tab:hover{color:var(--text-primary)}
.tab.active{color:var(--accent-light);border-bottom-color:var(--accent)}

/* ── Modal ──────────────────────────────────────────────────────── */
.modal-overlay{position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,0.6);display:flex;align-items:center;
  justify-content:center;z-index:100}
.modal{background:var(--bg-secondary);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:24px;width:90%;max-width:600px;
  max-height:80vh;overflow-y:auto}
.modal-title{font-size:16px;font-weight:700;margin-bottom:16px;color:var(--text-primary)}
.modal-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:20px}

/* ── Toast ──────────────────────────────────────────────────────── */
#toast-container{position:fixed;top:16px;right:16px;z-index:200;display:flex;
  flex-direction:column;gap:8px}
.toast{padding:12px 16px;border-radius:var(--radius);font-size:12px;font-weight:500;
  color:#fff;min-width:280px;box-shadow:var(--shadow);
  animation:slideIn 0.2s ease}
.toast-success{background:var(--success)}
.toast-error{background:var(--danger)}
.toast-info{background:var(--accent)}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}

/* ── VRAM Gauge ─────────────────────────────────────────────────── */
.vram-gauge{position:relative;height:24px;background:var(--bg-input);
  border-radius:12px;overflow:hidden;margin:8px 0}
.vram-gauge-fill{height:100%;border-radius:12px;transition:width 0.5s ease}
.vram-gauge-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:11px;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.5)}

/* ── Responsive ─────────────────────────────────────────────────── */
@media(max-width:768px){
  #sidebar{width:56px;min-width:56px}
  .sidebar-logo h1,.nav-item span,.sidebar-footer span{display:none}
  .nav-item{justify-content:center;padding:12px}
  .model-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div id="app">
  <!-- Sidebar -->
  <div id="sidebar">
    <div class="sidebar-logo">
      <h1>llama-smart-gpu</h1>
      <span>v2.1.0</span>
    </div>
    <div class="sidebar-nav">
      <div class="nav-item active" data-page="dashboard" onclick="navigate('dashboard')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
        <span>Dashboard</span>
      </div>
      <div class="nav-item" data-page="models" onclick="navigate('models')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg>
        <span>Models</span>
      </div>
      <div class="nav-item" data-page="download" onclick="navigate('download')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        <span>Download</span>
      </div>
      <div class="nav-item" data-page="chat" onclick="navigate('chat')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        <span>Chat</span>
      </div>
      <div class="nav-item" data-page="settings" onclick="navigate('settings')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
        <span>Settings</span>
      </div>
    </div>
    <div class="sidebar-footer">
      <span class="status-dot ok" id="conn-dot"></span>
      <span id="conn-status">Connected</span>
    </div>
  </div>

  <!-- Main -->
  <div id="main">
    <div id="topbar">
      <h2 id="page-title">Dashboard</h2>
      <div class="topbar-actions">
        <span id="model-badge" class="badge badge-warning" style="display:none">No model loaded</span>
        <button class="btn btn-sm" onclick="refreshPage()">Refresh</button>
      </div>
    </div>
    <div id="content"></div>
  </div>
</div>

<div id="toast-container"></div>

<script>
/* ═══════════════════════════════════════════════════════════════════
   STATE & CONFIG
   ═══════════════════════════════════════════════════════════════════ */
const API = '';
let currentPage = 'dashboard';
let refreshTimer = null;
let state = {
  models: [],
  downloads: [],
  gpu: {},
  status: {},
  performance: {},
  loadedModel: null,
  chatMessages: [],
};

/* ═══════════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════════ */
function fmt(n, d=1) {
  if (n == null || isNaN(n)) return '—';
  if (n >= 1e9) return (n/1e9).toFixed(d) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(d) + 'M';
  if (n >= 1024) return (n/1024).toFixed(d) + 'K';
  return n.toFixed(d);
}
function fmtBytes(mb) {
  if (mb == null || isNaN(mb)) return '—';
  if (mb >= 1024) return (mb/1024).toFixed(1) + ' GB';
  return mb.toFixed(0) + ' MB';
}
function fmtPct(v) { return v != null ? v.toFixed(1) + '%' : '—'; }
function escHtml(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function badgeClass(t) {
  const m = {highest:'badge-success',high:'badge-success',medium:'badge-info',low:'badge-warning',low2:'badge-danger'};
  return m[t] || 'badge-accent';
}
function fitBadge(score) {
  if (score >= 0.7) return '<span class="badge badge-success">Fits</span>';
  if (score >= 0.3) return '<span class="badge badge-warning">Tight</span>';
  return '<span class="badge badge-danger">Over</span>';
}

async function api(path, opts={}) {
  try {
    const r = await fetch(API + path, {
      headers: {'Content-Type':'application/json', ...opts.headers},
      ...opts
    });
    if (!r.ok) { const e = await r.json().catch(()=>({detail:r.statusText})); throw new Error(e.detail||r.statusText); }
    return await r.json();
  } catch(e) {
    toast(e.message, 'error');
    throw e;
  }
}

function toast(msg, type='info') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

/* ═══════════════════════════════════════════════════════════════════
   NAVIGATION
   ═══════════════════════════════════════════════════════════════════ */
function navigate(page) {
  currentPage = page;
  document.querySelectorAll('.nav-item').forEach(n => {
    n.classList.toggle('active', n.dataset.page === page);
  });
  const titles = {dashboard:'Dashboard',models:'Model Library',download:'Download Models',chat:'Chat Playground',settings:'Settings'};
  document.getElementById('page-title').textContent = titles[page] || page;
  renderPage();
}

function refreshPage() { renderPage(); }

async function renderPage() {
  const el = document.getElementById('content');
  try {
    switch(currentPage) {
      case 'dashboard': await renderDashboard(el); break;
      case 'models':    await renderModels(el); break;
      case 'download':  await renderDownload(el); break;
      case 'chat':      renderChat(el); break;
      case 'settings':  await renderSettings(el); break;
    }
  } catch(e) { el.innerHTML = '<div class="card"><p style="color:var(--danger)">Error: '+escHtml(e.message)+'</p></div>'; }
}

/* ═══════════════════════════════════════════════════════════════════
   DASHBOARD
   ═══════════════════════════════════════════════════════════════════ */
async function renderDashboard(el) {
  const [gpuData, statusData, healthData] = await Promise.all([
    api('/gpu').catch(()=>({})),
    api('/status').catch(()=>({})),
    api('/health').catch(()=>({}))
  ]);
  state.gpu = gpuData; state.status = statusData;

  // /gpu returns: {gpu_name, vram:{total_mb,used_mb,free_mb,...}, ram:{...}, gpu_utilization_pct, temperature_c, power_w, pressure}
  const vram = gpuData.vram || {};
  const ram = gpuData.ram || {};
  const vramPct = (vram.total_mb > 0) ? (vram.used_mb / vram.total_mb * 100) : 0;
  const vramColor = vramPct > 90 ? 'var(--danger)' : vramPct > 75 ? 'var(--warning)' : 'var(--success)';

  // /status returns: {engine:{...}, gpu:{...}, performance:{...}, current_model:{...}, ...}
  const eng = statusData.engine || {};
  const perf = statusData.performance || {};
  const gpuS = statusData.gpu || {};
  const curModel = statusData.current_model;
  const loadedModel = curModel?.name || eng.model_path || null;
  state.loadedModel = loadedModel;

  // Update model badge
  const badge = document.getElementById('model-badge');
  if (loadedModel) {
    badge.style.display = '';
    badge.className = 'badge badge-success';
    badge.textContent = loadedModel.split('/').pop();
  } else {
    badge.style.display = '';
    badge.className = 'badge badge-warning';
    badge.textContent = 'No model loaded';
  }

  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">GPU</div>
        <div class="stat-value" style="font-size:14px">${escHtml(gpuData.gpu_name || 'N/A')}</div>
        <div class="stat-detail">Utilization: ${fmtPct(gpuData.gpu_utilization_pct)} | Temp: ${gpuData.temperature_c?.toFixed(0) || '—'}°C</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">VRAM</div>
        <div class="stat-value">${fmtBytes(vram.used_mb)}</div>
        <div class="stat-detail">of ${fmtBytes(vram.total_mb)} | Free: ${fmtBytes(vram.free_mb)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">RAM</div>
        <div class="stat-value">${fmtBytes(ram.used_mb)}</div>
        <div class="stat-detail">of ${fmtBytes(ram.total_mb)} | Free: ${fmtBytes(ram.free_mb)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Throughput</div>
        <div class="stat-value">${fmt(perf.avg_tps || eng.avg_tps || 0)} <small style="font-size:12px">tok/s</small></div>
        <div class="stat-detail">Generations: ${perf.total_generations || eng.total_generations || 0}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div><div class="card-title">VRAM Usage</div></div>
        <span class="badge ${vramPct > 90 ? 'badge-danger' : vramPct > 75 ? 'badge-warning' : 'badge-success'}">${fmtPct(vramPct)}</span>
      </div>
      <div class="vram-gauge">
        <div class="vram-gauge-fill" style="width:${Math.min(vramPct,100)}%;background:${vramColor}"></div>
        <div class="vram-gauge-text">${fmtBytes(vram.used_mb)} / ${fmtBytes(vram.total_mb)}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><div class="card-title">System Status</div></div>
      <div class="table-wrap"><table>
        <tr><td style="color:var(--text-primary)">Health</td><td>${(() => { const h=healthData.status||''; const c={ok:'badge-success',no_model:'badge-warning',degraded:'badge-danger'}; return '<span class="badge '+(c[h]||'badge-accent')+'">'+escHtml(h||'unknown')+'</span>'; })()}</td></tr>
        <tr><td style="color:var(--text-primary)">Current Model</td><td>${loadedModel ? escHtml(loadedModel.split('/').pop()) : '<span style="color:var(--text-muted)">None</span>'}</td></tr>
        <tr><td style="color:var(--text-primary)">GPU Pressure</td><td>${(() => { const p=gpuData.pressure||gpuS.pressure||''; const c={low:'badge-success',moderate:'badge-warning',high:'badge-danger',critical:'badge-danger'}; return '<span class="badge '+(c[p]||'badge-accent')+'">'+escHtml(p||'unknown')+'</span>'; })()}</td></tr>
        <tr><td style="color:var(--text-primary)">GPU Layers</td><td>${eng.gpu_layers || gpuS.strategy || '—'}</td></tr>
        <tr><td style="color:var(--text-primary)">Power</td><td>${gpuData.power_w?.toFixed(1) || '—'} W</td></tr>
        <tr><td style="color:var(--text-primary)">Total Tokens</td><td>${fmt(perf.total_tokens || eng.total_tokens_generated || 0, 0)}</td></tr>
        <tr><td style="color:var(--text-primary)">P99 Latency</td><td>${perf.p99_latency_ms?.toFixed(0) || '—'} ms</td></tr>
        <tr><td style="color:var(--text-primary)">Layer Adjustments</td><td>${gpuData.layer_adjustments || 0}</td></tr>
        ${perf.warnings && perf.warnings.length ? perf.warnings.map(w=>'<tr><td style="color:var(--danger)">Warning</td><td style="color:var(--danger)">'+escHtml(w)+'</td></tr>').join('') : ''}
      </table></div>
    </div>
  `;
}

/* ═══════════════════════════════════════════════════════════════════
   MODELS PAGE
   ═══════════════════════════════════════════════════════════════════ */
async function renderModels(el) {
  const models = await api('/models');
  state.models = models.models || models || [];
  const list = state.models;

  el.innerHTML = `
    <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;align-items:center">
      <div class="search-box" style="flex:1;min-width:200px">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input class="input" id="model-search" placeholder="Search models..." oninput="filterModels()">
      </div>
      <select class="select" id="model-family-filter" style="width:140px" onchange="filterModels()">
        <option value="">All Families</option>
        <option value="llama">LLaMA</option><option value="mistral">Mistral</option>
        <option value="qwen">Qwen</option><option value="gemma">Gemma</option>
        <option value="phi">Phi</option><option value="deepseek">DeepSeek</option>
      </select>
      <select class="select" id="model-sort" style="width:140px" onchange="filterModels()">
        <option value="name">Sort: Name</option>
        <option value="size_desc">Size (Large first)</option>
        <option value="size_asc">Size (Small first)</option>
        <option value="vram_asc">VRAM Fit (Best first)</option>
      </select>
      <button class="btn" onclick="renderModels(document.getElementById('content'))">Refresh</button>
    </div>
    <div class="model-grid" id="model-grid"></div>
  `;
  filterModels();
}

function filterModels() {
  const q = (document.getElementById('model-search')?.value || '').toLowerCase();
  const family = document.getElementById('model-family-filter')?.value || '';
  const sort = document.getElementById('model-sort')?.value || 'name';
  let list = state.models.filter(m => {
    if (q && !m.name.toLowerCase().includes(q) && !(m.aliases||[]).some(a=>a.toLowerCase().includes(q))) return false;
    if (family && m.family !== family) return false;
    return true;
  });
  list.sort((a,b) => {
    if (sort==='size_desc') return (b.size_mb||0)-(a.size_mb||0);
    if (sort==='size_asc') return (a.size_mb||0)-(b.size_mb||0);
    if (sort==='vram_asc') return (b.vram_fit_score||0)-(a.vram_fit_score||0);
    return a.name.localeCompare(b.name);
  });
  const grid = document.getElementById('model-grid');
  if (!grid) return;
  if (!list.length) { grid.innerHTML = '<div class="card" style="text-align:center;padding:40px;color:var(--text-muted)">No models found. Download one from the Download tab.</div>'; return; }
  grid.innerHTML = list.map(m => {
    const isLoaded = state.loadedModel && (m.name === state.loadedModel || m.path === state.loadedModel);
    return `
    <div class="model-card ${isLoaded ? 'loaded' : ''}" onclick="showModelDetail('${escHtml(m.name)}')">
      <div class="model-card-name">${escHtml(m.name)} ${isLoaded ? '<span class="badge badge-success" style="margin-left:6px">LOADED</span>' : ''}</div>
      <div class="model-card-meta">
        <span class="badge badge-accent">${escHtml(m.family||'unknown')}</span>
        ${m.quantization ? '<span class="badge '+badgeClass(m.quant_info?.quality_tier||'')+'">'+escHtml(m.quantization)+'</span>' : ''}
        ${fitBadge(m.vram_fit_score||0)}
      </div>
      <div class="model-card-detail">
        <span>${fmt(m.n_params_b,1)}B params</span>
        <span>${fmtBytes(m.size_mb)}</span>
        <span>VRAM: ${fmtBytes(m.estimated_vram_full_mb)}</span>
      </div>
      ${(m.tags||[]).length ? '<div style="margin-top:6px">'+m.tags.map(t=>'<span class="tag">'+escHtml(t)+'</span>').join('')+'</div>' : ''}
      <div class="model-card-actions" onclick="event.stopPropagation()">
        ${isLoaded ? '<button class="btn btn-sm btn-danger" onclick="unloadModel()">Unload</button>'
                   : '<button class="btn btn-sm btn-primary" onclick="loadModel(\''+escHtml(m.name)+'\')">Load</button>'}
        <button class="btn btn-sm" onclick="validateModel(\''+escHtml(m.name)+'\')">Validate</button>
      </div>
    </div>`;
  }).join('');
}

async function showModelDetail(name) {
  try {
    const m = await api('/models/' + encodeURIComponent(name));
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.onclick = e => { if(e.target===modal) modal.remove(); };
    modal.innerHTML = `<div class="modal">
      <div class="modal-title">${escHtml(m.name)}</div>
      <div class="table-wrap"><table>
        <tr><td style="color:var(--text-primary);width:140px">Family</td><td>${escHtml(m.family||'unknown')}</td></tr>
        <tr><td style="color:var(--text-primary)">Parameters</td><td>${fmt(m.n_params_b,2)}B</td></tr>
        <tr><td style="color:var(--text-primary)">Quantization</td><td>${escHtml(m.quantization||'unknown')} ${m.quant_info?.quality_tier ? '('+m.quant_info.quality_tier+')' : ''}</td></tr>
        <tr><td style="color:var(--text-primary)">File Size</td><td>${fmtBytes(m.size_mb)}</td></tr>
        <tr><td style="color:var(--text-primary)">VRAM Full</td><td>${fmtBytes(m.estimated_vram_full_mb)}</td></tr>
        <tr><td style="color:var(--text-primary)">VRAM Optimal</td><td>${fmtBytes(m.estimated_vram_optimal_mb)}</td></tr>
        <tr><td style="color:var(--text-primary)">VRAM Fit</td><td>${fitBadge(m.vram_fit_score||0)} (${fmtPct((m.vram_fit_score||0)*100)})</td></tr>
        <tr><td style="color:var(--text-primary)">GPU Layers</td><td>${m.recommended_gpu_layers ?? 'auto'}</td></tr>
        <tr><td style="color:var(--text-primary)">Context</td><td>${m.context_length || '—'}</td></tr>
        <tr><td style="color:var(--text-primary)">Architecture</td><td>${escHtml(m.architecture||'llama')}</td></tr>
        <tr><td style="color:var(--text-primary)">Status</td><td>${escHtml(m.status||'unknown')}</td></tr>
        <tr><td style="color:var(--text-primary)">Path</td><td style="word-break:break-all">${escHtml(m.path||'')}</td></tr>
        <tr><td style="color:var(--text-primary)">Avg TPS</td><td>${m.avg_tps?.toFixed(1) || '—'}</td></tr>
        <tr><td style="color:var(--text-primary)">Load Count</td><td>${m.load_count || 0}</td></tr>
      </table></div>
      ${(m.auto_config && Object.keys(m.auto_config).length) ? '<div style="margin-top:12px"><div style="font-size:12px;font-weight:600;margin-bottom:6px">Auto Config</div><pre style="background:var(--bg-input);padding:10px;border-radius:var(--radius);font-size:11px;overflow-x:auto;color:var(--accent-light)">'+escHtml(JSON.stringify(m.auto_config,null,2))+'</pre></div>' : ''}
      <div class="modal-actions">
        <button class="btn" onclick="this.closest('.modal-overlay').remove()">Close</button>
        <button class="btn btn-primary" onclick="loadModel('${escHtml(m.name)}');this.closest('.modal-overlay').remove()">Load Model</button>
      </div>
    </div>`;
    document.body.appendChild(modal);
  } catch(e) { toast('Failed to load model detail: ' + e.message, 'error'); }
}

async function loadModel(name) {
  toast('Loading model: ' + name, 'info');
  try {
    await api('/models/load', {method:'POST', body:JSON.stringify({name})});
    state.loadedModel = name;
    toast('Model loaded: ' + name, 'success');
    refreshPage();
  } catch(e) { /* toast already shown by api() */ }
}

async function unloadModel() {
  try {
    await api('/models/unload', {method:'POST'});
    state.loadedModel = null;
    toast('Model unloaded', 'success');
    refreshPage();
  } catch(e) {}
}

async function validateModel(name) {
  toast('Validating: ' + name, 'info');
  try {
    const r = await api('/models/validate/' + encodeURIComponent(name), {method:'POST'});
    if (r.quick_check?.passed) toast('Validation PASSED: ' + name, 'success');
    else toast('Validation FAILED: ' + (r.quick_check?.errors?.join(', ') || 'unknown'), 'error');
  } catch(e) {}
}

/* ═══════════════════════════════════════════════════════════════════
   DOWNLOAD PAGE
   ═══════════════════════════════════════════════════════════════════ */
async function renderDownload(el) {
  const downloads = await api('/models/downloads').catch(()=>[]);
  state.downloads = downloads;

  el.innerHTML = `
    <div class="card">
      <div class="card-header"><div class="card-title">Download GGUF Model</div></div>
      <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px">
        Paste a direct URL to a .gguf file from HuggingFace or any other source.
        The download will be saved to your model directory.
      </p>
      <div style="display:flex;gap:8px">
        <input class="input" id="dl-url" placeholder="https://huggingface.co/.../model.gguf" style="flex:1">
        <input class="input" id="dl-name" placeholder="Filename (optional)" style="width:200px">
        <button class="btn btn-primary" onclick="startDownload()">Download</button>
      </div>

      <div style="margin-top:20px">
        <div style="font-size:12px;font-weight:600;margin-bottom:8px;color:var(--text-secondary)">Popular GGUF Repositories</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">
          <a href="https://huggingface.co/TheBloke?search=gguf" target="_blank" class="btn btn-sm">TheBloke (HF)</a>
          <a href="https://huggingface.co/models?search=gguf+llama" target="_blank" class="btn btn-sm">LLaMA GGUF</a>
          <a href="https://huggingface.co/models?search=gguf+mistral" target="_blank" class="btn btn-sm">Mistral GGUF</a>
          <a href="https://huggingface.co/models?search=gguf+qwen" target="_blank" class="btn btn-sm">Qwen GGUF</a>
          <a href="https://huggingface.co/models?search=gguf+phi" target="_blank" class="btn btn-sm">Phi GGUF</a>
          <a href="https://huggingface.co/models?search=gguf+deepseek" target="_blank" class="btn btn-sm">DeepSeek GGUF</a>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Active Downloads</div>
        <button class="btn btn-sm" onclick="renderDownload(document.getElementById('content'))">Refresh</button>
      </div>
      <div id="download-list">
        ${downloads.length === 0 ? '<p style="color:var(--text-muted);text-align:center;padding:20px">No active downloads</p>' : ''}
        ${downloads.map(dl => renderDownloadItem(dl)).join('')}
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Quick-Add Popular Models</div>
      </div>
      <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px">
        One-click download popular quantized models. Make sure your VRAM can handle them!
      </p>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px">
        ${quickModels().map(m => `
          <div style="background:var(--bg-tertiary);border:1px solid var(--border);border-radius:var(--radius);padding:12px">
            <div style="font-size:13px;font-weight:600;color:var(--text-primary)">${escHtml(m.name)}</div>
            <div style="font-size:11px;color:var(--text-muted);margin:4px 0">${escHtml(m.desc)}</div>
            <div style="font-size:11px;color:var(--text-secondary)">~${m.size} | VRAM: ${m.vram}</div>
            <button class="btn btn-sm btn-primary" style="margin-top:8px"
              onclick="startDownloadUrl('${escHtml(m.url)}','${escHtml(m.filename)}')">
              Download
            </button>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function quickModels() {
  return [
    {name:'LLaMA 3.1 8B Q4_K_M', desc:'Meta LLaMA 3.1 8B Instruct, Q4_K_M quantization', size:'4.9 GB', vram:'~6 GB', url:'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', filename:'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'},
    {name:'Mistral 7B Q4_K_M', desc:'Mistral 7B Instruct v0.3, Q4_K_M quantization', size:'4.4 GB', vram:'~5.5 GB', url:'https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf', filename:'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'},
    {name:'Qwen2.5 7B Q4_K_M', desc:'Alibaba Qwen2.5 7B Instruct, Q4_K_M quantization', size:'4.7 GB', vram:'~6 GB', url:'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf', filename:'Qwen2.5-7B-Instruct-Q4_K_M.gguf'},
    {name:'Phi-3.5 Mini Q4_K_M', desc:'Microsoft Phi-3.5 Mini Instruct, Q4_K_M quantization', size:'2.4 GB', vram:'~3.5 GB', url:'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf', filename:'Phi-3.5-mini-instruct-Q4_K_M.gguf'},
    {name:'DeepSeek-R1 Distill Qwen 7B Q4_K_M', desc:'DeepSeek R1 distilled to Qwen 7B', size:'4.4 GB', vram:'~5.5 GB', url:'https://huggingface.co/bartowski/deepseek-r1-distill-qwen-7B-GGUF/resolve/main/deepseek-r1-distill-qwen-7B-Q4_K_M.gguf', filename:'deepseek-r1-distill-qwen-7B-Q4_K_M.gguf'},
    {name:'Gemma 2 9B Q4_K_M', desc:'Google Gemma 2 9B Instruct, Q4_K_M quantization', size:'5.5 GB', vram:'~7 GB', url:'https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf', filename:'gemma-2-9b-it-Q4_K_M.gguf'},
  ];
}

function renderDownloadItem(dl) {
  const pct = dl.progress_pct || 0;
  const statusClass = dl.status === 'completed' ? 'success' : dl.status === 'failed' ? 'danger' : '';
  const statusBadge = dl.status === 'completed' ? 'badge-success' : dl.status === 'failed' ? 'badge-danger' : dl.status === 'active' ? 'badge-accent' : 'badge-info';
  return `
    <div class="download-item">
      <div class="download-header">
        <div class="download-name">${escHtml(dl.filename || dl.url?.split('/').pop() || 'Unknown')}</div>
        <span class="badge ${statusBadge} download-status">${escHtml(dl.status)}</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill ${statusClass}" style="width:${pct}%"></div>
      </div>
      <div class="download-stats">
        <span>${pct.toFixed(1)}%</span>
        <span>${fmtBytes(dl.downloaded_bytes/1024)} / ${fmtBytes(dl.total_bytes/1024 || 0)}</span>
        <span>${(dl.speed_mbps||0).toFixed(1)} MB/s</span>
        <span>ETA: ${(dl.eta_seconds||0).toFixed(0)}s</span>
      </div>
      ${dl.status !== 'completed' && dl.status !== 'failed' ? '<button class="btn btn-sm btn-danger" style="margin-top:8px" onclick="cancelDownload(\''+escHtml(dl.id)+'\')">Cancel</button>' : ''}
    </div>`;
}

async function startDownload() {
  const url = document.getElementById('dl-url').value.trim();
  const name = document.getElementById('dl-name').value.trim();
  if (!url) { toast('Please enter a download URL', 'error'); return; }
  toast('Starting download...', 'info');
  try {
    await api('/models/download', {method:'POST', body:JSON.stringify({url, filename:name||undefined})});
    toast('Download started!', 'success');
    document.getElementById('dl-url').value = '';
    document.getElementById('dl-name').value = '';
    setTimeout(() => renderDownload(document.getElementById('content')), 500);
  } catch(e) {}
}

async function startDownloadUrl(url, filename) {
  toast('Starting download: ' + filename, 'info');
  try {
    await api('/models/download', {method:'POST', body:JSON.stringify({url, filename})});
    toast('Download started: ' + filename, 'success');
    setTimeout(() => renderDownload(document.getElementById('content')), 500);
  } catch(e) {}
}

async function cancelDownload(id) {
  try {
    await api('/models/downloads/' + id, {method:'DELETE'});
    toast('Download cancelled', 'success');
    renderDownload(document.getElementById('content'));
  } catch(e) {}
}

/* ═══════════════════════════════════════════════════════════════════
   CHAT PAGE
   ═══════════════════════════════════════════════════════════════════ */
function renderChat(el) {
  el.innerHTML = `
    <div class="chat-container">
      <div class="chat-messages" id="chat-messages">
        ${state.chatMessages.length === 0 ? '<div style="text-align:center;padding:40px;color:var(--text-muted)"><p style="font-size:16px;margin-bottom:8px">No messages yet</p><p style="font-size:12px">Type a message below to start chatting with the loaded model.</p></div>' : ''}
        ${state.chatMessages.map(m => `
          <div class="chat-msg ${m.role}">
            <div class="chat-msg-label">${m.role === 'user' ? 'You' : 'Assistant'}</div>
            <div class="chat-msg-${m.role}">${escHtml(m.content)}</div>
          </div>
        `).join('')}
      </div>
      <div class="chat-input-bar">
        <input class="input" id="chat-input" placeholder="Type a message... (Enter to send)"
               onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}">
        <button class="btn btn-primary" onclick="sendChat()" id="chat-send-btn">Send</button>
        <button class="btn btn-sm" onclick="state.chatMessages=[];renderChat(document.getElementById('content'))">Clear</button>
      </div>
    </div>
  `;
  const msgs = document.getElementById('chat-messages');
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  state.chatMessages.push({role:'user', content:msg});
  renderChat(document.getElementById('content'));

  try {
    const messages = state.chatMessages.map(m => ({role:m.role, content:m.content}));
    const r = await fetch(API + '/v1/chat/completions', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages, stream:false, max_tokens:512})
    });
    if (!r.ok) throw new Error((await r.json().catch(()=>({})).detail) || 'Request failed');
    const data = await r.json();
    const reply = data.choices?.[0]?.message?.content || 'No response';
    state.chatMessages.push({role:'assistant', content:reply});
  } catch(e) {
    state.chatMessages.push({role:'assistant', content:'Error: ' + e.message});
  }
  renderChat(document.getElementById('content'));
}

/* ═══════════════════════════════════════════════════════════════════
   SETTINGS PAGE
   ═══════════════════════════════════════════════════════════════════ */
async function renderSettings(el) {
  const statusData = await api('/status').catch(()=>({}));
  const gpuData = await api('/gpu').catch(()=>({}));
  const eng = statusData.engine || {};
  const gpuS = statusData.gpu || {};
  const vram = gpuData.vram || {};

  el.innerHTML = `
    <div class="card">
      <div class="card-header"><div class="card-title">Server Configuration</div></div>
      <div class="table-wrap"><table>
        <tr><td style="color:var(--text-primary);width:200px">Model Loaded</td><td>${eng.model_loaded ? '<span class="badge badge-success">Yes</span>' : '<span class="badge badge-warning">No</span>'}</td></tr>
        <tr><td style="color:var(--text-primary)">Slots</td><td>${eng.slots_busy || 0} / ${eng.slots_total || 0} busy</td></tr>
        <tr><td style="color:var(--text-primary)">Total Generations</td><td>${eng.total_generations || 0}</td></tr>
        <tr><td style="color:var(--text-primary)">Total Tokens</td><td>${eng.total_tokens_generated || 0}</td></tr>
      </table></div>
    </div>

    <div class="card">
      <div class="card-header"><div class="card-title">GPU Settings</div></div>
      <div class="table-wrap"><table>
        <tr><td style="color:var(--text-primary);width:200px">GPU Name</td><td>${escHtml(gpuData.gpu_name || 'N/A')}</td></tr>
        <tr><td style="color:var(--text-primary)">VRAM Budget</td><td>${fmtBytes(vram.total_mb || 0)}</td></tr>
        <tr><td style="color:var(--text-primary)">VRAM Used</td><td>${fmtBytes(vram.used_mb || 0)}</td></tr>
        <tr><td style="color:var(--text-primary)">GPU Pressure</td><td>${escHtml(gpuS.pressure || gpuData.pressure || 'unknown')}</td></tr>
        <tr><td style="color:var(--text-primary)">Strategy</td><td>${escHtml(gpuS.strategy || 'none')}</td></tr>
        <tr><td style="color:var(--text-primary)">Layer Adjustments</td><td>${gpuData.layer_adjustments || 0}</td></tr>
      </table></div>
    </div>

    <div class="card">
      <div class="card-header"><div class="card-title">Performance</div></div>
      <div class="table-wrap"><table>
        <tr><td style="color:var(--text-primary);width:200px">Avg TPS</td><td>${(statusData.performance?.avg_tps || 0).toFixed(1)} tok/s</td></tr>
        <tr><td style="color:var(--text-primary)">P50 Latency</td><td>${statusData.performance?.p50_latency_ms?.toFixed(0) || '—'} ms</td></tr>
        <tr><td style="color:var(--text-primary)">P99 Latency</td><td>${statusData.performance?.p99_latency_ms?.toFixed(0) || '—'} ms</td></tr>
        <tr><td style="color:var(--text-primary)">Health</td><td>${escHtml(statusData.performance?.health || 'unknown')}</td></tr>
      </table></div>
    </div>

    <div class="card">
      <div class="card-header"><div class="card-title">Actions</div></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn" onclick="runBenchmark()">Run Benchmark</button>
        <button class="btn" onclick="cleanupDownloads()">Cleanup Partial Downloads</button>
        <button class="btn" onclick="getRecommendations()">Get Recommendations</button>
      </div>
    </div>
  `;
}

async function runBenchmark() {
  toast('Running benchmark...', 'info');
  try {
    const r = await api('/benchmark');
    toast('Benchmark: ' + r.tokens_per_second?.toFixed(1) + ' tok/s', 'success');
  } catch(e) {}
}

async function cleanupDownloads() {
  try {
    const r = await api('/models/cleanup', {method:'POST'});
    toast('Cleaned up ' + (r.cleaned || 0) + ' partial downloads', 'success');
  } catch(e) {}
}

async function getRecommendations() {
  try {
    const r = await api('/models/recommendations?use_case=general');
    if (r.recommendations && r.recommendations.length) {
      const modal = document.createElement('div');
      modal.className = 'modal-overlay';
      modal.onclick = e => { if(e.target===modal) modal.remove(); };
      modal.innerHTML = `<div class="modal">
        <div class="modal-title">Model Recommendations</div>
        ${r.recommendations.map(rec => `
          <div style="margin-bottom:12px;padding:10px;background:var(--bg-tertiary);border-radius:var(--radius)">
            <div style="font-weight:600;color:var(--text-primary)">${escHtml(rec.model_name)}</div>
            <div style="font-size:12px;color:var(--text-secondary);margin:4px 0">${escHtml(rec.reason)}</div>
            <div style="font-size:11px;color:var(--text-muted)">Confidence: ${(rec.confidence*100).toFixed(0)}% | VRAM: ${fmtBytes(rec.estimated_vram_mb)}</div>
          </div>
        `).join('')}
        <div class="modal-actions"><button class="btn" onclick="this.closest('.modal-overlay').remove()">Close</button></div>
      </div>`;
      document.body.appendChild(modal);
    } else {
      toast('No recommendations available', 'info');
    }
  } catch(e) {}
}

/* ═══════════════════════════════════════════════════════════════════
   INIT & AUTO-REFRESH
   ═══════════════════════════════════════════════════════════════════ */
function startAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(async () => {
    try {
      const status = await api('/health').catch(()=>({}));
      const dot = document.getElementById('conn-dot');
      const label = document.getElementById('conn-status');
      if (status.status === 'healthy' || status.status === 'degraded') {
        dot.className = 'status-dot ' + (status.status === 'healthy' ? 'ok' : 'warn');
        label.textContent = status.status;
      }
      // Auto-refresh downloads if on download page
      if (currentPage === 'download') {
        const dls = await api('/models/downloads').catch(()=>[]);
        const list = document.getElementById('download-list');
        if (list && dls.length > 0) {
          const active = dls.filter(d => d.status === 'active' || d.status === 'pending');
          if (active.length > 0) {
            list.innerHTML = dls.map(dl => renderDownloadItem(dl)).join('');
          }
        }
      }
    } catch(e) {
      const dot = document.getElementById('conn-dot');
      dot.className = 'status-dot err';
      document.getElementById('conn-status').textContent = 'Disconnected';
    }
  }, 3000);
}

// Boot
navigate('dashboard');
startAutoRefresh();
</script>
</body>
</html>
"""


# ─── GUI Mounting ────────────────────────────────────────────────────────────

def mount_gui(
    app: FastAPI,
    model_manager,
    gpu_manager,
    engine,
    perf_monitor,
):
    """Mount the web GUI onto an existing FastAPI application.

    Adds:
      - GET /gui/       — Serves the single-page application HTML
      - GET /gui        — Redirects to /gui/
    """
    @app.get("/gui/", response_class=HTMLResponse, include_in_schema=False)
    async def gui_page():
        """Serve the model management GUI."""
        return _GUI_HTML

    @app.get("/gui", response_class=HTMLResponse, include_in_schema=False)
    async def gui_page_no_slash():
        """Redirect /gui to /gui/."""
        return _GUI_HTML

    # Add a /models/search-hf endpoint for HuggingFace search from GUI
    @app.get("/models/search-hf", include_in_schema=False)
    async def search_huggingface(q: str = "", limit: int = 20):
        """Search HuggingFace for GGUF models. Uses HF API."""
        if not q:
            return {"results": []}
        import urllib.request
        import urllib.parse
        try:
            url = f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}+gguf&limit={limit}&sort=downloads&direction=-1"
            req = urllib.request.Request(url, headers={"User-Agent": "llama-cpp-smart-gpu/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            results = []
            for m in data[:limit]:
                results.append({
                    "id": m.get("id", ""),
                    "author": m.get("author", ""),
                    "downloads": m.get("downloads", 0),
                    "tags": m.get("tags", []),
                    "pipeline_tag": m.get("pipeline_tag", ""),
                    "last_modified": m.get("lastModified", ""),
                })
            return {"results": results}
        except Exception as e:
            logger.warning("HF search failed: %s", e)
            return {"results": [], "error": str(e)}

    logger.info("Web GUI mounted at /gui/")


# ==============================================================================
# SECTION: main.py
# ==============================================================================

#!/usr/bin/env python3
"""
llama-cpp-smart-gpu — Main entry point + shared utilities.

Smart GPU LLM inference engine powered by llama.cpp (no Ollama).
Features:
  - Professional model management with registry, validation, and download
  - Automatic GPU VRAM detection and layer offload optimization
  - Real-time VRAM pressure monitoring with adaptive scaling
  - Performance monitoring with Prometheus-compatible metrics
  - OpenAI-compatible API endpoints
  - Session-based conversations with context recycling
  - Continuous batching support
  - Model search, comparison, and recommendations
  - Web GUI for model downloading and management

Usage:
  python main.py -m /path/to/model.gguf
  python main.py -m model_name --n-gpu-layers -1 --vram-budget 8192
  python main.py --info
  python main.py --list-models
  python main.py --benchmark -m /path/to/model.gguf
  python main.py --download https://huggingface.co/.../model.gguf
"""


# ─── Shared Utilities (formerly utils.py) ────────────────────────────────────

def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure application-wide logging.

    Sets up console output with optional file rotation.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)

    handlers: list = [logging.StreamHandler(sys.stdout)]

    if config.file:
        Path(config.file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.file,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.max_backups,
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # Quiet down noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return logging.getLogger("llama-smart-gpu")


def get_system_info() -> Dict[str, Any]:
    """Collect system information for diagnostics."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "architecture": platform.machine(),
    }

    # RAM info
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_total_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("MemAvailable:"):
                    info["ram_available_mb"] = int(line.split()[1]) / 1024
    except Exception:
        pass

    # GPU info
    try:
        gpu_info = GPUDetector.detect()
        info["gpu_name"] = gpu_info.gpu_name
        info["gpu_vram_mb"] = gpu_info.total_vram_mb
    except Exception:
        info["gpu_available"] = False

    # llama-cpp-python version
    try:
        import llama_cpp
        info["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        info["llama_cpp_version"] = "not installed"

    return info


def format_bytes(num_bytes: float) -> str:
    """Format byte count to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="llama-smart-gpu",
        description="Smart GPU LLM inference engine powered by llama.cpp",
    )

    parser.add_argument("-m", "--model", type=str, help="Path or name of the GGUF model to load")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration file (JSON)")
    parser.add_argument("--host", type=str, help="Server host (default: 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, help="Server port (default: 8080)")
    parser.add_argument("--n-gpu-layers", type=int, help="Number of GPU layers (-1 = auto)")
    parser.add_argument("--n-ctx", type=int, help="Context window size in tokens")
    parser.add_argument("--n-batch", type=int, help="Batch size for prompt processing")
    parser.add_argument("--vram-budget", type=int, help="VRAM budget in MB (0 = auto-detect)")
    parser.add_argument("--vram-reserve", type=int, help="VRAM reserve in MB for system")
    parser.add_argument("--cache-type-k", type=str, choices=["f16", "q8_0", "q4_0", "q4_1"],
                        help="KV cache type for keys")
    parser.add_argument("--cache-type-v", type=str, choices=["f16", "q8_0", "q4_0", "q4_1"],
                        help="KV cache type for values")
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark and exit")
    parser.add_argument("--info", action="store_true", help="Print system info and exit")
    parser.add_argument("--list-models", action="store_true", help="List discovered models and exit")
    parser.add_argument("--download", type=str, help="Download a GGUF model from a URL")
    parser.add_argument("--download-name", type=str, help="Filename for downloaded model")
    parser.add_argument("--validate", type=str, help="Validate a model by name and exit")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare models side-by-side and exit")
    parser.add_argument("--recommend", type=str, choices=["general", "chat", "code", "reasoning"],
                        help="Get model recommendations and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def apply_cli_to_config(args: argparse.Namespace, config: SmartGPUConfig) -> SmartGPUConfig:
    """Apply CLI argument overrides to a configuration object."""
    overrides = {}

    if args.host:
        overrides["server.host"] = args.host
    if args.port:
        overrides["server.port"] = args.port
    if args.n_gpu_layers is not None:
        overrides["gpu.n_gpu_layers"] = args.n_gpu_layers
    if args.n_ctx:
        overrides["performance.n_ctx"] = args.n_ctx
    if args.n_batch:
        overrides["performance.n_batch"] = args.n_batch
    if args.vram_budget:
        overrides["gpu.vram_budget_mb"] = args.vram_budget
    if args.vram_reserve:
        overrides["gpu.vram_reserve_mb"] = args.vram_reserve
    if args.cache_type_k:
        overrides["performance.cache_type_k"] = args.cache_type_k
    if args.cache_type_v:
        overrides["performance.cache_type_v"] = args.cache_type_v
    if args.api_key:
        overrides["server.api_key"] = args.api_key
    if args.verbose:
        overrides["logging.level"] = "DEBUG"

    if overrides:
        return apply_cli_overrides(config, overrides)
    return config


def print_banner(config: SmartGPUConfig):
    """Print a startup banner."""
    banner = """
    ╔══════════════════════════════════════════════╗
    ║   llama-cpp-smart-gpu  ·  v2.1.0            ║
    ║   Smart GPU LLM Inference Engine             ║
    ╚══════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    Server:  {config.server.host}:{config.server.port}")
    print(f"    GPU:     {'enabled' if config.gpu.enabled else 'disabled'}")
    print(f"    Context: {config.performance.n_ctx} tokens")
    print(f"    Batch:   {config.performance.n_batch}")
    print(f"    GUI:     http://{config.server.host}:{config.server.port}/gui/")
    print()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    """Application entry point."""
    # Parse CLI arguments
    args = parse_args()

    # Load configuration (defaults + file + env + CLI)
    config = load_config(args.config)
    config = apply_cli_to_config(args, config)

    # Setup logging
    log = setup_logging(config.logging)

    # ── Info mode ────────────────────────────────────────────────────────
    if args.info:
        info = get_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    # Print banner
    print_banner(config)

    # ── Initialize components ────────────────────────────────────────────
    log.info("Initializing smart GPU manager...")
    gpu_manager = SmartGPUManager(
        vram_budget_mb=config.gpu.vram_budget_mb,
        vram_reserve_mb=config.gpu.vram_reserve_mb,
        vram_target_fraction=config.gpu.vram_target_fraction,
    )

    # Detect GPU hardware
    gpu_info = gpu_manager.initialize()
    log.info(
        "GPU: %s | VRAM: %.0f / %.0f MB",
        gpu_info.gpu_name, gpu_info.used_vram_mb, gpu_info.total_vram_mb,
    )

    # Create engine
    engine = LlamaEngine(config, gpu_manager)

    # Create model manager
    model_manager = ModelManager(config, gpu_manager)
    model_manager.set_engine(engine)

    # Create performance monitor
    perf_monitor = PerformanceMonitor(window_size=1000)

    # Create inference pipeline
    pipeline = InferencePipeline(engine, gpu_manager, perf_monitor, config.inference)

    # ── List models mode ─────────────────────────────────────────────────
    if args.list_models:
        models = model_manager.discover_models()
        if not models:
            print("No models found. Place .gguf files in:")
            print(f"  {config.model_dir}")
            print("Or specify a path with -m / --model")
            print("Or download with --download <url>")
            print("Or use the web GUI at http://localhost:8080/gui/")
        else:
            available_vram = gpu_manager.get_available_vram()
            print(f"\n  Found {len(models)} model(s) | Available VRAM: {available_vram:.0f} MB\n")
            print(f"  {'Name':<35} {'Params':>7} {'Quant':<10} {'Size':>9} {'VRAM est':>10} {'Fit':>8} {'Family':<10}")
            print(f"  {'─'*35} {'─'*7} {'─'*10} {'─'*9} {'─'*10} {'─'*8} {'─'*10}")
            for m in models:
                fit_str = "✓ FITS" if m.fits_in_vram else "✗ TIGHT"
                if m.vram_fit_score < 0.3:
                    fit_str = "✗ OVER"
                size_str = f"{m.size_gb:.1f} GB" if m.size_gb >= 1 else f"{m.size_mb:.0f} MB"
                vram_str = f"{m.estimated_vram_full_mb:.0f} MB" if m.estimated_vram_full_mb > 0 else "—"
                print(f"  {m.name:<35} {m.n_params_b:>6.1f}B {m.quantization:<10} {size_str:>9} {vram_str:>10} {fit_str:>8} {m.family:<10}")
                if m.aliases:
                    print(f"  {'':35} aliases: {', '.join(m.aliases)}")
                if m.tags:
                    print(f"  {'':35} tags: {', '.join(m.tags)}")
                if m.auto_config:
                    gpu_layers = m.auto_config.get("n_gpu_layers", "auto")
                    cache = f"k={m.auto_config.get('cache_type_k','?')}, v={m.auto_config.get('cache_type_v','?')}"
                    print(f"  {'':35} auto-config: gpu_layers={gpu_layers}, cache={cache}")
                if m.avg_tps > 0:
                    print(f"  {'':35} avg tps: {m.avg_tps:.1f} | best: {m.best_tps:.1f} | loads: {m.load_count}")
            print()
        sys.exit(0)

    # ── Validate mode ────────────────────────────────────────────────────
    if hasattr(args, 'validate') and args.validate:
        print(f"  Validating model: {args.validate}")
        result = model_manager.validate_model(args.validate, deep=False)
        print(f"  Quick check: {'PASS' if result['quick_check']['passed'] else 'FAIL'}")
        for err in result.get('quick_check', {}).get('errors', []):
            print(f"    Error: {err}")
        print(f"  Status: {result['status']}")
        sys.exit(0 if result['quick_check']['passed'] else 1)

    # ── Compare mode ────────────────────────────────────────────────────
    if hasattr(args, 'compare') and args.compare:
        if len(args.compare) < 2:
            print("  Need at least 2 model names to compare")
            sys.exit(1)
        comparison = model_manager.compare_models(args.compare)
        print(f"\n  Model Comparison\n")
        for m in comparison.get("models", []):
            print(f"  {m['name']}")
            print(f"    Params: {m['params_b']}B | Quant: {m['quantization']} ({m['quality_tier']})")
            print(f"    Size: {m['size_mb']:.0f} MB | VRAM: {m['estimated_vram_mb']:.0f} MB")
            print(f"    Fits VRAM: {'Yes' if m['fits_vram'] else 'No'} (score: {m['vram_fit_score']})")
            print(f"    Context: {m['context_length']} | GPU layers: {m['recommended_gpu_layers']}")
            print(f"    Score: {m['score']}")
            print()
        print(f"  {comparison.get('verdict', '')}")
        sys.exit(0)

    # ── Recommend mode ──────────────────────────────────────────────────
    if hasattr(args, 'recommend') and args.recommend:
        recs = model_manager.get_recommendations(args.recommend)
        if not recs:
            print("  No model recommendations available")
        else:
            print(f"\n  Recommendations for '{args.recommend}' use case:\n")
            for i, r in enumerate(recs[:5], 1):
                print(f"  {i}. {r.model_name} (confidence: {r.confidence:.0%})")
                print(f"     Reason: {r.reason}")
                print(f"     Suggested: ctx={r.suggested_n_ctx}, batch={r.suggested_n_batch}, "
                      f"gpu_layers={r.suggested_n_gpu_layers}, cache_k={r.suggested_cache_k}")
                print(f"     Estimated VRAM: {r.estimated_vram_mb:.0f} MB")
                print()
        sys.exit(0)

    # ── Download mode ────────────────────────────────────────────────────
    if hasattr(args, 'download') and args.download:
        model_manager.start_downloads()
        print(f"  Downloading: {args.download}")
        task = model_manager.download_model(args.download, name=getattr(args, 'download_name', None))
        import time as _time
        while True:
            status = model_manager.get_download_status(task.id)
            if not status:
                break
            pct = status.get("progress_pct", 0)
            speed = status.get("speed_mbps", 0)
            eta = status.get("eta_seconds", 0)
            dl_status = status.get("status", "unknown")
            dl_bytes = status.get("downloaded_bytes", 0)
            total_bytes = status.get("total_bytes", 0)
            mb_dl = dl_bytes / (1024**2)
            mb_total = total_bytes / (1024**2)
            print(f"\r  Progress: {pct:.1f}% ({mb_dl:.0f}/{mb_total:.0f} MB) | "
                  f"{speed:.1f} MB/s | ETA: {eta:.0f}s | {dl_status}", end="", flush=True)
            if dl_status in ("completed", "failed", "cancelled"):
                print()
                break
            _time.sleep(1)
        model_manager.stop_downloads()
        sys.exit(0)

    # ── Load model ───────────────────────────────────────────────────────
    model_to_load = args.model
    if not model_to_load and config.model_paths:
        model_to_load = config.model_paths[0]

    if not model_to_load:
        # Try auto-selecting
        best = model_manager.select_best_model()
        if best:
            model_to_load = best.name
            log.info("Auto-selected model: %s", model_to_load)
        else:
            log.warning(
                "No model specified. Server starting without a loaded model. "
                "Use the GUI at http://%s:%d/gui/ to download and load models.",
                config.server.host, config.server.port,
            )

    if model_to_load:
        try:
            model_manager.load_model(model_to_load)
        except ModelNotFoundError as exc:
            log.error("Model not found: %s", exc)
            sys.exit(1)
        except ModelLoadError as exc:
            log.error("Failed to load model: %s", exc)
            sys.exit(1)
        except GPUOutOfMemoryError as exc:
            log.error("GPU OOM: %s", exc)
            log.error("Try: --vram-budget %d --n-gpu-layers -1", int(exc.available_mb * 0.8))
            sys.exit(1)
        except LlamaSmartGPUError as exc:
            log.error("Startup error [%s]: %s", exc.code, exc)
            sys.exit(1)
        except Exception as exc:
            log.error("Unexpected error: %s", exc)
            sys.exit(1)

    # ── Benchmark mode ───────────────────────────────────────────────────
    if args.benchmark:
        try:
            result = engine.benchmark()
            print("\n  Benchmark Results:")
            print(f"  Throughput:     {result['tokens_per_second']:.1f} tok/s")
            print(f"  Prompt tokens:  {result['prompt_tokens']}")
            print(f"  Completion:     {result['completion_tokens']} tokens")
            print(f"  Total time:     {result['total_time_ms']:.0f} ms")
            perf_monitor.set_baseline(result["tokens_per_second"])

            # Record in model history
            current = model_manager.get_current_model()
            if current:
                model_manager.record_performance(current.name, result["tokens_per_second"])
        except Exception as exc:
            log.error("Benchmark failed: %s", exc)
            sys.exit(1)
        sys.exit(0)

    # Set performance baseline after first generation
    perf_monitor.auto_baseline()

    # ── Start server ─────────────────────────────────────────────────────
    app = create_app(config, engine, gpu_manager, model_manager, perf_monitor, pipeline)

    # Mount the web GUI
    mount_gui(app, model_manager, gpu_manager, engine, perf_monitor)

    import uvicorn
    log.info(
        "Starting server on %s:%d (API key: %s) | GUI: /gui/",
        config.server.host, config.server.port,
        "enabled" if config.server.api_key else "disabled",
    )

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
        timeout_keep_alive=config.server.timeout_seconds,
    )


if __name__ == "__main__":
    main()
