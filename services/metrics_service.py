"""
Comprehensive Metrics Service for RAG Pipeline.

Tracks cost, latency, and accuracy KPIs across all components:
- Retrieval performance (BM25, cross-encoder)
- LLM usage and costs (OpenRouter API)
- Crawling efficiency and success rates
- End-to-end advisor performance
- User satisfaction and recommendation quality
"""

import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid

logger = logging.getLogger("metrics")


class MetricType(Enum):
    """Types of metrics we track."""
    LATENCY = "latency"
    COST = "cost"
    ACCURACY = "accuracy"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    QUALITY = "quality"


class ComponentType(Enum):
    """RAG pipeline components."""
    RETRIEVER = "retriever"
    CRAWLER = "crawler"
    LLM = "llm"
    CROSS_ENCODER = "cross_encoder"
    ADVISOR = "advisor"
    AGENT = "agent"
    API = "api"


@dataclass
class MetricEvent:
    """Individual metric event."""
    timestamp: str
    request_id: str
    component: ComponentType
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Ensure timestamp is ISO format."""
        if isinstance(self.timestamp, datetime):
            self.timestamp = self.timestamp.isoformat()
        elif not isinstance(self.timestamp, str):
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class LatencyMetric:
    """Latency-specific metrics."""
    component: ComponentType
    operation: str
    duration_ms: float
    success: bool
    request_id: str
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CostMetric:
    """Cost-specific metrics."""
    component: ComponentType
    operation: str
    cost_usd: float
    tokens_used: Optional[int]
    model_name: Optional[str]
    request_id: str
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AccuracyMetric:
    """Accuracy and quality metrics."""
    component: ComponentType
    operation: str
    accuracy_score: float  # 0.0 to 1.0
    total_items: int
    correct_items: int
    request_id: str
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


class MetricsCollector:
    """Central metrics collection and storage."""

    def __init__(self, storage_dir: str = "metrics"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # In-memory buffers for real-time metrics
        self._latency_buffer: List[LatencyMetric] = []
        self._cost_buffer: List[CostMetric] = []
        self._accuracy_buffer: List[AccuracyMetric] = []

        # File paths
        self.latency_file = self.storage_dir / "latency_metrics.csv"
        self.cost_file = self.storage_dir / "cost_metrics.csv"
        self.accuracy_file = self.storage_dir / "accuracy_metrics.csv"
        self.summary_file = self.storage_dir / "metrics_summary.json"

        # Initialize CSV files with headers
        self._initialize_csv_files()

        # Background task for periodic flushing
        self._flush_task = None
        self._flush_interval = 30  # seconds

    def _initialize_csv_files(self):
        """Initialize CSV files with proper headers."""
        # Latency CSV
        if not self.latency_file.exists():
            with open(self.latency_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'request_id', 'component', 'operation',
                    'duration_ms', 'success', 'metadata'
                ])

        # Cost CSV
        if not self.cost_file.exists():
            with open(self.cost_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'request_id', 'component', 'operation',
                    'cost_usd', 'tokens_used', 'model_name', 'metadata'
                ])

        # Accuracy CSV
        if not self.accuracy_file.exists():
            with open(self.accuracy_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'request_id', 'component', 'operation',
                    'accuracy_score', 'total_items', 'correct_items', 'metadata'
                ])

    async def start_background_flush(self):
        """Start background task for periodic metric flushing."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop_background_flush(self):
        """Stop background flushing and flush remaining metrics."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self.flush_metrics()

    async def _periodic_flush(self):
        """Periodically flush metrics to disk."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    def record_latency(self, component: ComponentType, operation: str,
                      duration_ms: float, success: bool = True,
                      request_id: str = None, **metadata):
        """Record a latency metric."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        metric = LatencyMetric(
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            request_id=request_id,
            metadata=metadata
        )

        self._latency_buffer.append(metric)
        logger.debug(f"Recorded latency: {component.value}.{operation} = {duration_ms:.2f}ms")

    def record_cost(self, component: ComponentType, operation: str,
                   cost_usd: float, tokens_used: int = None,
                   model_name: str = None, request_id: str = None, **metadata):
        """Record a cost metric."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        metric = CostMetric(
            component=component,
            operation=operation,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            model_name=model_name,
            request_id=request_id,
            metadata=metadata
        )

        self._cost_buffer.append(metric)
        logger.debug(f"Recorded cost: {component.value}.{operation} = ${cost_usd:.4f}")

    def record_accuracy(self, component: ComponentType, operation: str,
                       accuracy_score: float, total_items: int, correct_items: int,
                       request_id: str = None, **metadata):
        """Record an accuracy metric."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        metric = AccuracyMetric(
            component=component,
            operation=operation,
            accuracy_score=accuracy_score,
            total_items=total_items,
            correct_items=correct_items,
            request_id=request_id,
            metadata=metadata
        )

        self._accuracy_buffer.append(metric)
        logger.debug(f"Recorded accuracy: {component.value}.{operation} = {accuracy_score:.2f}")

    async def flush_metrics(self):
        """Flush all buffered metrics to CSV files."""
        try:
            # Flush latency metrics
            if self._latency_buffer:
                await self._flush_latency_metrics()

            # Flush cost metrics
            if self._cost_buffer:
                await self._flush_cost_metrics()

            # Flush accuracy metrics
            if self._accuracy_buffer:
                await self._flush_accuracy_metrics()

            logger.debug("Metrics flushed to disk")

        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")

    async def _flush_latency_metrics(self):
        """Flush latency metrics to CSV."""
        def _write_latency():
            with open(self.latency_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for metric in self._latency_buffer:
                    writer.writerow([
                        metric.timestamp,
                        metric.request_id,
                        metric.component.value,
                        metric.operation,
                        metric.duration_ms,
                        metric.success,
                        json.dumps(metric.metadata)
                    ])

        await asyncio.to_thread(_write_latency)
        self._latency_buffer.clear()

    async def _flush_cost_metrics(self):
        """Flush cost metrics to CSV."""
        def _write_cost():
            with open(self.cost_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for metric in self._cost_buffer:
                    writer.writerow([
                        metric.timestamp,
                        metric.request_id,
                        metric.component.value,
                        metric.operation,
                        metric.cost_usd,
                        metric.tokens_used,
                        metric.model_name,
                        json.dumps(metric.metadata)
                    ])

        await asyncio.to_thread(_write_cost)
        self._cost_buffer.clear()

    async def _flush_accuracy_metrics(self):
        """Flush accuracy metrics to CSV."""
        def _write_accuracy():
            with open(self.accuracy_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for metric in self._accuracy_buffer:
                    writer.writerow([
                        metric.timestamp,
                        metric.request_id,
                        metric.component.value,
                        metric.operation,
                        metric.accuracy_score,
                        metric.total_items,
                        metric.correct_items,
                        json.dumps(metric.metadata)
                    ])

        await asyncio.to_thread(_write_accuracy)
        self._accuracy_buffer.clear()

    async def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        try:
            summary = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'latency_stats': await self._calculate_latency_stats(),
                'cost_stats': await self._calculate_cost_stats(),
                'accuracy_stats': await self._calculate_accuracy_stats(),
                'component_performance': await self._calculate_component_performance()
            }

            # Save summary to JSON
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            return summary

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {}

    async def _calculate_latency_stats(self) -> Dict[str, Any]:
        """Calculate latency statistics from CSV."""
        # Implementation would read from CSV and calculate stats
        # For now, return placeholder
        return {
            'avg_response_time_ms': 0.0,
            'p95_response_time_ms': 0.0,
            'p99_response_time_ms': 0.0,
            'success_rate': 1.0,
            'total_requests': 0
        }

    async def _calculate_cost_stats(self) -> Dict[str, Any]:
        """Calculate cost statistics from CSV."""
        return {
            'total_cost_usd': 0.0,
            'avg_cost_per_request': 0.0,
            'total_tokens': 0,
            'cost_by_model': {}
        }

    async def _calculate_accuracy_stats(self) -> Dict[str, Any]:
        """Calculate accuracy statistics from CSV."""
        return {
            'overall_accuracy': 0.0,
            'accuracy_by_component': {},
            'total_evaluations': 0
        }

    async def _calculate_component_performance(self) -> Dict[str, Any]:
        """Calculate per-component performance metrics."""
        return {
            'retriever': {'avg_latency_ms': 0.0, 'success_rate': 1.0},
            'crawler': {'avg_latency_ms': 0.0, 'success_rate': 1.0},
            'llm': {'avg_latency_ms': 0.0, 'avg_cost_usd': 0.0},
            'advisor': {'avg_latency_ms': 0.0, 'success_rate': 1.0},
            'agent': {'avg_latency_ms': 0.0, 'success_rate': 1.0},
            'api': {'avg_latency_ms': 0.0, 'success_rate': 1.0}
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions for common metrics
async def record_latency(component: ComponentType, operation: str,
                        duration_ms: float, success: bool = True,
                        request_id: str = None, **metadata):
    """Record latency metric."""
    collector = get_metrics_collector()
    collector.record_latency(component, operation, duration_ms, success, request_id, **metadata)


async def record_cost(component: ComponentType, operation: str,
                     cost_usd: float, tokens_used: int = None,
                     model_name: str = None, request_id: str = None, **metadata):
    """Record cost metric."""
    collector = get_metrics_collector()
    collector.record_cost(component, operation, cost_usd, tokens_used, model_name, request_id, **metadata)


async def record_accuracy(component: ComponentType, operation: str,
                         accuracy_score: float, total_items: int, correct_items: int,
                         request_id: str = None, **metadata):
    """Record accuracy metric."""
    collector = get_metrics_collector()
    collector.record_accuracy(component, operation, accuracy_score, total_items, correct_items, request_id, **metadata)


class MetricsContext:
    """Context manager for tracking operation metrics."""

    def __init__(self, component: ComponentType, operation: str, request_id: str = None):
        self.component = component
        self.operation = operation
        self.request_id = request_id or str(uuid.uuid4())
        self.start_time = None
        self.success = True
        self.metadata = {}

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aenter__(self):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            await record_latency(
                self.component,
                self.operation,
                duration_ms,
                self.success,
                self.request_id,
                **self.metadata
            )

    def set_success(self, success: bool):
        """Set operation success status."""
        self.success = success

    def add_metadata(self, **metadata):
        """Add metadata to the metric."""
        self.metadata.update(metadata)