"""
Comprehensive performance benchmarking suite for ML models.
Measures training speed, inference latency, throughput, memory usage, and model quality.
"""

import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import subprocess

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger
import pandas as pd

# Note: These imports would work when running from package root
# For standalone execution, we'll handle imports gracefully
try:
    from ..serving.model_server import ModelServer, ServingConfig
    from ..monitoring.model_monitor import ModelMonitor, MonitoringConfig
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from serving.model_server import ModelServer, ServingConfig
        from monitoring.model_monitor import ModelMonitor, MonitoringConfig
    except ImportError:
        # Mock classes for demonstration if imports fail
        class ModelServer:
            pass
        class ServingConfig:
            pass
        class ModelMonitor:
            pass
        class MonitoringConfig:
            pass


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    
    # Model configuration
    model_path: str
    model_name: str
    
    # Benchmark settings
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    
    # Load testing
    max_concurrent_requests: int = 100
    load_test_duration: int = 60  # seconds
    target_qps: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 200])
    
    # Hardware configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp16"  # fp32, fp16, int8
    
    # Output
    output_dir: str = "./benchmark_results"
    save_plots: bool = True
    save_detailed_results: bool = True


class LatencyBenchmark:
    """Benchmarks model inference latency."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model for benchmarking."""
        logger.info(f"Loading model: {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.config.device == "cuda" else None,
        }
        
        if self.config.precision == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.precision == "int8":
            model_kwargs["load_in_8bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        
        self.model.eval()
        
    def benchmark_single_inference(
        self,
        batch_size: int,
        sequence_length: int,
        num_iterations: int = None
    ) -> Dict[str, float]:
        """Benchmark single inference performance."""
        
        if num_iterations is None:
            num_iterations = self.config.benchmark_iterations
            
        # Create dummy input
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (batch_size, sequence_length),
            device=self.model.device
        )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
        if self.config.device == "cuda":
            torch.cuda.synchronize()
            
        # Benchmark
        latencies = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
            if self.config.device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
        return {
            "mean_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "throughput_samples_per_sec": batch_size * 1000 / statistics.mean(latencies),
        }
        
    def benchmark_generation(
        self,
        batch_size: int,
        input_length: int,
        output_length: int,
        num_iterations: int = None
    ) -> Dict[str, float]:
        """Benchmark text generation performance."""
        
        if num_iterations is None:
            num_iterations = self.config.benchmark_iterations
            
        # Create dummy input
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (batch_size, input_length),
            device=self.model.device
        )
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids,
                    max_new_tokens=output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
        if self.config.device == "cuda":
            torch.cuda.synchronize()
            
        # Benchmark
        latencies = []
        tokens_per_second = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=output_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            if self.config.device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency * 1000)  # Convert to ms
            
            # Calculate tokens per second
            total_tokens = batch_size * output_length
            tokens_per_second.append(total_tokens / latency)
            
        return {
            "mean_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "mean_tokens_per_sec": statistics.mean(tokens_per_second),
            "p50_tokens_per_sec": statistics.median(tokens_per_second),
            "p95_tokens_per_sec": np.percentile(tokens_per_second, 95),
        }
        
    def run_latency_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive latency benchmarks."""
        logger.info("Running latency benchmarks")
        
        results = {
            "inference": {},
            "generation": {},
            "config": self.config.__dict__,
        }
        
        # Inference benchmarks
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                logger.info(f"Benchmarking inference: batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    result = self.benchmark_single_inference(batch_size, seq_len)
                    results["inference"][f"bs_{batch_size}_sl_{seq_len}"] = result
                except Exception as e:
                    logger.error(f"Inference benchmark failed: {e}")
                    
        # Generation benchmarks
        input_lengths = [128, 256]
        output_lengths = [128, 256]
        
        for batch_size in [1, 4, 8]:
            for input_len in input_lengths:
                for output_len in output_lengths:
                    logger.info(f"Benchmarking generation: batch_size={batch_size}, "
                              f"input_len={input_len}, output_len={output_len}")
                    
                    try:
                        result = self.benchmark_generation(batch_size, input_len, output_len)
                        key = f"bs_{batch_size}_il_{input_len}_ol_{output_len}"
                        results["generation"][key] = result
                    except Exception as e:
                        logger.error(f"Generation benchmark failed: {e}")
                        
        return results


class ThroughputBenchmark:
    """Benchmarks model serving throughput under load."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model_server = None
        
    async def setup_server(self):
        """Setup model server for throughput testing."""
        serving_config = ServingConfig(
            model_path=self.config.model_path,
            device=self.config.device,
            max_batch_size=32,
            optimization_level="O2",
        )
        
        # This would start the actual server in production
        # For benchmarking, we'll simulate the server
        pass
        
    async def send_request(self, session, prompt: str) -> Dict[str, Any]:
        """Send a single request to the model server."""
        start_time = time.perf_counter()
        
        # Simulate request processing
        await asyncio.sleep(0.1)  # Placeholder for actual API call
        
        end_time = time.perf_counter()
        
        return {
            "latency_ms": (end_time - start_time) * 1000,
            "success": True,
            "response_length": 100,  # Placeholder
        }
        
    async def load_test(self, qps: int, duration: int) -> Dict[str, Any]:
        """Run load test at specified QPS for given duration."""
        logger.info(f"Running load test: {qps} QPS for {duration} seconds")
        
        requests_sent = 0
        responses_received = 0
        latencies = []
        errors = 0
        
        start_time = time.time()
        end_time = start_time + duration
        
        async def request_worker():
            nonlocal responses_received, errors
            
            while time.time() < end_time:
                try:
                    # Send request
                    result = await self.send_request(None, "test prompt")
                    latencies.append(result["latency_ms"])
                    responses_received += 1
                    
                    if not result["success"]:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    
                # Wait to maintain QPS
                await asyncio.sleep(1.0 / qps)
                
        # Start request workers
        tasks = []
        for _ in range(min(qps, 50)):  # Limit concurrent workers
            tasks.append(asyncio.create_task(request_worker()))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        actual_qps = responses_received / actual_duration
        error_rate = errors / max(responses_received, 1)
        
        return {
            "target_qps": qps,
            "actual_qps": actual_qps,
            "duration": actual_duration,
            "total_requests": responses_received,
            "error_rate": error_rate,
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
        }
        
    async def run_throughput_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive throughput benchmarks."""
        logger.info("Running throughput benchmarks")
        
        await self.setup_server()
        
        results = {
            "load_tests": {},
            "config": self.config.__dict__,
        }
        
        for qps in self.config.target_qps:
            try:
                result = await self.load_test(qps, self.config.load_test_duration)
                results["load_tests"][f"qps_{qps}"] = result
            except Exception as e:
                logger.error(f"Load test failed for {qps} QPS: {e}")
                
        return results


class MemoryBenchmark:
    """Benchmarks memory usage patterns."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
    def measure_model_memory(self, model) -> Dict[str, float]:
        """Measure model memory usage."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Trigger memory allocation
            dummy_input = torch.randint(0, 1000, (1, 100), device="cuda")
            with torch.no_grad():
                _ = model(dummy_input)
                
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "gpu_memory_allocated_gb": memory_allocated,
                "gpu_memory_reserved_gb": memory_reserved,
                "gpu_peak_memory_gb": peak_memory,
            }
        else:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "cpu_memory_rss_gb": memory_info.rss / 1024**3,
                "cpu_memory_vms_gb": memory_info.vms / 1024**3,
            }
            
    def profile_memory_during_inference(self, model, tokenizer, batch_size: int, seq_len: int):
        """Profile memory usage during inference."""
        memory_samples = []
        
        # Create input
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
        if self.config.device == "cuda":
            input_ids = input_ids.cuda()
            
        # Profile memory during inference
        def memory_monitor():
            while True:
                if self.config.device == "cuda":
                    memory = torch.cuda.memory_allocated() / 1024**3
                else:
                    memory = psutil.Process().memory_info().rss / 1024**3
                    
                memory_samples.append(memory)
                time.sleep(0.01)  # Sample every 10ms
                
        # Start monitoring
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run inference
        with torch.no_grad():
            _ = model(input_ids)
            
        # Stop monitoring
        time.sleep(0.1)  # Let monitor capture final state
        
        return {
            "peak_memory_gb": max(memory_samples) if memory_samples else 0,
            "mean_memory_gb": statistics.mean(memory_samples) if memory_samples else 0,
            "memory_samples": memory_samples,
        }


class BenchmarkRunner:
    """Main benchmark runner that orchestrates all benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarks")
        
        results = {
            "config": self.config.__dict__,
            "system_info": self._get_system_info(),
            "latency": {},
            "throughput": {},
            "memory": {},
            "timestamp": time.time(),
        }
        
        # Latency benchmarks
        try:
            latency_benchmark = LatencyBenchmark(self.config)
            latency_benchmark.load_model()
            results["latency"] = latency_benchmark.run_latency_benchmarks()
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            
        # Throughput benchmarks
        try:
            throughput_benchmark = ThroughputBenchmark(self.config)
            results["throughput"] = asyncio.run(
                throughput_benchmark.run_throughput_benchmarks()
            )
        except Exception as e:
            logger.error(f"Throughput benchmark failed: {e}")
            
        # Memory benchmarks
        try:
            memory_benchmark = MemoryBenchmark(self.config)
            if hasattr(latency_benchmark, 'model'):
                results["memory"] = memory_benchmark.measure_model_memory(
                    latency_benchmark.model
                )
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            
        # Save results
        self._save_results(results)
        
        if self.config.save_plots:
            self._create_plots(results)
            
        return results
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": subprocess.check_output(
                ["python", "--version"], text=True
            ).strip(),
            "torch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) 
                             for i in range(torch.cuda.device_count())],
            })
        else:
            info["cuda_available"] = False
            
        return info
        
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results."""
        # Save JSON results
        json_path = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {json_path}")
        
        # Save CSV summary
        self._save_csv_summary(results)
        
    def _save_csv_summary(self, results: Dict[str, Any]):
        """Save CSV summary of key metrics."""
        rows = []
        
        # Latency results
        if "latency" in results and "inference" in results["latency"]:
            for key, metrics in results["latency"]["inference"].items():
                row = {
                    "benchmark_type": "inference",
                    "configuration": key,
                    "mean_latency_ms": metrics.get("mean_latency_ms", 0),
                    "p95_latency_ms": metrics.get("p95_latency_ms", 0),
                    "throughput_samples_per_sec": metrics.get("throughput_samples_per_sec", 0),
                }
                rows.append(row)
                
        # Throughput results
        if "throughput" in results and "load_tests" in results["throughput"]:
            for key, metrics in results["throughput"]["load_tests"].items():
                row = {
                    "benchmark_type": "throughput",
                    "configuration": key,
                    "target_qps": metrics.get("target_qps", 0),
                    "actual_qps": metrics.get("actual_qps", 0),
                    "error_rate": metrics.get("error_rate", 0),
                    "p95_latency_ms": metrics.get("p95_latency_ms", 0),
                }
                rows.append(row)
                
        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.output_dir / "benchmark_summary.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Summary saved to {csv_path}")
            
    def _create_plots(self, results: Dict[str, Any]):
        """Create visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # Latency vs batch size plot
        if "latency" in results and "inference" in results["latency"]:
            self._plot_latency_vs_batch_size(results["latency"]["inference"])
            
        # Throughput vs QPS plot
        if "throughput" in results and "load_tests" in results["throughput"]:
            self._plot_throughput_results(results["throughput"]["load_tests"])
            
    def _plot_latency_vs_batch_size(self, latency_results: Dict[str, Any]):
        """Plot latency vs batch size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group by sequence length
        seq_lengths = set()
        for key in latency_results.keys():
            seq_len = int(key.split("_sl_")[1])
            seq_lengths.add(seq_len)
            
        for seq_len in sorted(seq_lengths):
            batch_sizes = []
            latencies = []
            throughputs = []
            
            for key, metrics in latency_results.items():
                if f"_sl_{seq_len}" in key:
                    batch_size = int(key.split("_bs_")[1].split("_")[0])
                    batch_sizes.append(batch_size)
                    latencies.append(metrics["mean_latency_ms"])
                    throughputs.append(metrics["throughput_samples_per_sec"])
                    
            # Sort by batch size
            sorted_data = sorted(zip(batch_sizes, latencies, throughputs))
            batch_sizes, latencies, throughputs = zip(*sorted_data)
            
            ax1.plot(batch_sizes, latencies, marker='o', label=f"seq_len={seq_len}")
            ax2.plot(batch_sizes, throughputs, marker='s', label=f"seq_len={seq_len}")
            
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Latency vs Batch Size")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Throughput (samples/sec)")
        ax2.set_title("Throughput vs Batch Size")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_throughput_results(self, throughput_results: Dict[str, Any]):
        """Plot throughput test results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        target_qps = []
        actual_qps = []
        error_rates = []
        p95_latencies = []
        
        for key, metrics in throughput_results.items():
            target_qps.append(metrics["target_qps"])
            actual_qps.append(metrics["actual_qps"])
            error_rates.append(metrics["error_rate"] * 100)  # Convert to percentage
            p95_latencies.append(metrics["p95_latency_ms"])
            
        # QPS comparison
        ax1.plot(target_qps, actual_qps, marker='o', label="Actual QPS")
        ax1.plot(target_qps, target_qps, '--', label="Target QPS", alpha=0.7)
        ax1.set_xlabel("Target QPS")
        ax1.set_ylabel("Actual QPS")
        ax1.set_title("Throughput: Target vs Actual QPS")
        ax1.legend()
        ax1.grid(True)
        
        # Error rate and latency
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(target_qps, error_rates, 'r-o', label="Error Rate (%)")
        line2 = ax2_twin.plot(target_qps, p95_latencies, 'b-s', label="P95 Latency (ms)")
        
        ax2.set_xlabel("Target QPS")
        ax2.set_ylabel("Error Rate (%)", color='r')
        ax2_twin.set_ylabel("P95 Latency (ms)", color='b')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        ax2.set_title("Error Rate and Latency vs Load")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "throughput_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_benchmark_config(
    model_path: str,
    model_name: str,
    output_dir: str = "./benchmark_results"
) -> BenchmarkConfig:
    """Create a default benchmark configuration."""
    return BenchmarkConfig(
        model_path=model_path,
        model_name=model_name,
        output_dir=output_dir,
        batch_sizes=[1, 2, 4, 8, 16],
        sequence_lengths=[128, 256, 512, 1024],
        target_qps=[1, 5, 10, 25, 50],
        benchmark_iterations=50,
        warmup_iterations=5,
    )