#!/usr/bin/env python3
"""
LLM Engine Benchmark Runner

This script reads a YAML configuration file, starts the specified LLM engine,
and runs benchmark tests using the engine's native benchmarking tools.
"""

import argparse
import os
import subprocess
import sys
import time
import yaml
import signal
import requests
from pathlib import Path
from typing import Dict, Any, Optional


class EngineManager:
    """Base class for managing LLM engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        
    def start(self) -> None:
        """Start the LLM engine server"""
        raise NotImplementedError
        
    def stop(self) -> None:
        """Stop the LLM engine server"""
        if self.process:
            print(f"Stopping {self.__class__.__name__}...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing process...")
                self.process.kill()
            self.process = None
            
    def is_ready(self) -> bool:
        """Check if the server is ready to accept requests"""
        raise NotImplementedError
        
    def wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for server to be ready"""
        print(f"Waiting for server to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready():
                print("Server is ready!")
                return True
            time.sleep(2)
        print("Server failed to become ready within timeout")
        return False


class SGLangManager(EngineManager):
    """Manager for SGLang engine"""
    
    def start(self) -> None:
        """Start SGLang server"""
        server_config = self.config['server']
        sglang_config = server_config.get('sglang', {})
        common_config = server_config.get('common', {})
        
        model = server_config['model']
        host = sglang_config.get('host', '127.0.0.1')
        port = sglang_config.get('port', 30000)
        
        cmd = [
            'python3', '-m', 'sglang.launch_server',
            '--model-path', model,
            '--host', host,
            '--port', str(port),
        ]
        
        # Add SGLang-specific parameters
        if 'tp_size' in sglang_config:
            cmd.extend(['--tp-size', str(sglang_config['tp_size'])])
        if 'pp_size' in sglang_config:
            cmd.extend(['--pp-size', str(sglang_config['pp_size'])])
        if 'mem_fraction_static' in sglang_config:
            cmd.extend(['--mem-fraction-static', str(sglang_config['mem_fraction_static'])])
        if 'context_length' in sglang_config:
            cmd.extend(['--context-length', str(sglang_config['context_length'])])
        if sglang_config.get('disable_radix_cache', False):
            cmd.append('--disable-radix-cache')
        if sglang_config.get('enable_flashinfer', False):
            cmd.append('--enable-flashinfer')
        if sglang_config.get('enable_metrics', False):
            cmd.append('--enable-metrics')
        if 'chunked_prefill_size' in sglang_config:
            cmd.extend(['--chunked-prefill-size', str(sglang_config['chunked_prefill_size'])])
            
        # Add common parameters
        if common_config.get('trust_remote_code', False):
            cmd.append('--trust-remote-code')
            
        print(f"Starting SGLang server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.host = host
        self.port = port
        
    def is_ready(self) -> bool:
        """Check if SGLang server is ready"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False


class VLLMManager(EngineManager):
    """Manager for vLLM engine"""
    
    def start(self) -> None:
        """Start vLLM server"""
        server_config = self.config['server']
        vllm_config = server_config.get('vllm', {})
        common_config = server_config.get('common', {})
        
        model = server_config['model']
        host = vllm_config.get('host', '127.0.0.1')
        port = vllm_config.get('port', 8000)
        
        cmd = [
            'python3', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', model,
            '--host', host,
            '--port', str(port),
        ]
        
        # Add vLLM-specific parameters
        if 'tensor_parallel_size' in vllm_config:
            cmd.extend(['--tensor-parallel-size', str(vllm_config['tensor_parallel_size'])])
        if 'gpu_memory_utilization' in vllm_config:
            cmd.extend(['--gpu-memory-utilization', str(vllm_config['gpu_memory_utilization'])])
        if 'max_model_len' in vllm_config:
            cmd.extend(['--max-model-len', str(vllm_config['max_model_len'])])
            
        # Add common parameters
        if common_config.get('trust_remote_code', False):
            cmd.append('--trust-remote-code')
            
        print(f"Starting vLLM server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.host = host
        self.port = port
        
    def is_ready(self) -> bool:
        """Check if vLLM server is ready"""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False


def get_engine_manager(engine_name: str, config: Dict[str, Any]) -> EngineManager:
    """Factory function to get the appropriate engine manager"""
    managers = {
        'sglang': SGLangManager,
        'vllm': VLLMManager,
    }
    
    manager_class = managers.get(engine_name.lower())
    if not manager_class:
        raise ValueError(f"Unsupported engine: {engine_name}. Supported engines: {list(managers.keys())}")
    
    return manager_class(config)


def run_benchmark(config: Dict[str, Any]) -> int:
    """Run the benchmark with the specified configuration"""
    benchmark_config = config['benchmark']
    
    # Build benchmark command
    script = benchmark_config['script']
    cmd = ['python3', '-m', script]
    
    # Add backend
    cmd.extend(['--backend', benchmark_config['backend']])
    
    # Add benchmark parameters
    params = benchmark_config.get('params', {})
    for key, value in params.items():
        param_name = key.replace('_', '-')
        cmd.extend([f'--{param_name}', str(value)])
    
    # Add additional options
    options = benchmark_config.get('options', {})
    for key, value in options.items():
        param_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{param_name}')
        else:
            cmd.extend([f'--{param_name}', str(value)])
    
    print(f"\nRunning benchmark: {' '.join(cmd)}\n")
    
    # Run benchmark
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='LLM Engine Benchmark Runner')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--skip-server', action='store_true', help='Skip starting the server (use existing instance)')
    parser.add_argument('--keep-server', action='store_true', help='Keep server running after benchmark')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    print(f"LLM Engine: {config['llm_engine']}")
    print(f"Model: {config['server']['model']}\n")
    
    manager = None
    
    try:
        if not args.skip_server:
            # Get engine manager
            engine_name = config['llm_engine']
            manager = get_engine_manager(engine_name, config)
            
            # Start server
            manager.start()
            
            # Wait for server to be ready
            execution_config = config.get('execution', {})
            timeout = execution_config.get('wait_for_ready_timeout', 300)
            if not manager.wait_for_ready(timeout):
                print("Failed to start server. Exiting.")
                return 1
            
            # Warmup (optional)
            warmup_requests = execution_config.get('warmup_requests', 0)
            if warmup_requests > 0:
                print(f"Running {warmup_requests} warmup requests...")
                time.sleep(5)  # Simple warmup delay
        
        # Run benchmark
        exit_code = run_benchmark(config)
        
        if exit_code == 0:
            print("\nBenchmark completed successfully!")
        else:
            print(f"\nBenchmark failed with exit code: {exit_code}")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        if manager and not args.keep_server:
            execution_config = config.get('execution', {})
            if execution_config.get('cleanup_on_exit', True):
                manager.stop()


if __name__ == '__main__':
    sys.exit(main())
