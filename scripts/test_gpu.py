# ...existing code...
"""Simple GPU test script for Windows + PyTorch.

Usage (PowerShell):
  python .\test_gpu.py

This will:
  - detect whether CUDA is available
  - print device name and memory
  - run a small matrix-multiply benchmark on GPU (if available)

The script intentionally avoids forcing CUDA installs. If PyTorch is not
installed, it will explain how to install it.
"""
import sys
import time
import argparse


def main():
	parser = argparse.ArgumentParser(description="GPU test + micro-benchmark using PyTorch")
	parser.add_argument("--size", "-n", type=int, default=4096, help="matrix size (n for n x n)")
	parser.add_argument("--reps", "-r", type=int, default=10, help="number of repetitions")
	args = parser.parse_args()

	try:
		import torch
	except Exception as e:
		print("PyTorch import failed:", e)
		print("\nIf you don't have PyTorch installed, install a matching CUDA-enabled build. Example (conda):")
		print("  conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia")
		sys.exit(1)

	print("PyTorch version:", torch.__version__)
	cuda_available = torch.cuda.is_available()
	print("CUDA available:", cuda_available)

	if not cuda_available:
		print("No CUDA device detected. If you expect an NVIDIA GPU, ensure drivers and a CUDA-enabled PyTorch build are installed.")
		return

	dev = torch.device("cuda:0")
	try:
		name = torch.cuda.get_device_name(0)
	except Exception:
		name = "<unknown>"
	props = torch.cuda.get_device_properties(0)

	print(f"Device 0: {name}")
	print(f"  Compute capability: {props.major}.{props.minor}")
	print(f"  Total memory (MB): {props.total_memory / 1024**2:.1f}")

	n = args.size
	reps = args.reps

	print(f"\nRunning matrix multiply benchmark: {reps} reps of {n}x{n} (float32) using torch.cuda.Event timing")
	try:
		# allocate matrices
		a = torch.randn((n, n), device=dev, dtype=torch.float32)
		b = torch.randn((n, n), device=dev, dtype=torch.float32)
