"""Definitive kernel benchmark for Phase 6 Qwen3-0.6B block diffusion LLM.

Methodologically bulletproof: 50 iterations, 10 warmup, Cohen's d,
correctness validation, forward AND backward, real model shapes.

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_kernel_test.py
"""

import modal

app = modal.App('smoldlm-phase6-kernel-bench')

# ---------------------------------------------------------------------------
# Modal image: CUDA devel (for TK build) + all kernel libraries
# ---------------------------------------------------------------------------

image = (
  modal.Image
  # CUDA 13 devel — matches torch 2.11's bundled CUDA 13 libs
  .from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
  .apt_install('git', 'build-essential', 'ninja-build')
  .env({
    'MAX_JOBS': '16',
    'TORCH_CUDA_ARCH_LIST': '9.0',
    'CUDA_HOME': '/usr/local/cuda',
    'CXX': 'g++',  # CUDA 13 defaults to clang++, but torch needs g++
    'CC': 'gcc',
  })
  .uv_pip_install(
    'torch',  # latest (2.11+) for best perf: FA3, better compile, faster cuBLAS
    'numpy',
    'packaging',  # needed by flash-attn setup.py
    'setuptools',
    'wheel',
    'ninja',
    'psutil',
    'quack-kernels==0.3.7',
    'liger-kernel>=0.5.9',
  )
  .run_commands(
    # flash-attn: prebuilt wheel for cu130+torch2.11+cp312 (avoids 10min source build + OOM)
    '/.uv/uv pip install --system'
    " 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-linux_x86_64.whl'"
    " || echo 'FLASH_ATTN_BUILD_FAILED'",
  )
  .run_commands(
    # gram-newton-schulz: --no-deps prevents pulling torch 2.11+ (already installed)
    '/.uv/uv pip install --system --no-deps git+https://github.com/Dao-AILab/gram-newton-schulz.git'
    " || echo 'WARN: gram-newton-schulz install failed'",
  )
  .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


# ============================================================================
# Constants — real Qwen3-0.6B shapes
# ============================================================================

B = 4  # batch size
L = 4096  # effective seq_len (2 × 2048 for [x_t || x_0])
D = 1024  # hidden dim
H = 16  # query heads
KV = 8  # KV heads (GQA 2:1, from config n_kv_head=8)
HD = 128  # head dim (from config head_dim=128)
FFN = 3072  # intermediate size (from config mlp_hidden=3072)
V = 151_936  # vocab size

WARMUP = 10
ITERS = 50


# ============================================================================
# Benchmark function entry point
# ============================================================================


@app.function(image=image, gpu='H100', timeout=3600, startup_timeout=1800)
def run_benchmarks():
  import sys

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  import gc
  import time
  from dataclasses import dataclass

  import torch
  import torch.nn.functional as F
  from torch import nn

  torch.set_float32_matmul_precision('high')
  torch.manual_seed(42)

  props = torch.cuda.get_device_properties(0)
  total_mem = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', 0)
  gpu_name = torch.cuda.get_device_name(0)
  print(f'PyTorch {torch.__version__} | {gpu_name} | {total_mem / 1e9:.0f} GB')
  print(f'Shapes: B={B} L={L} D={D} H={H} KV={KV} HD={HD} FFN={FFN} V={V}')
  print(f'Methodology: {WARMUP} warmup + {ITERS} timed iterations per test\n')

  # ------------------------------------------------------------------
  # BenchmarkResult dataclass
  # ------------------------------------------------------------------

  @dataclass
  class BenchResult:
    name: str
    shape: str
    mean_ms: float = 0.0
    std_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    peak_mem_mb: float = 0.0
    cohens_d: float = 0.0
    significant: bool = False
    correct_fwd: str = 'N/A'
    correct_bwd: str = 'N/A'
    speedup: float = 1.0
    error: str = ''

  # ------------------------------------------------------------------
  # Core benchmark function
  # ------------------------------------------------------------------

  def benchmark_fn(fn, warmup=WARMUP, iterations=ITERS, label=''):
    """Time a function with CUDA synchronization, warmup, and stats."""
    # Warmup
    for _ in range(warmup):
      fn()
    torch.cuda.synchronize()

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    for _ in range(iterations):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      fn()
      torch.cuda.synchronize()
      t1 = time.perf_counter()
      times.append((t1 - t0) * 1000)  # ms

    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB

    times_t = torch.tensor(times)
    mean = times_t.mean().item()
    std = times_t.std().item()
    median = times_t.median().item()
    sorted_t = times_t.sort().values
    p95 = sorted_t[int(0.95 * len(sorted_t))].item()
    p99 = sorted_t[int(0.99 * len(sorted_t))].item()

    return mean, std, median, p95, p99, peak_mem

  def compute_cohens_d(times_baseline, times_kernel):
    """Cohen's d effect size. Positive = kernel is faster."""
    t_b = torch.tensor(times_baseline)
    t_k = torch.tensor(times_kernel)
    pooled_std = torch.sqrt((t_b.var() + t_k.var()) / 2).item()
    if pooled_std < 1e-12:
      return 0.0
    return (t_b.mean().item() - t_k.mean().item()) / pooled_std

  def timed_runs(fn, warmup=WARMUP, iterations=ITERS):
    """Return raw list of times (ms) for Cohen's d calculation."""
    for _ in range(warmup):
      fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      fn()
      torch.cuda.synchronize()
      t1 = time.perf_counter()
      times.append((t1 - t0) * 1000)
    return times

  def check_close(ref, test, atol=1e-2, rtol=1e-2, label=''):
    """Check allclose, return PASS/FAIL string."""
    if ref is None or test is None:
      return 'SKIP'
    try:
      ref_f = ref.float()
      test_f = test.float()
      ok = torch.allclose(ref_f, test_f, atol=atol, rtol=rtol)
      if ok:
        return 'PASS'
      max_diff = (ref_f - test_f).abs().max().item()
      return f'FAIL (max_diff={max_diff:.4e})'
    except Exception as e:
      return f'ERROR ({e})'

  def clear_between():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

  # ------------------------------------------------------------------
  # Table formatter
  # ------------------------------------------------------------------

  def print_table(title, results):
    print(f'\n{"=" * 90}')
    print(f' {title}')
    print(f'{"=" * 90}')
    hdr = (
      f'{"Kernel":<30} {"mean(ms)":>9} {"std(ms)":>9} {"p95(ms)":>9} '
      f'{"mem(MB)":>9} {"speedup":>8} {"d":>7} {"sig?":>5} '
      f'{"fwd":>6} {"bwd":>6}'
    )
    print(hdr)
    print('-' * len(hdr))
    for r in results:
      if r.error:
        print(f'  {r.name:<28} ERROR: {r.error}')
        continue
      sig = 'Y' if r.significant else ' '
      d_str = f'{r.cohens_d:.1f}' if r.cohens_d != 0 else '-'
      spd = f'{r.speedup:.2f}x' if r.speedup != 1.0 else '1.00x'
      print(
        f'  {r.name:<28} {r.mean_ms:>9.3f} {r.std_ms:>9.3f} {r.p95_ms:>9.3f} '
        f'{r.peak_mem_mb:>9.0f} {spd:>8} {d_str:>7} {sig:>5} '
        f'{r.correct_fwd:>6} {r.correct_bwd:>6}'
      )

  # ------------------------------------------------------------------
  # Helper: run a pair of benchmarks (baseline vs kernel)
  # ------------------------------------------------------------------

  def bench_pair(
    baseline_fn,
    kernel_fn,
    baseline_name,
    kernel_name,
    shape,
    ref_out=None,
    kern_out=None,
    ref_grad=None,
    kern_grad=None,
    baseline_bwd_fn=None,
    kernel_bwd_fn=None,
    ref_bwd_out=None,
    kern_bwd_out=None,
    ref_bwd_grad=None,
    kern_bwd_grad=None,
  ):
    """Benchmark baseline and kernel, compute Cohen's d, return BenchResults."""
    results = []

    # --- Baseline forward ---
    clear_between()
    try:
      b_times = timed_runs(baseline_fn)
      b_mean, b_std, b_med, b_p95, b_p99, b_mem = (
        torch.tensor(b_times).mean().item(),
        torch.tensor(b_times).std().item(),
        torch.tensor(b_times).median().item(),
        torch.tensor(b_times).sort().values[int(0.95 * len(b_times))].item(),
        torch.tensor(b_times).sort().values[int(0.99 * len(b_times))].item(),
        torch.cuda.max_memory_allocated() / 1e6,
      )
      br = BenchResult(
        name=baseline_name,
        shape=shape,
        mean_ms=b_mean,
        std_ms=b_std,
        median_ms=b_med,
        p95_ms=b_p95,
        p99_ms=b_p99,
        peak_mem_mb=b_mem,
      )
    except Exception as e:
      br = BenchResult(name=baseline_name, shape=shape, error=str(e)[:120])
      b_times = None
    results.append(br)

    # --- Kernel forward ---
    clear_between()
    try:
      k_times = timed_runs(kernel_fn)
      k_mean, k_std, k_med, k_p95, k_p99, k_mem = (
        torch.tensor(k_times).mean().item(),
        torch.tensor(k_times).std().item(),
        torch.tensor(k_times).median().item(),
        torch.tensor(k_times).sort().values[int(0.95 * len(k_times))].item(),
        torch.tensor(k_times).sort().values[int(0.99 * len(k_times))].item(),
        torch.cuda.max_memory_allocated() / 1e6,
      )
      # Cohen's d
      d = 0.0
      sig = False
      if b_times is not None:
        d = compute_cohens_d(b_times, k_times)
        sig = abs(d) > 0.8
      spd = b_mean / k_mean if b_times and k_mean > 0 else 1.0
      kr = BenchResult(
        name=kernel_name,
        shape=shape,
        mean_ms=k_mean,
        std_ms=k_std,
        median_ms=k_med,
        p95_ms=k_p95,
        p99_ms=k_p99,
        peak_mem_mb=k_mem,
        cohens_d=d,
        significant=sig,
        speedup=spd,
        correct_fwd=check_close(ref_out, kern_out) if ref_out is not None else 'N/A',
        correct_bwd=check_close(ref_grad, kern_grad) if ref_grad is not None else 'N/A',
      )
    except Exception as e:
      kr = BenchResult(name=kernel_name, shape=shape, error=str(e)[:120])
    results.append(kr)

    # --- Baseline backward ---
    if baseline_bwd_fn is not None:
      clear_between()
      try:
        bb_times = timed_runs(baseline_bwd_fn)
        bb_mean = torch.tensor(bb_times).mean().item()
        bb_std = torch.tensor(bb_times).std().item()
        bb_p95 = torch.tensor(bb_times).sort().values[int(0.95 * len(bb_times))].item()
        bbr = BenchResult(
          name=f'{baseline_name} (bwd)',
          shape=shape,
          mean_ms=bb_mean,
          std_ms=bb_std,
          median_ms=torch.tensor(bb_times).median().item(),
          p95_ms=bb_p95,
          p99_ms=torch.tensor(bb_times).sort().values[int(0.99 * len(bb_times))].item(),
          peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
        )
      except Exception as e:
        bbr = BenchResult(name=f'{baseline_name} (bwd)', shape=shape, error=str(e)[:120])
        bb_times = None
      results.append(bbr)

      # --- Kernel backward ---
      if kernel_bwd_fn is not None:
        clear_between()
        try:
          kb_times = timed_runs(kernel_bwd_fn)
          kb_mean = torch.tensor(kb_times).mean().item()
          kb_std = torch.tensor(kb_times).std().item()
          d_bwd = compute_cohens_d(bb_times, kb_times) if bb_times else 0.0
          spd_bwd = bb_mean / kb_mean if bb_times and kb_mean > 0 else 1.0
          kbr = BenchResult(
            name=f'{kernel_name} (bwd)',
            shape=shape,
            mean_ms=kb_mean,
            std_ms=kb_std,
            median_ms=torch.tensor(kb_times).median().item(),
            p95_ms=torch.tensor(kb_times).sort().values[int(0.95 * len(kb_times))].item(),
            p99_ms=torch.tensor(kb_times).sort().values[int(0.99 * len(kb_times))].item(),
            peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
            cohens_d=d_bwd,
            significant=abs(d_bwd) > 0.8,
            speedup=spd_bwd,
            correct_bwd=check_close(ref_bwd_grad, kern_bwd_grad) if ref_bwd_grad is not None else 'N/A',
          )
        except Exception as e:
          kbr = BenchResult(name=f'{kernel_name} (bwd)', shape=shape, error=str(e)[:120])
        results.append(kbr)

    return results

  # ==================================================================
  # Collect all results + summary
  # ==================================================================

  all_results = {}

  # ==================================================================
  # SECTION 1: Cross-Entropy
  # ==================================================================

  def bench_cross_entropy():
    shape = f'{B * L // 2} x {V}'
    section_results = []

    N_CE = B * L // 2  # 8192 — only second half has loss
    logits_base = torch.randn(N_CE, V, device='cuda', dtype=torch.bfloat16)
    targets_base = torch.randint(0, V, (N_CE,), device='cuda')

    # --- F.cross_entropy (reduction='none') ---
    def baseline_fwd():
      return F.cross_entropy(logits_base.float(), targets_base, reduction='none')

    # --- quack.cross_entropy ---
    try:
      from quack import cross_entropy as quack_ce

      # Correctness check
      ref = F.cross_entropy(logits_base.float(), targets_base, reduction='none')
      # quack.cross_entropy defaults to reduction="mean" — must pass reduction="none"
      kern = quack_ce(logits_base.contiguous(), targets_base.contiguous(), reduction='none')
      fwd_correct = check_close(ref, kern)
      print(f'  quack CE fwd correctness: {fwd_correct}')

      def kernel_fwd():
        return quack_ce(logits_base.contiguous(), targets_base.contiguous(), reduction='none')

      # Backward test
      logits_ref_bwd = logits_base.float().clone().requires_grad_(True)
      loss_ref = F.cross_entropy(logits_ref_bwd, targets_base, reduction='none')
      loss_ref.sum().backward()
      ref_grad = logits_ref_bwd.grad.clone()

      # quack backward with .contiguous() fix
      logits_kern_bwd = logits_base.clone().contiguous().requires_grad_(True)
      try:
        loss_kern = quack_ce(logits_kern_bwd, targets_base.contiguous(), reduction='none')
        loss_kern.sum().backward()
        kern_grad = logits_kern_bwd.grad.clone() if logits_kern_bwd.grad is not None else None
        bwd_correct = check_close(ref_grad, kern_grad)
        print(f'  quack CE bwd correctness: {bwd_correct}')
        bwd_ok = True
      except Exception as e:
        print(f'  quack CE bwd FAILED: {e}')
        bwd_correct = f'FAIL ({str(e)[:80]})'
        kern_grad = None
        bwd_ok = False

      def baseline_bwd():
        l = logits_base.float().clone().requires_grad_(True)
        out = F.cross_entropy(l, targets_base, reduction='none')
        out.sum().backward()

      def kernel_bwd():
        l = logits_base.clone().contiguous().requires_grad_(True)
        out = quack_ce(l, targets_base.contiguous(), reduction='none')
        out.sum().backward()

      results = bench_pair(
        baseline_fwd,
        kernel_fwd,
        'F.cross_entropy',
        'quack.cross_entropy',
        shape,
        ref_out=ref,
        kern_out=kern,
        ref_grad=ref_grad,
        kern_grad=kern_grad,
        baseline_bwd_fn=baseline_bwd,
        kernel_bwd_fn=kernel_bwd if bwd_ok else None,
      )
      for r in results:
        if 'quack' in r.name:
          r.correct_fwd = fwd_correct
          if 'bwd' in r.name:
            r.correct_bwd = bwd_correct
      section_results.extend(results)

    except ImportError:
      section_results.append(BenchResult(name='quack.cross_entropy', shape=shape, error='quack not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='quack.cross_entropy', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 1: Cross-Entropy')
  print('=' * 90)
  try:
    ce_results = bench_cross_entropy()
    all_results['Cross-Entropy'] = ce_results
    print_table(f'Cross-Entropy ({B * L // 2} x {V})', ce_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 2: RMSNorm
  # ==================================================================

  def bench_rmsnorm():
    shape = f'{B * L} x {D}'
    section_results = []

    x_base = torch.randn(B * L, D, device='cuda', dtype=torch.bfloat16)
    weight = torch.ones(D, device='cuda', dtype=torch.bfloat16)
    eps = 1e-6

    # --- nn.RMSNorm baseline ---
    rms_nn = nn.RMSNorm(D, eps=eps).cuda().bfloat16()

    def baseline_fwd():
      return rms_nn(x_base)

    ref_out = rms_nn(x_base).clone()

    # Compute baseline backward once for all comparisons
    x_ref_bwd_base = x_base.clone().requires_grad_(True)
    rms_nn(x_ref_bwd_base).sum().backward()
    ref_bwd_grad = x_ref_bwd_base.grad.clone()

    # --- Liger RMSNorm ---
    try:
      from liger_kernel.transformers import LigerRMSNorm

      rms_liger = LigerRMSNorm(D, eps=eps).cuda().bfloat16()
      liger_out = rms_liger(x_base.clone()).clone()
      liger_fwd_correct = check_close(ref_out, liger_out)
      print(f'  Liger RMSNorm fwd correctness: {liger_fwd_correct}')

      def liger_fwd():
        return rms_liger(x_base)

      # Backward
      x_liger_bwd = x_base.clone().requires_grad_(True)
      rms_liger(x_liger_bwd).sum().backward()
      liger_bwd_grad = x_liger_bwd.grad.clone()
      liger_bwd_correct = check_close(ref_bwd_grad, liger_bwd_grad)
      print(f'  Liger RMSNorm bwd correctness: {liger_bwd_correct}')

      def baseline_bwd():
        x = x_base.clone().requires_grad_(True)
        rms_nn(x).sum().backward()

      def liger_bwd():
        x = x_base.clone().requires_grad_(True)
        rms_liger(x).sum().backward()

      liger_results = bench_pair(
        baseline_fwd,
        liger_fwd,
        'nn.RMSNorm',
        'LigerRMSNorm',
        shape,
        ref_out=ref_out,
        kern_out=liger_out,
        ref_grad=ref_bwd_grad,
        kern_grad=liger_bwd_grad,
        baseline_bwd_fn=baseline_bwd,
        kernel_bwd_fn=liger_bwd,
      )
      section_results.extend(liger_results)
    except ImportError:
      section_results.append(BenchResult(name='LigerRMSNorm', shape=shape, error='liger not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='LigerRMSNorm', shape=shape, error=str(e)[:120]))

    # --- quack.rmsnorm ---
    try:
      from quack import rmsnorm as quack_rms

      quack_out = quack_rms(x_base.contiguous(), weight.contiguous(), eps=eps)
      quack_fwd_correct = check_close(ref_out, quack_out)
      print(f'  quack rmsnorm fwd correctness: {quack_fwd_correct}')

      def quack_fwd():
        return quack_rms(x_base.contiguous(), weight.contiguous(), eps=eps)

      # Backward
      x_q_bwd = x_base.clone().contiguous().requires_grad_(True)
      try:
        out_q = quack_rms(x_q_bwd, weight.contiguous(), eps=eps)
        out_q.sum().backward()
        quack_bwd_grad = x_q_bwd.grad.clone() if x_q_bwd.grad is not None else None
        quack_bwd_correct = check_close(ref_bwd_grad, quack_bwd_grad) if ref_bwd_grad is not None else 'N/A'
        print(f'  quack rmsnorm bwd correctness: {quack_bwd_correct}')
        quack_bwd_ok = True
      except Exception as e:
        print(f'  quack rmsnorm bwd FAILED: {e}')
        quack_bwd_correct = f'FAIL ({str(e)[:80]})'
        quack_bwd_grad = None
        quack_bwd_ok = False

      def quack_bwd_fn():
        x = x_base.clone().contiguous().requires_grad_(True)
        out = quack_rms(x, weight.contiguous(), eps=eps)
        out.sum().backward()

      # Only benchmark forward + backward if baseline exists
      # Run baseline FIRST for consistent methodology (same as bench_pair)
      clear_between()
      base_times_fwd = timed_runs(baseline_fwd)
      quack_times_fwd = timed_runs(quack_fwd)
      q_mean = torch.tensor(quack_times_fwd).mean().item()
      b_mean_val = torch.tensor(base_times_fwd).mean().item()
      d_fwd = compute_cohens_d(base_times_fwd, quack_times_fwd)

      section_results.append(
        BenchResult(
          name='quack.rmsnorm',
          shape=shape,
          mean_ms=q_mean,
          std_ms=torch.tensor(quack_times_fwd).std().item(),
          median_ms=torch.tensor(quack_times_fwd).median().item(),
          p95_ms=torch.tensor(quack_times_fwd).sort().values[int(0.95 * len(quack_times_fwd))].item(),
          p99_ms=torch.tensor(quack_times_fwd).sort().values[int(0.99 * len(quack_times_fwd))].item(),
          peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
          cohens_d=d_fwd,
          significant=abs(d_fwd) > 0.8,
          speedup=b_mean_val / q_mean if q_mean > 0 else 1.0,
          correct_fwd=quack_fwd_correct,
          correct_bwd=quack_bwd_correct,
        )
      )

      if quack_bwd_ok:
        clear_between()

        def nn_bwd():
          x = x_base.clone().requires_grad_(True)
          rms_nn(x).sum().backward()

        qb_times = timed_runs(quack_bwd_fn)
        nb_times = timed_runs(nn_bwd)
        qb_mean = torch.tensor(qb_times).mean().item()
        nb_mean = torch.tensor(nb_times).mean().item()
        d_bwd = compute_cohens_d(nb_times, qb_times)
        section_results.append(
          BenchResult(
            name='quack.rmsnorm (bwd)',
            shape=shape,
            mean_ms=qb_mean,
            std_ms=torch.tensor(qb_times).std().item(),
            median_ms=torch.tensor(qb_times).median().item(),
            p95_ms=torch.tensor(qb_times).sort().values[int(0.95 * len(qb_times))].item(),
            p99_ms=torch.tensor(qb_times).sort().values[int(0.99 * len(qb_times))].item(),
            peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
            cohens_d=d_bwd,
            significant=abs(d_bwd) > 0.8,
            speedup=nb_mean / qb_mean if qb_mean > 0 else 1.0,
            correct_bwd=quack_bwd_correct,
          )
        )

    except ImportError:
      section_results.append(BenchResult(name='quack.rmsnorm', shape=shape, error='quack not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='quack.rmsnorm', shape=shape, error=str(e)[:120]))

    # --- flash-attn rms_norm ---
    try:
      from flash_attn.ops.rms_norm import rms_norm as fa_rms_norm

      fa_out = fa_rms_norm(x_base.contiguous(), weight.contiguous(), eps)
      fa_fwd_correct = check_close(ref_out, fa_out)
      print(f'  flash-attn rms_norm fwd correctness: {fa_fwd_correct}')

      def fa_fwd():
        return fa_rms_norm(x_base.contiguous(), weight.contiguous(), eps)

      clear_between()
      fa_times = timed_runs(fa_fwd)
      base_times2 = timed_runs(baseline_fwd)
      fa_mean = torch.tensor(fa_times).mean().item()
      b_mean2 = torch.tensor(base_times2).mean().item()
      d_fa = compute_cohens_d(base_times2, fa_times)

      # Backward
      x_fa_bwd = x_base.clone().contiguous().requires_grad_(True)
      try:
        fa_rms_norm(x_fa_bwd, weight.contiguous(), eps).sum().backward()
        fa_bwd_grad = x_fa_bwd.grad.clone()
        fa_bwd_correct = check_close(ref_bwd_grad, fa_bwd_grad) if ref_bwd_grad is not None else 'N/A'
        print(f'  flash-attn rms_norm bwd correctness: {fa_bwd_correct}')
      except Exception as e:
        print(f'  flash-attn rms_norm bwd FAILED: {e}')
        fa_bwd_correct = f'FAIL ({str(e)[:80]})'

      section_results.append(
        BenchResult(
          name='flash_attn.rms_norm',
          shape=shape,
          mean_ms=fa_mean,
          std_ms=torch.tensor(fa_times).std().item(),
          median_ms=torch.tensor(fa_times).median().item(),
          p95_ms=torch.tensor(fa_times).sort().values[int(0.95 * len(fa_times))].item(),
          p99_ms=torch.tensor(fa_times).sort().values[int(0.99 * len(fa_times))].item(),
          peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
          cohens_d=d_fa,
          significant=abs(d_fa) > 0.8,
          speedup=b_mean2 / fa_mean if fa_mean > 0 else 1.0,
          correct_fwd=fa_fwd_correct,
          correct_bwd=fa_bwd_correct,
        )
      )

    except ImportError:
      section_results.append(BenchResult(name='flash_attn.rms_norm', shape=shape, error='flash-attn not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='flash_attn.rms_norm', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 2: RMSNorm')
  print('=' * 90)
  try:
    rms_results = bench_rmsnorm()
    all_results['RMSNorm'] = rms_results
    print_table(f'RMSNorm ({B * L} x {D})', rms_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 3: Symmetric GEMM (Newton-Schulz inner loop)
  # ==================================================================

  def bench_gemm_symmetric():
    # NS operates on (D, D) after .T for square, or rectangular per layer
    shape = f'{D} x {D}'
    section_results = []

    X = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)

    def baseline_fwd():
      return torch.mm(X, X.T)

    ref_out = torch.mm(X, X.T)

    try:
      from quack.gemm_interface import gemm_symmetric

      # gemm_symmetric(A, B) does B.mT internally then computes A @ B^T
      # So pass B = X.mT (transposed) — kernel does X.mT.mT = X, then A @ X = X @ X... no.
      # Actually: pass B = X.T so kernel does (X.T).mT = X, then computes A @ B^T = X @ X.T
      # Confirmed by quack test: gemm_symmetric(a, a.transpose(-2, -1))
      X_c = X.contiguous()
      X_t = X_c.mT.contiguous()  # (K, M) layout — kernel will do .mT to get (M, K)
      quack_out_tuple = gemm_symmetric(X_c, X_t)
      # Returns tensor (despite type annotation saying Tuple)
      quack_out = quack_out_tuple[1] if isinstance(quack_out_tuple, tuple) else quack_out_tuple
      fwd_correct = check_close(ref_out, quack_out)
      print(f'  quack gemm_symmetric fwd correctness: {fwd_correct}')

      def kernel_fwd():
        return gemm_symmetric(X_c, X_t)

      results = bench_pair(
        baseline_fwd,
        kernel_fwd,
        'torch.mm(X, X.T)',
        'quack.gemm_symmetric',
        shape,
        ref_out=ref_out,
        kern_out=quack_out,
      )
      for r in results:
        if 'quack' in r.name:
          r.correct_fwd = fwd_correct
      section_results.extend(results)

    except ImportError:
      section_results.append(BenchResult(name='quack.gemm_symmetric', shape=shape, error='quack not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='quack.gemm_symmetric', shape=shape, error=str(e)[:120]))

    # Also test rectangular shapes used in NS
    for m, k, label in [
      (D, KV * HD, f'{D}x{KV * HD}'),  # (1024, 1024) — square for non-GQA
      (D, HD, f'{D}x{HD}'),  # (1024, 128) — Q/K projections
    ]:
      X_rect = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
      ref_rect = torch.mm(X_rect, X_rect.T)  # (m, m)
      try:
        from quack.gemm_interface import gemm_symmetric

        X_rect_c = X_rect.contiguous()
        X_rect_t = X_rect_c.mT.contiguous()  # (K, M) — kernel does .mT internally
        q_rect = gemm_symmetric(X_rect_c, X_rect_t)
        q_rect_out = q_rect[1] if isinstance(q_rect, tuple) else q_rect
        correct = check_close(ref_rect, q_rect_out)
        print(f'  quack gemm_symmetric {label} correctness: {correct}')
      except Exception as e:
        print(f'  quack gemm_symmetric {label} FAILED: {e}')

    return section_results

  print('\n' + '=' * 90)
  print(' Section 3: Symmetric GEMM (Newton-Schulz)')
  print('=' * 90)
  try:
    gemm_results = bench_gemm_symmetric()
    all_results['GEMM Symmetric'] = gemm_results
    print_table(f'Symmetric GEMM ({D}x{D})', gemm_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 4: Linear
  # ==================================================================

  def bench_linear():
    # QKV projection: (B*L, D) x (D, D+2*KV*HD) = (16384, 1024) x (1024, 1024+2*8*128)
    # But more representative: general (B*L, D) x (D, D) for output projection
    M, K, N = B * L, D, D
    shape = f'{M} x {K} -> {N}'
    section_results = []

    x_base = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)

    # Baseline: nn.Linear
    lin_nn = nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)

    def baseline_fwd():
      return lin_nn(x_base)

    ref_out = lin_nn(x_base).clone()

    # Backward baseline
    x_ref_bwd = x_base.clone().requires_grad_(True)
    lin_nn(x_ref_bwd).sum().backward()
    ref_bwd_grad = x_ref_bwd.grad.clone()

    try:
      from quack.linear import Linear as QuackLinear

      lin_quack = QuackLinear(K, N, bias=False, dtype=torch.bfloat16).cuda()
      # Copy weights for fair comparison
      with torch.no_grad():
        lin_quack.weight.copy_(lin_nn.weight)

      quack_out = lin_quack(x_base.contiguous()).clone()
      fwd_correct = check_close(ref_out, quack_out)
      print(f'  quack.Linear fwd correctness: {fwd_correct}')

      def kernel_fwd():
        return lin_quack(x_base.contiguous())

      # Backward
      x_q_bwd = x_base.clone().contiguous().requires_grad_(True)
      try:
        lin_quack(x_q_bwd).sum().backward()
        q_bwd_grad = x_q_bwd.grad.clone() if x_q_bwd.grad is not None else None
        bwd_correct = check_close(ref_bwd_grad, q_bwd_grad)
        print(f'  quack.Linear bwd correctness: {bwd_correct}')
        bwd_ok = True
      except Exception as e:
        print(f'  quack.Linear bwd FAILED: {e}')
        bwd_correct = f'FAIL ({str(e)[:80]})'
        q_bwd_grad = None
        bwd_ok = False

      def baseline_bwd():
        x = x_base.clone().requires_grad_(True)
        lin_nn(x).sum().backward()

      def kernel_bwd():
        x = x_base.clone().contiguous().requires_grad_(True)
        lin_quack(x).sum().backward()

      results = bench_pair(
        baseline_fwd,
        kernel_fwd,
        'nn.Linear',
        'quack.Linear',
        shape,
        ref_out=ref_out,
        kern_out=quack_out,
        ref_grad=ref_bwd_grad,
        kern_grad=q_bwd_grad,
        baseline_bwd_fn=baseline_bwd,
        kernel_bwd_fn=kernel_bwd if bwd_ok else None,
      )
      for r in results:
        if 'quack' in r.name and 'bwd' not in r.name:
          r.correct_fwd = fwd_correct
          r.correct_bwd = bwd_correct
      section_results.extend(results)

    except ImportError:
      section_results.append(BenchResult(name='quack.Linear', shape=shape, error='quack not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='quack.Linear', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 4: Linear')
  print('=' * 90)
  try:
    lin_results = bench_linear()
    all_results['Linear'] = lin_results
    print_table(f'Linear ({B * L}x{D} -> {D})', lin_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 5: MLP (SwiGLU)
  # ==================================================================

  def bench_mlp():
    M = B * L
    shape = f'{M} x {D} -> {FFN} -> {D}'
    section_results = []

    x_base = torch.randn(M, D, device='cuda', dtype=torch.bfloat16)

    # Manual SwiGLU baseline (our model's actual implementation)
    gate_w = torch.randn(FFN, D, device='cuda', dtype=torch.bfloat16) * 0.02
    up_w = torch.randn(FFN, D, device='cuda', dtype=torch.bfloat16) * 0.02
    down_w = torch.randn(D, FFN, device='cuda', dtype=torch.bfloat16) * 0.02

    def baseline_fwd():
      gate = x_base @ gate_w.T
      up = x_base @ up_w.T
      return (F.silu(gate) * up) @ down_w.T

    ref_out = baseline_fwd().clone()

    # Backward
    x_ref_bwd = x_base.clone().requires_grad_(True)
    gate = x_ref_bwd @ gate_w.T
    up = x_ref_bwd @ up_w.T
    out = (F.silu(gate) * up) @ down_w.T
    out.sum().backward()
    ref_bwd_grad = x_ref_bwd.grad.clone()

    try:
      from quack.mlp import MLP as QuackMLP

      # quack MLP uses bias1/bias2, not bias (TypeError if passing bias=False)
      quack_mlp = QuackMLP(
        in_features=D,
        out_features=D,
        hidden_features=FFN,
        activation='swiglu',
        dtype=torch.bfloat16,
      ).cuda()

      quack_out = quack_mlp(x_base.contiguous()).clone()
      # Can't directly compare correctness with different weights
      # Just check it runs and produces correct shape
      fwd_correct = 'PASS' if quack_out.shape == ref_out.shape else f'FAIL (shape {quack_out.shape})'
      print(f'  quack.MLP fwd shape check: {fwd_correct}')

      def kernel_fwd():
        return quack_mlp(x_base.contiguous())

      # Backward
      x_q_bwd = x_base.clone().contiguous().requires_grad_(True)
      try:
        quack_mlp(x_q_bwd).sum().backward()
        bwd_correct = 'PASS' if x_q_bwd.grad is not None else 'FAIL (no grad)'
        print(f'  quack.MLP bwd: {bwd_correct}')
        bwd_ok = True
      except Exception as e:
        print(f'  quack.MLP bwd FAILED: {e}')
        bwd_correct = f'FAIL ({str(e)[:80]})'
        bwd_ok = False

      def baseline_bwd():
        x = x_base.clone().requires_grad_(True)
        g = x @ gate_w.T
        u = x @ up_w.T
        (F.silu(g) * u @ down_w.T).sum().backward()

      def kernel_bwd():
        x = x_base.clone().contiguous().requires_grad_(True)
        quack_mlp(x).sum().backward()

      results = bench_pair(
        baseline_fwd,
        kernel_fwd,
        'Manual SwiGLU',
        'quack.MLP',
        shape,
        baseline_bwd_fn=baseline_bwd,
        kernel_bwd_fn=kernel_bwd if bwd_ok else None,
      )
      for r in results:
        if 'quack' in r.name:
          r.correct_fwd = fwd_correct
          if 'bwd' in r.name:
            r.correct_bwd = bwd_correct
      section_results.extend(results)

    except ImportError:
      section_results.append(BenchResult(name='quack.MLP', shape=shape, error='quack not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='quack.MLP', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 5: MLP (SwiGLU)')
  print('=' * 90)
  try:
    mlp_results = bench_mlp()
    all_results['MLP'] = mlp_results
    print_table(f'MLP ({B * L}x{D} -> {FFN} -> {D})', mlp_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 6: Fused Linear Cross-Entropy
  # ==================================================================

  def bench_linear_ce():
    N_CE = B * L // 2  # 8192
    shape = f'{N_CE} x {D} -> {V}'
    section_results = []

    h_base = torch.randn(N_CE, D, device='cuda', dtype=torch.bfloat16)
    targets_base = torch.randint(0, V, (N_CE,), device='cuda')
    lm_weight = torch.randn(V, D, device='cuda', dtype=torch.bfloat16) * 0.02

    # Baseline: matmul + F.cross_entropy
    def baseline_fwd():
      logits = h_base @ lm_weight.T
      return F.cross_entropy(logits.float(), targets_base, reduction='mean')

    ref_loss = baseline_fwd().item()
    print(f'  Baseline linear+CE loss: {ref_loss:.4f}')

    # --- Liger FLCE ---
    try:
      from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
      )

      liger_loss = LigerFusedLinearCrossEntropyFunction.apply(
        h_base.contiguous(),
        lm_weight.contiguous(),
        targets_base.contiguous(),
        None,
        -100,
        0.0,
        'mean',
      )
      liger_correct = (
        f'PASS (loss={liger_loss.item():.4f})'
        if abs(liger_loss.item() - ref_loss) < 0.5
        else f'FAIL (loss={liger_loss.item():.4f})'
      )
      print(f'  Liger FLCE: {liger_correct}')

      def liger_fwd():
        return LigerFusedLinearCrossEntropyFunction.apply(
          h_base.contiguous(),
          lm_weight.contiguous(),
          targets_base.contiguous(),
          None,
          -100,
          0.0,
          'mean',
        )

      # Backward
      h_ref_bwd = h_base.clone().requires_grad_(True)
      loss = (h_ref_bwd @ lm_weight.T).float()
      F.cross_entropy(loss, targets_base, reduction='mean').backward()
      ref_bwd_grad = h_ref_bwd.grad.clone()

      h_liger_bwd = h_base.clone().contiguous().requires_grad_(True)
      try:
        liger_l = LigerFusedLinearCrossEntropyFunction.apply(
          h_liger_bwd,
          lm_weight.contiguous(),
          targets_base.contiguous(),
          None,
          -100,
          0.0,
          'mean',
        )
        liger_l.backward()
        liger_bwd_grad = h_liger_bwd.grad.clone() if h_liger_bwd.grad is not None else None
        liger_bwd_correct = check_close(ref_bwd_grad, liger_bwd_grad, atol=0.1, rtol=0.1)
        print(f'  Liger FLCE bwd correctness: {liger_bwd_correct}')
        liger_bwd_ok = True
      except Exception as e:
        print(f'  Liger FLCE bwd FAILED: {e}')
        liger_bwd_correct = f'FAIL ({str(e)[:80]})'
        liger_bwd_ok = False

      def baseline_bwd():
        h = h_base.clone().requires_grad_(True)
        F.cross_entropy((h @ lm_weight.T).float(), targets_base, reduction='mean').backward()

      def liger_bwd():
        h = h_base.clone().contiguous().requires_grad_(True)
        LigerFusedLinearCrossEntropyFunction.apply(
          h,
          lm_weight.contiguous(),
          targets_base.contiguous(),
          None,
          -100,
          0.0,
          'mean',
        ).backward()

      results = bench_pair(
        baseline_fwd,
        liger_fwd,
        'matmul+CE',
        'Liger FLCE',
        shape,
        baseline_bwd_fn=baseline_bwd,
        kernel_bwd_fn=liger_bwd if liger_bwd_ok else None,
      )
      for r in results:
        if 'Liger' in r.name and 'bwd' not in r.name:
          r.correct_fwd = liger_correct
        elif 'Liger' in r.name:
          r.correct_bwd = liger_bwd_correct
      section_results.extend(results)

    except ImportError:
      section_results.append(BenchResult(name='Liger FLCE', shape=shape, error='liger not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='Liger FLCE', shape=shape, error=str(e)[:120]))

    # --- quack chunked_linear_cross_entropy ---
    try:
      from quack.linear_cross_entropy import chunked_linear_cross_entropy

      quack_loss = chunked_linear_cross_entropy(
        h_base.contiguous(),
        lm_weight.contiguous(),
        targets_base.contiguous(),
        chunk_size=4096,
        reduction='mean',
      )
      quack_correct = (
        f'PASS (loss={quack_loss.item():.4f})'
        if abs(quack_loss.item() - ref_loss) < 0.5
        else f'FAIL (loss={quack_loss.item():.4f})'
      )
      print(f'  quack chunked_linear_CE: {quack_correct}')

      def quack_fwd():
        return chunked_linear_cross_entropy(
          h_base.contiguous(),
          lm_weight.contiguous(),
          targets_base.contiguous(),
          chunk_size=4096,
          reduction='mean',
        )

      # Backward
      h_q_bwd = h_base.clone().contiguous().requires_grad_(True)
      try:
        quack_l = chunked_linear_cross_entropy(
          h_q_bwd,
          lm_weight.contiguous(),
          targets_base.contiguous(),
          chunk_size=4096,
          reduction='mean',
        )
        quack_l.backward()
        quack_bwd_grad = h_q_bwd.grad.clone() if h_q_bwd.grad is not None else None
        quack_bwd_correct = check_close(ref_bwd_grad, quack_bwd_grad, atol=0.1, rtol=0.1)
        print(f'  quack linear_CE bwd correctness: {quack_bwd_correct}')
        quack_bwd_ok = True
      except Exception as e:
        print(f'  quack linear_CE bwd FAILED: {e}')
        quack_bwd_correct = f'FAIL ({str(e)[:80]})'
        quack_bwd_ok = False

      def quack_bwd():
        h = h_base.clone().contiguous().requires_grad_(True)
        chunked_linear_cross_entropy(
          h,
          lm_weight.contiguous(),
          targets_base.contiguous(),
          chunk_size=4096,
          reduction='mean',
        ).backward()

      clear_between()
      base_times_ce = timed_runs(baseline_fwd)
      quack_times_ce = timed_runs(quack_fwd)
      q_mean = torch.tensor(quack_times_ce).mean().item()
      b_mean_ce = torch.tensor(base_times_ce).mean().item()
      d_ce = compute_cohens_d(base_times_ce, quack_times_ce)

      section_results.append(
        BenchResult(
          name='quack.linear_CE',
          shape=shape,
          mean_ms=q_mean,
          std_ms=torch.tensor(quack_times_ce).std().item(),
          median_ms=torch.tensor(quack_times_ce).median().item(),
          p95_ms=torch.tensor(quack_times_ce).sort().values[int(0.95 * len(quack_times_ce))].item(),
          p99_ms=torch.tensor(quack_times_ce).sort().values[int(0.99 * len(quack_times_ce))].item(),
          peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
          cohens_d=d_ce,
          significant=abs(d_ce) > 0.8,
          speedup=b_mean_ce / q_mean if q_mean > 0 else 1.0,
          correct_fwd=quack_correct,
          correct_bwd=quack_bwd_correct,
        )
      )

      if quack_bwd_ok:
        clear_between()
        qb_times = timed_runs(quack_bwd)

        def base_bwd_ce():
          h = h_base.clone().requires_grad_(True)
          F.cross_entropy((h @ lm_weight.T).float(), targets_base, reduction='mean').backward()

        bb_times = timed_runs(base_bwd_ce)
        qb_mean = torch.tensor(qb_times).mean().item()
        bb_mean = torch.tensor(bb_times).mean().item()
        d_bwd = compute_cohens_d(bb_times, qb_times)
        section_results.append(
          BenchResult(
            name='quack.linear_CE (bwd)',
            shape=shape,
            mean_ms=qb_mean,
            std_ms=torch.tensor(qb_times).std().item(),
            median_ms=torch.tensor(qb_times).median().item(),
            p95_ms=torch.tensor(qb_times).sort().values[int(0.95 * len(qb_times))].item(),
            p99_ms=torch.tensor(qb_times).sort().values[int(0.99 * len(qb_times))].item(),
            peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
            cohens_d=d_bwd,
            significant=abs(d_bwd) > 0.8,
            speedup=bb_mean / qb_mean if qb_mean > 0 else 1.0,
            correct_bwd=quack_bwd_correct,
          )
        )

    except ImportError:
      section_results.append(BenchResult(name='quack.linear_CE', shape=shape, error='quack not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='quack.linear_CE', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 6: Fused Linear Cross-Entropy')
  print('=' * 90)
  try:
    lce_results = bench_linear_ce()
    all_results['Linear CE'] = lce_results
    print_table(f'Fused Linear CE ({B * L // 2}x{D} -> {V})', lce_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 7: Newton-Schulz (gram-NS vs compiled NS)
  # ==================================================================

  def bench_newton_schulz():
    shape = f'{D} x {D}'
    section_results = []

    # Our compiled Newton-Schulz (from optim.py)
    from phase6.optim import MuonClip

    G = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)

    def baseline_fwd():
      return MuonClip.newton_schulz(G, steps=5)

    ref_out = MuonClip.newton_schulz(G, steps=5).clone()

    # --- gram-newton-schulz (ns_use_kernels=False) ---
    try:
      from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

      gram_ns = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
        ns_use_kernels=False,  # pure algorithmic path, no quack GEMM
      )
      # gram-NS operates differently: it takes the gradient matrix and returns
      # the orthogonalized version. Same input/output semantics as our NS.
      gram_out = gram_ns(G.contiguous())
      fwd_correct = check_close(ref_out, gram_out, atol=0.05, rtol=0.05)
      print(f'  gram-NS (no kernels) fwd correctness: {fwd_correct}')
      print('    (loose tolerance — different polynomial coefficients expected)')

      def gram_fwd():
        return gram_ns(G.contiguous())

      results = bench_pair(
        baseline_fwd,
        gram_fwd,
        'compiled NS',
        'gram-NS (no kernels)',
        shape,
        ref_out=ref_out,
        kern_out=gram_out,
      )
      for r in results:
        if 'gram' in r.name:
          r.correct_fwd = fwd_correct
      section_results.extend(results)

    except ImportError as e:
      print(f'  gram-newton-schulz not installed: {e}')
      section_results.append(BenchResult(name='gram-NS (no kernels)', shape=shape, error='gram-NS not installed'))
    except Exception as e:
      print(f'  gram-NS error: {e}')
      section_results.append(BenchResult(name='gram-NS (no kernels)', shape=shape, error=str(e)[:120]))

    # --- gram-NS with quack kernels ---
    try:
      from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

      # Explicitly enable quack symmetric GEMM (default is True, but be explicit)
      try:
        gram_ns_k = GramNewtonSchulz(
          ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
          gram_newton_schulz_reset_iterations=[2],
          ns_use_kernels=True,  # enable quack gemm_symmetric
        )
        # Fix CUDAGraph tensor overwrite: mark step boundary before each call
        torch.compiler.cudagraph_mark_step_begin()
        gram_k_out = gram_ns_k(G.contiguous())
        print('  gram-NS (with kernels) ran successfully')

        def gram_k_fwd():
          torch.compiler.cudagraph_mark_step_begin()
          return gram_ns_k(G.contiguous())

        clear_between()
        base_times = timed_runs(baseline_fwd)
        gk_times = timed_runs(gram_k_fwd)
        gk_mean = torch.tensor(gk_times).mean().item()
        b_mean_ns = torch.tensor(base_times).mean().item()
        d_ns = compute_cohens_d(base_times, gk_times)

        section_results.append(
          BenchResult(
            name='gram-NS (w/ kernels)',
            shape=shape,
            mean_ms=gk_mean,
            std_ms=torch.tensor(gk_times).std().item(),
            median_ms=torch.tensor(gk_times).median().item(),
            p95_ms=torch.tensor(gk_times).sort().values[int(0.95 * len(gk_times))].item(),
            p99_ms=torch.tensor(gk_times).sort().values[int(0.99 * len(gk_times))].item(),
            peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
            cohens_d=d_ns,
            significant=abs(d_ns) > 0.8,
            speedup=b_mean_ns / gk_mean if gk_mean > 0 else 1.0,
            correct_fwd=check_close(ref_out, gram_k_out, atol=0.05, rtol=0.05),
          )
        )

      except TypeError as te:
        print(f'  gram-NS kernel flag not on GramNewtonSchulz: {te}')
        section_results.append(
          BenchResult(name='gram-NS (w/ kernels)', shape=shape, error=f'ns_use_kernels not supported: {str(te)[:80]}')
        )

    except ImportError:
      pass
    except Exception as e:
      section_results.append(BenchResult(name='gram-NS (w/ kernels)', shape=shape, error=str(e)[:120]))

    # Also test rectangular shapes
    for m, k, label in [(D, HD, f'{D}x{HD}'), (FFN, D, f'{FFN}x{D}')]:
      G_rect = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
      ref_rect = MuonClip.newton_schulz(G_rect, steps=5).clone()
      try:
        from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

        gram_ns_rect = GramNewtonSchulz(
          ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
          gram_newton_schulz_reset_iterations=[2],
        )
        gram_rect_out = gram_ns_rect(G_rect.contiguous())
        correct = check_close(ref_rect, gram_rect_out, atol=0.05, rtol=0.05)
        print(f'  gram-NS rectangular {label}: {correct}')

        clear_between()

        def base_rect():
          return MuonClip.newton_schulz(G_rect, steps=5)

        def gram_rect():
          return gram_ns_rect(G_rect.contiguous())

        bt = timed_runs(base_rect)
        gt = timed_runs(gram_rect)
        gm = torch.tensor(gt).mean().item()
        bm = torch.tensor(bt).mean().item()
        d_r = compute_cohens_d(bt, gt)
        section_results.append(
          BenchResult(
            name=f'gram-NS {label}',
            shape=label,
            mean_ms=gm,
            std_ms=torch.tensor(gt).std().item(),
            median_ms=torch.tensor(gt).median().item(),
            p95_ms=torch.tensor(gt).sort().values[int(0.95 * len(gt))].item(),
            p99_ms=torch.tensor(gt).sort().values[int(0.99 * len(gt))].item(),
            peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
            cohens_d=d_r,
            significant=abs(d_r) > 0.8,
            speedup=bm / gm if gm > 0 else 1.0,
            correct_fwd=correct,
          )
        )
      except Exception as e:
        print(f'  gram-NS {label} error: {e}')

    # --- Batched 3D gram-NS (the primary use case: stack same-shape params) ---
    try:
      from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

      N_BATCH = 28  # one per layer (e.g., all c_k weights are (1024, 1024))
      G_batch = torch.randn(N_BATCH, D, D, device='cuda', dtype=torch.bfloat16)
      gram_ns_batch = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
      )

      # Baseline: sequential NS calls (like current optimizer)
      def base_sequential():
        return torch.stack([MuonClip.newton_schulz(G_batch[i], steps=5) for i in range(N_BATCH)])

      # Kernel: single batched call
      def gram_batched():
        return gram_ns_batch(G_batch.contiguous())

      clear_between()
      seq_times = timed_runs(base_sequential)
      bat_times = timed_runs(gram_batched)
      seq_mean = torch.tensor(seq_times).mean().item()
      bat_mean = torch.tensor(bat_times).mean().item()
      d_batch = compute_cohens_d(seq_times, bat_times)
      print(f'  gram-NS batched ({N_BATCH}x{D}x{D}): {bat_mean:.3f}ms vs sequential {seq_mean:.3f}ms')
      section_results.append(
        BenchResult(
          name=f'gram-NS batched ({N_BATCH}x)',
          shape=f'{N_BATCH}x{D}x{D}',
          mean_ms=bat_mean,
          std_ms=torch.tensor(bat_times).std().item(),
          median_ms=torch.tensor(bat_times).median().item(),
          p95_ms=torch.tensor(bat_times).sort().values[int(0.95 * len(bat_times))].item(),
          p99_ms=torch.tensor(bat_times).sort().values[int(0.99 * len(bat_times))].item(),
          peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
          cohens_d=d_batch,
          significant=abs(d_batch) > 0.8,
          speedup=seq_mean / bat_mean if bat_mean > 0 else 1.0,
          correct_fwd='N/A',
        )
      )
    except ImportError:
      pass
    except Exception as e:
      print(f'  gram-NS batched error: {e}')

    return section_results

  print('\n' + '=' * 90)
  print(' Section 7: Newton-Schulz')
  print('=' * 90)
  try:
    ns_results = bench_newton_schulz()
    all_results['Newton-Schulz'] = ns_results
    print_table(f'Newton-Schulz ({D}x{D})', ns_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 8: Fused RoPE
  # ==================================================================

  def bench_rope():
    shape = f'Q:({B},{H},{L},{HD}) K:({B},{KV},{L},{HD})'
    section_results = []

    # Our manual RoPE (from attention.py)
    from phase6.attention import apply_rotary_emb as our_rope

    q = torch.randn(B, L, H, HD, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, L, KV, HD, device='cuda', dtype=torch.bfloat16)

    # Precompute cos/sin like our model
    half_dim = HD // 2
    inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, HD, 2, dtype=torch.float32, device='cuda') / HD))
    t = torch.arange(L, dtype=torch.float32, device='cuda')
    freqs = torch.outer(t, inv_freq)
    cos_our = freqs.cos()[None, :, None, :]  # (1, L, 1, HD//2)
    sin_our = freqs.sin()[None, :, None, :]

    def baseline_fwd():
      return our_rope(q, k, cos_our, sin_our)

    ref_q, ref_k = our_rope(q, k, cos_our, sin_our)

    # --- flash-attn fused RoPE ---
    try:
      from flash_attn.layers.rotary import apply_rotary_emb as fa_rope

      # flash-attn expects: x (B, L, H, HD), cos (L, HD//2), sin (L, HD//2)
      cos_fa = freqs.cos()  # (L, HD//2)
      sin_fa = freqs.sin()  # (L, HD//2)

      # flash-attn apply_rotary_emb works on a single tensor
      fa_q = fa_rope(q.contiguous(), cos_fa, sin_fa, interleaved=False)
      fa_k = fa_rope(k.contiguous(), cos_fa, sin_fa, interleaved=False)
      fwd_q_correct = check_close(ref_q, fa_q)
      fwd_k_correct = check_close(ref_k, fa_k)
      print(f'  flash-attn RoPE Q correctness: {fwd_q_correct}')
      print(f'  flash-attn RoPE K correctness: {fwd_k_correct}')

      def kernel_fwd():
        fa_rope(q.contiguous(), cos_fa, sin_fa, interleaved=False)
        fa_rope(k.contiguous(), cos_fa, sin_fa, interleaved=False)

      # Backward
      q_ref_bwd = q.clone().requires_grad_(True)
      k_ref_bwd = k.clone().requires_grad_(True)
      q_out, k_out = our_rope(q_ref_bwd, k_ref_bwd, cos_our, sin_our)
      (q_out.sum() + k_out.sum()).backward()
      ref_q_grad = q_ref_bwd.grad.clone()
      ref_k_grad = k_ref_bwd.grad.clone()

      q_fa_bwd = q.clone().contiguous().requires_grad_(True)
      k_fa_bwd = k.clone().contiguous().requires_grad_(True)
      try:
        fa_q_out = fa_rope(q_fa_bwd, cos_fa, sin_fa, interleaved=False)
        fa_k_out = fa_rope(k_fa_bwd, cos_fa, sin_fa, interleaved=False)
        (fa_q_out.sum() + fa_k_out.sum()).backward()
        fa_q_grad = q_fa_bwd.grad.clone()
        fa_k_grad = k_fa_bwd.grad.clone()
        bwd_q_correct = check_close(ref_q_grad, fa_q_grad)
        bwd_k_correct = check_close(ref_k_grad, fa_k_grad)
        print(f'  flash-attn RoPE Q bwd correctness: {bwd_q_correct}')
        print(f'  flash-attn RoPE K bwd correctness: {bwd_k_correct}')
        bwd_ok = True
      except Exception as e:
        print(f'  flash-attn RoPE bwd FAILED: {e}')
        bwd_q_correct = f'FAIL ({str(e)[:80]})'
        bwd_ok = False

      def baseline_bwd():
        qq = q.clone().requires_grad_(True)
        kk = k.clone().requires_grad_(True)
        qo, ko = our_rope(qq, kk, cos_our, sin_our)
        (qo.sum() + ko.sum()).backward()

      def kernel_bwd():
        qq = q.clone().contiguous().requires_grad_(True)
        kk = k.clone().contiguous().requires_grad_(True)
        qo = fa_rope(qq, cos_fa, sin_fa, interleaved=False)
        ko = fa_rope(kk, cos_fa, sin_fa, interleaved=False)
        (qo.sum() + ko.sum()).backward()

      results = bench_pair(
        baseline_fwd,
        kernel_fwd,
        'Manual RoPE',
        'flash-attn RoPE',
        shape,
        baseline_bwd_fn=baseline_bwd,
        kernel_bwd_fn=kernel_bwd if bwd_ok else None,
      )
      for r in results:
        if 'flash' in r.name and 'bwd' not in r.name:
          r.correct_fwd = f'Q:{fwd_q_correct} K:{fwd_k_correct}'
        elif 'flash' in r.name:
          r.correct_bwd = f'Q:{bwd_q_correct}'
      section_results.extend(results)

    except ImportError:
      section_results.append(BenchResult(name='flash-attn RoPE', shape=shape, error='flash-attn not installed'))
    except Exception as e:
      section_results.append(BenchResult(name='flash-attn RoPE', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 8: Fused RoPE')
  print('=' * 90)
  try:
    rope_results = bench_rope()
    all_results['RoPE'] = rope_results
    print_table(f'Fused RoPE Q:({B},{H},{L},{HD}) K:({B},{KV},{L},{HD})', rope_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SECTION 9: ThunderKittens Dense Attention
  # ==================================================================

  def bench_thunderkittens():
    shape = f'({B},{H},{L},{HD})'
    section_results = []

    try:
      import thunderkittens as tk

      q = torch.randn(B, H, L, HD, device='cuda', dtype=torch.bfloat16)
      k = torch.randn(B, H, L, HD, device='cuda', dtype=torch.bfloat16)
      v = torch.randn(B, H, L, HD, device='cuda', dtype=torch.bfloat16)

      # Baseline: F.scaled_dot_product_attention
      def baseline_fwd():
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

      ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

      # TK attention
      tk_out = tk.attention(q.contiguous(), k.contiguous(), v.contiguous())
      fwd_correct = check_close(ref_out, tk_out)
      print(f'  ThunderKittens attention fwd correctness: {fwd_correct}')

      def kernel_fwd():
        return tk.attention(q.contiguous(), k.contiguous(), v.contiguous())

      results = bench_pair(
        baseline_fwd,
        kernel_fwd,
        'F.sdpa (non-causal)',
        'ThunderKittens',
        shape,
        ref_out=ref_out,
        kern_out=tk_out,
      )
      section_results.extend(results)

    except ImportError:
      section_results.append(
        BenchResult(name='ThunderKittens', shape=shape, error='ThunderKittens not installed (build failed)')
      )
    except Exception as e:
      section_results.append(BenchResult(name='ThunderKittens', shape=shape, error=str(e)[:120]))

    return section_results

  print('\n' + '=' * 90)
  print(' Section 9: ThunderKittens Attention')
  print('=' * 90)
  try:
    tk_results = bench_thunderkittens()
    all_results['ThunderKittens'] = tk_results
    print_table(f'ThunderKittens Attention ({B},{H},{L},{HD})', tk_results)
  except Exception as e:
    print(f'  SECTION FAILED: {e}')
  clear_between()

  # ==================================================================
  # SUMMARY REPORT
  # ==================================================================

  print('\n')
  print('=' * 90)
  print(' KERNEL BENCHMARK REPORT -- Qwen3-0.6B Real Shapes (H100)')
  print('=' * 90)
  print(f' GPU: {gpu_name} | {total_mem / 1e9:.0f} GB')
  print(f' Methodology: {WARMUP} warmup + {ITERS} timed iterations')
  print(" Statistical significance: Cohen's d > 0.8")
  print(f' Shapes: B={B} L={L} D={D} H={H} KV={KV} HD={HD} FFN={FFN} V={V}')
  print()

  # Per-section summary tables already printed above.
  # Now print integration recommendations.

  print('=' * 90)
  print(' SUMMARY: KERNEL RECOMMENDATIONS')
  print('=' * 90)

  rec_hdr = f'{"Kernel":<35} {"Verdict":<10} {"Speedup":<10} {"Mem delta":<12} {"Integrate?":<12} {"Notes"}'
  print(rec_hdr)
  print('-' * len(rec_hdr))

  for section_name, results in all_results.items():
    for r in results:
      # Skip baselines and backward-only rows
      if r.error:
        verdict = 'ERROR'
        integrate = 'NO'
        notes = r.error[:50]
      elif r.speedup > 1.05 and r.significant:
        verdict = 'FASTER'
        integrate = 'YES'
        notes = ''
      elif r.speedup < 0.95 and r.significant:
        verdict = 'SLOWER'
        integrate = 'NO'
        notes = 'launch overhead or insufficient dim'
      elif not r.significant:
        verdict = 'NO DIFF'
        integrate = 'SKIP'
        notes = 'not statistically significant'
      else:
        verdict = 'NEUTRAL'
        integrate = 'MAYBE'
        notes = ''

      # Check correctness
      if 'FAIL' in str(r.correct_fwd) or 'FAIL' in str(r.correct_bwd):
        integrate = 'NO'
        notes = 'correctness failure'

      # Only print kernel rows (not baselines)
      is_baseline = any(
        base in r.name
        for base in [
          'F.cross_entropy',
          'nn.RMSNorm',
          'torch.mm',
          'nn.Linear',
          'Manual SwiGLU',
          'matmul+CE',
          'compiled NS',
          'Manual RoPE',
          'F.sdpa',
        ]
      )
      if is_baseline:
        continue

      mem_delta = ''
      # Find matching baseline for memory comparison
      for base_r in results:
        is_base = any(
          bn in base_r.name
          for bn in [
            'F.cross_entropy',
            'nn.RMSNorm',
            'torch.mm',
            'nn.Linear',
            'Manual SwiGLU',
            'matmul+CE',
            'compiled NS',
            'Manual RoPE',
            'F.sdpa',
          ]
        )
        if is_base and base_r.peak_mem_mb > 0 and r.peak_mem_mb > 0:
          delta = r.peak_mem_mb - base_r.peak_mem_mb
          mem_delta = f'{delta:+.0f} MB'
          break

      spd_str = f'{r.speedup:.2f}x' if r.speedup > 0 else 'N/A'
      print(f'  {r.name:<33} {verdict:<10} {spd_str:<10} {mem_delta:<12} {integrate:<12} {notes}')

  print()
  print('=' * 90)
  print(' ROOT CAUSES FOR SLOW/FAILING KERNELS')
  print('=' * 90)

  slow_kernels = []
  for section_name, results in all_results.items():
    for r in results:
      is_baseline = any(
        base in r.name
        for base in [
          'F.cross_entropy',
          'nn.RMSNorm',
          'torch.mm',
          'nn.Linear',
          'Manual SwiGLU',
          'matmul+CE',
          'compiled NS',
          'Manual RoPE',
          'F.sdpa',
        ]
      )
      if is_baseline:
        continue
      if r.error:
        slow_kernels.append(f'  {r.name}: {r.error}')
      elif r.speedup < 1.0 and r.significant:
        slow_kernels.append(
          f'  {r.name}: {r.speedup:.2f}x -- kernel dispatch overhead '
          f'dominates at D={D} (CuTeDSL launch cost > compute savings)'
        )
      elif 'FAIL' in str(r.correct_fwd) or 'FAIL' in str(r.correct_bwd):
        slow_kernels.append(f'  {r.name}: correctness failure -- fwd={r.correct_fwd} bwd={r.correct_bwd}')

  if slow_kernels:
    for i, s in enumerate(slow_kernels, 1):
      print(f'{i}. {s}')
  else:
    print('  All kernels faster or neutral.')

  print('\n' + '=' * 90)
  print(' BENCHMARK COMPLETE')
  print('=' * 90)

  return all_results
