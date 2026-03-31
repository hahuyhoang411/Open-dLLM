"""Pre-flight validation for GPU kernel candidates — Qwen3-0.6B block diffusion LLM.

Runs BEFORE the expensive benchmark (modal_kernel_test.py) to catch:
  1. Build failures (flash-attn, gram-NS, ThunderKittens)
  2. API contract bugs (wrong args, wrong shapes, wrong results)
  3. Environment mismatches (torch version, CUDA version, SM arch)

NO timing benchmarks. Correctness and API validation only.

Usage:
    MODAL_PROFILE=meddiesresearch uv run modal run 06_qwen3_dllm/scripts/modal_kernel_validate.py
"""

import modal

app = modal.App('smoldlm-phase6-kernel-validate')

# ---------------------------------------------------------------------------
# Modal image: CUDA devel + all kernel libraries
# ---------------------------------------------------------------------------

image = (
  modal.Image
  .from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
  .apt_install('git', 'build-essential', 'ninja-build')
  .env({
    'MAX_JOBS': '16',
    'TORCH_CUDA_ARCH_LIST': '9.0',
    'CUDA_HOME': '/usr/local/cuda',
    'CXX': 'g++',
    'CC': 'gcc',
  })
  .uv_pip_install(
    'torch',
    'numpy',
    'packaging',
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
    "/.uv/uv pip install --system --no-deps git+https://github.com/Dao-AILab/gram-newton-schulz.git || echo 'GRAM_NS_INSTALL_FAILED'",
  )
  .add_local_dir('06_qwen3_dllm', '/root/06_qwen3_dllm')
)


# ============================================================================
# Shapes — real Qwen3-0.6B
# ============================================================================

B = 4
L = 4096
D = 1024
H = 16
KV = 8
HD = 128
FFN = 3072
V = 151_936


# ============================================================================
# Validation entry point
# ============================================================================


@app.function(image=image, gpu='H100', timeout=3600, startup_timeout=1800)
def validate():
  import importlib
  import sys
  import traceback

  sys.path.insert(0, '/root/06_qwen3_dllm')
  sys.path.insert(0, '/root')

  import torch
  import torch.nn.functional as F
  from torch import nn

  torch.set_float32_matmul_precision('high')
  torch.manual_seed(42)

  # ================================================================
  # Test result tracking
  # ================================================================

  results = []  # list of (section, name, status, detail)

  def record(section, name, *, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    results.append((section, name, status, detail))
    mark = 'PASS' if passed else 'FAIL'
    det = f' -- {detail}' if detail else ''
    print(f'  [{mark}] {name}{det}')

  def record_error(section, name, exc):
    tb = traceback.format_exc()
    results.append((section, name, 'FAIL', str(exc)[:200]))
    print(f'  [FAIL] {name} -- {exc}')
    print(f'    {tb[-500:]}')

  # ================================================================
  # STAGE 0: Environment Validation
  # ================================================================

  print('=' * 80)
  print(' STAGE 0: Environment Validation')
  print('=' * 80)

  # torch + CUDA
  print(f'  torch version:  {torch.__version__}')
  print(f'  CUDA version:   {torch.version.cuda}')
  gpu_name = torch.cuda.get_device_name(0)
  print(f'  GPU:            {gpu_name}')
  props = torch.cuda.get_device_properties(0)
  sm = props.major * 10 + props.minor
  print(f'  SM version:     {sm}')
  total_mem = getattr(props, 'total_mem', None) or getattr(props, 'total_memory', 0)
  print(f'  VRAM:           {total_mem / 1e9:.0f} GB')

  record('env', 'GPU is SM90+ (H100/H200)', passed=sm >= 90, detail=f'SM{sm}')

  # Import checks — use importlib instead of exec
  lib_modules = [
    ('quack', 'quack'),
    ('quack.cross_entropy', 'quack.cross_entropy'),
    ('quack.rmsnorm', 'quack.rmsnorm'),
    ('quack.gemm_interface', 'quack.gemm_interface'),
    ('quack.linear', 'quack.linear'),
    ('quack.linear_cross_entropy', 'quack.linear_cross_entropy'),
    ('liger_kernel', 'liger_kernel.transformers'),
    ('gram_newton_schulz', 'gram_newton_schulz'),
    ('flash_attn.rotary', 'flash_attn.layers.rotary'),
    ('flash_attn.rms_norm', 'flash_attn.ops.rms_norm'),
  ]
  for lib_name, module_path in lib_modules:
    try:
      importlib.import_module(module_path)
      record('env', f'import {lib_name}', passed=True)
    except Exception as e:
      record('env', f'import {lib_name}', passed=False, detail=str(e)[:150])

  print()

  # ================================================================
  # STAGE 1: API Contract Tests
  # ================================================================

  print('=' * 80)
  print(' STAGE 1: API Contract Tests')
  print('=' * 80)

  # Helper: check allclose
  def check_close(ref, test, atol=1e-2, rtol=1e-2):
    ref_f = ref.float()
    test_f = test.float()
    ok = torch.allclose(ref_f, test_f, atol=atol, rtol=rtol)
    if ok:
      return True, ''
    max_diff = (ref_f - test_f).abs().max().item()
    return False, f'max_diff={max_diff:.4e}'

  # ------------------------------------------------------------------
  # 1. quack.cross_entropy
  # ------------------------------------------------------------------

  print('\n--- 1. quack.cross_entropy ---')

  try:
    from quack import cross_entropy as quack_ce

    # Flat list of test cases to avoid deep nesting
    ce_cases = [
      ('bf16', torch.bfloat16, 'V=1024', 1024),
      ('bf16', torch.bfloat16, 'V=151936', V),
      ('fp32', torch.float32, 'V=1024', 1024),
      ('fp32', torch.float32, 'V=151936', V),
    ]
    for dtype_label, dtype, v_label, v_size in ce_cases:
      section = f'quack.CE ({dtype_label}, {v_label})'
      N_CE = 128
      tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

      logits = torch.randn(N_CE, v_size, device='cuda', dtype=dtype)
      targets = torch.randint(0, v_size, (N_CE,), device='cuda')

      # reference
      ref_none = F.cross_entropy(logits.float(), targets, reduction='none')
      ref_mean = F.cross_entropy(logits.float(), targets, reduction='mean')

      # reduction='none' -> (N,) tensor
      try:
        kern_none = quack_ce(logits.contiguous(), targets.contiguous(), reduction='none')
        record(section, 'reduction=none shape', passed=kern_none.shape == (N_CE,), detail=f'{kern_none.shape}')
        record(section, 'reduction=none no NaN', passed=not kern_none.isnan().any().item())
        ok, detail = check_close(ref_none, kern_none, atol=tol)
        record(section, 'reduction=none correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'reduction=none', e)

      # reduction='mean' -> scalar
      try:
        kern_mean = quack_ce(logits.contiguous(), targets.contiguous(), reduction='mean')
        record(section, 'reduction=mean is scalar', passed=kern_mean.dim() == 0, detail=f'dim={kern_mean.dim()}')
        record(section, 'reduction=mean no NaN', passed=not kern_mean.isnan().any().item())
        ok, detail = check_close(ref_mean, kern_mean, atol=tol)
        record(section, 'reduction=mean correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'reduction=mean', e)

      # backward test (only at smaller V to save time)
      if v_size <= 1024:
        try:
          logits_ref = logits.float().clone().requires_grad_(True)
          loss_ref = F.cross_entropy(logits_ref, targets, reduction='none')
          loss_ref.sum().backward()
          ref_grad = logits_ref.grad.clone()

          logits_kern = logits.clone().contiguous().requires_grad_(True)
          loss_kern = quack_ce(logits_kern, targets.contiguous(), reduction='none')
          loss_kern.sum().backward()
          kern_grad = logits_kern.grad

          record(section, 'backward produces grad', passed=kern_grad is not None)
          if kern_grad is not None:
            ok, detail = check_close(ref_grad, kern_grad, atol=tol)
            record(section, 'backward correctness', passed=ok, detail=detail)
        except Exception as e:
          record_error(section, 'backward', e)

  except ImportError as e:
    record('quack.CE', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('quack.CE', 'unexpected', e)

  # ------------------------------------------------------------------
  # 2. quack.rmsnorm
  # ------------------------------------------------------------------

  print('\n--- 2. quack.rmsnorm ---')

  try:
    from quack import rmsnorm as quack_rms

    for d_label, d_size in [('D=64', 64), ('D=1024', D)]:
      section = f'quack.rmsnorm ({d_label})'
      N_RMS = 256

      x = torch.randn(N_RMS, d_size, device='cuda', dtype=torch.bfloat16)
      w = torch.ones(d_size, device='cuda', dtype=torch.bfloat16)
      eps = 1e-6

      # reference
      rms_nn = nn.RMSNorm(d_size, eps=eps).cuda().bfloat16()
      ref_out = rms_nn(x).clone()

      # forward
      try:
        out = quack_rms(x.contiguous(), w.contiguous(), eps=eps)
        record(section, 'forward shape', passed=out.shape == x.shape, detail=f'{out.shape}')
        record(section, 'forward no NaN', passed=not out.isnan().any().item())
        ok, detail = check_close(ref_out, out, atol=1e-2)
        record(section, 'forward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'forward', e)

      # backward with .contiguous() on grad_output (strides issue #88)
      try:
        x_ref = x.clone().requires_grad_(True)
        rms_nn(x_ref).sum().backward()
        ref_grad = x_ref.grad.clone()

        x_kern = x.clone().contiguous().requires_grad_(True)
        out_kern = quack_rms(x_kern, w.contiguous(), eps=eps)
        out_kern.sum().backward()
        kern_grad = x_kern.grad

        record(section, 'backward produces grad', passed=kern_grad is not None)
        if kern_grad is not None:
          ok, detail = check_close(ref_grad, kern_grad, atol=1e-2)
          record(section, 'backward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'backward', e)

  except ImportError as e:
    record('quack.rmsnorm', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('quack.rmsnorm', 'unexpected', e)

  # ------------------------------------------------------------------
  # 3. quack.gemm_interface.gemm_symmetric
  # ------------------------------------------------------------------

  print('\n--- 3. quack.gemm_symmetric ---')

  try:
    from quack.gemm_interface import gemm_symmetric

    for d_label, d_size in [('D=64', 64), ('D=256', 256), ('D=1024', D)]:
      section = f'quack.gemm_symmetric ({d_label})'

      X = torch.randn(d_size, d_size, device='cuda', dtype=torch.bfloat16)
      ref_out = torch.mm(X, X.T)

      try:
        # Correct call: gemm_symmetric(A, B) where B = A.mT
        # The kernel internally does B.mT, so pass X.mT so kernel computes X.mT.mT = X
        # Then result = A @ B^T = X @ X^T
        X_c = X.contiguous()
        X_t = X_c.mT.contiguous()
        result = gemm_symmetric(X_c, X_t)

        # Return type should be Tensor (not Tuple)
        if isinstance(result, tuple):
          record(section, 'return type is Tensor', passed=False, detail=f'got tuple of len {len(result)}')
          result = result[1] if len(result) > 1 else result[0]
        else:
          record(section, 'return type is Tensor', passed=True)

        record(
          section, 'output shape', passed=result.shape == ref_out.shape, detail=f'{result.shape} vs {ref_out.shape}'
        )
        record(section, 'no NaN', passed=not result.isnan().any().item())
        ok, detail = check_close(ref_out, result, atol=1e-2)
        record(section, 'correctness vs torch.mm(X, X.T)', passed=ok, detail=detail)

      except Exception as e:
        record_error(section, 'forward', e)

    # Rectangular shapes used in Newton-Schulz
    for m, k, label in [(D, HD, f'{D}x{HD}'), (D, KV * HD, f'{D}x{KV * HD}')]:
      section = f'quack.gemm_symmetric rect ({label})'
      X_rect = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
      ref_rect = torch.mm(X_rect, X_rect.T)  # (m, m)
      try:
        X_rc = X_rect.contiguous()
        X_rt = X_rc.mT.contiguous()
        result = gemm_symmetric(X_rc, X_rt)
        if isinstance(result, tuple):
          result = result[1] if len(result) > 1 else result[0]
        record(section, 'output shape', passed=result.shape == ref_rect.shape, detail=f'{result.shape}')
        ok, detail = check_close(ref_rect, result, atol=1e-2)
        record(section, 'correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'forward', e)

  except ImportError as e:
    record('quack.gemm_symmetric', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('quack.gemm_symmetric', 'unexpected', e)

  # ------------------------------------------------------------------
  # 4. quack.linear.Linear
  # ------------------------------------------------------------------

  print('\n--- 4. quack.Linear ---')

  try:
    from quack.linear import Linear as QuackLinear

    for d_label, d_in, d_out in [('D=64', 64, 64), ('D=1024', D, D)]:
      section = f'quack.Linear ({d_label})'
      N_LIN = 256

      # Constructor
      try:
        lin = QuackLinear(d_in, d_out, bias=False, dtype=torch.bfloat16).cuda()
        record(section, 'constructor (bias=False)', passed=True)
      except Exception as e:
        record_error(section, 'constructor', e)
        continue

      # Reference
      lin_ref = nn.Linear(d_in, d_out, bias=False, device='cuda', dtype=torch.bfloat16)
      with torch.no_grad():
        lin.weight.copy_(lin_ref.weight)

      x = torch.randn(N_LIN, d_in, device='cuda', dtype=torch.bfloat16)

      # Forward
      try:
        ref_out = lin_ref(x)
        kern_out = lin(x.contiguous())
        record(section, 'forward shape', passed=kern_out.shape == ref_out.shape, detail=f'{kern_out.shape}')
        ok, detail = check_close(ref_out, kern_out, atol=1e-2)
        record(section, 'forward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'forward', e)

      # Backward
      try:
        x_ref = x.clone().requires_grad_(True)
        lin_ref(x_ref).sum().backward()
        ref_grad = x_ref.grad.clone()

        x_kern = x.clone().contiguous().requires_grad_(True)
        lin(x_kern).sum().backward()
        kern_grad = x_kern.grad
        record(section, 'backward produces grad', passed=kern_grad is not None)
        if kern_grad is not None:
          ok, detail = check_close(ref_grad, kern_grad, atol=1e-2)
          record(section, 'backward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'backward', e)

  except ImportError as e:
    record('quack.Linear', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('quack.Linear', 'unexpected', e)

  # ------------------------------------------------------------------
  # 5. quack.linear_cross_entropy.chunked_linear_cross_entropy
  # ------------------------------------------------------------------

  print('\n--- 5. quack.chunked_linear_cross_entropy ---')

  try:
    from quack.linear_cross_entropy import chunked_linear_cross_entropy

    for label, n_ce, d_size, v_size in [
      ('small', 128, 64, 1024),
      ('real', 512, D, V),
    ]:
      section = f'quack.linear_CE ({label})'
      # Must be multiple of 8 for TMA alignment
      assert n_ce % 8 == 0

      h = torch.randn(n_ce, d_size, device='cuda', dtype=torch.bfloat16)
      w = torch.randn(v_size, d_size, device='cuda', dtype=torch.bfloat16) * 0.02
      targets = torch.randint(0, v_size, (n_ce,), device='cuda')

      # Reference: manual matmul + F.cross_entropy
      ref_loss = F.cross_entropy((h @ w.T).float(), targets, reduction='mean')

      # Forward
      try:
        kern_loss = chunked_linear_cross_entropy(
          h.contiguous(),
          w.contiguous(),
          targets.contiguous(),
          chunk_size=min(4096, n_ce),
          reduction='mean',
        )
        record(section, 'forward is scalar', passed=kern_loss.dim() == 0, detail=f'dim={kern_loss.dim()}')
        record(section, 'forward no NaN', passed=not kern_loss.isnan().item())
        # Loose tolerance — different CE implementations
        ok = abs(kern_loss.item() - ref_loss.item()) < 0.5
        record(
          section, 'forward correctness', passed=ok, detail=f'kern={kern_loss.item():.4f} ref={ref_loss.item():.4f}'
        )
      except Exception as e:
        record_error(section, 'forward', e)

      # Backward
      try:
        h_ref = h.clone().float().requires_grad_(True)
        F.cross_entropy(h_ref @ w.float().T, targets, reduction='mean').backward()
        ref_grad = h_ref.grad.clone()

        h_kern = h.clone().contiguous().requires_grad_(True)
        loss_k = chunked_linear_cross_entropy(
          h_kern,
          w.contiguous(),
          targets.contiguous(),
          chunk_size=min(4096, n_ce),
          reduction='mean',
        )
        loss_k.backward()
        kern_grad = h_kern.grad
        record(section, 'backward produces grad', passed=kern_grad is not None)
        if kern_grad is not None:
          ok, detail = check_close(ref_grad, kern_grad, atol=0.1, rtol=0.1)
          record(section, 'backward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'backward', e)

  except ImportError as e:
    record('quack.linear_CE', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('quack.linear_CE', 'unexpected', e)

  # ------------------------------------------------------------------
  # 6. gram_newton_schulz.GramNewtonSchulz
  # ------------------------------------------------------------------

  print('\n--- 6. gram-newton-schulz ---')

  try:
    from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

    # 6a. ns_use_kernels=False (pure algorithmic path)
    section = 'gram-NS (no kernels)'
    try:
      gns = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
        ns_use_kernels=False,
      )
      record(section, 'constructor', passed=True)

      G = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)
      out = gns(G.contiguous())
      record(section, f'forward shape ({D}x{D})', passed=out.shape == G.shape, detail=f'{out.shape}')
      record(section, 'forward no NaN', passed=not out.isnan().any().item())

      # Check approximate orthogonality: out @ out.T ~ I
      orth_err = (out.float() @ out.float().T - torch.eye(D, device='cuda')).abs().max().item()
      record(section, 'approximate orthogonality', passed=orth_err < 0.1, detail=f'max_err={orth_err:.4f}')
    except Exception as e:
      record_error(section, 'forward', e)

    # 6b. ns_use_kernels=True + cudagraph_mark_step_begin
    section = 'gram-NS (with kernels)'
    try:
      gns_k = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
        ns_use_kernels=True,
      )
      record(section, 'constructor (ns_use_kernels=True)', passed=True)

      G = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)
      torch.compiler.cudagraph_mark_step_begin()
      out_k = gns_k(G.contiguous())
      record(section, 'forward with cudagraph_mark_step_begin', passed=True)
      record(section, 'forward no NaN', passed=not out_k.isnan().any().item())
    except Exception as e:
      record_error(section, 'forward', e)

    # 6c. Batched 3D input
    section = 'gram-NS (batched 3D)'
    try:
      N_BATCH = 28
      G_batch = torch.randn(N_BATCH, D, D, device='cuda', dtype=torch.bfloat16)
      gns_batch = GramNewtonSchulz(
        ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
        ns_use_kernels=False,
      )
      out_batch = gns_batch(G_batch.contiguous())
      record(
        section, f'shape ({N_BATCH},{D},{D})', passed=out_batch.shape == G_batch.shape, detail=f'{out_batch.shape}'
      )
      record(section, 'no NaN', passed=not out_batch.isnan().any().item())
    except Exception as e:
      record_error(section, 'batched 3D', e)

    # 6d. Rectangular input
    for m, k, label in [(D, HD, f'{D}x{HD}'), (FFN, D, f'{FFN}x{D}')]:
      section = f'gram-NS rect ({label})'
      try:
        G_rect = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        gns_rect = GramNewtonSchulz(
          ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
          gram_newton_schulz_reset_iterations=[2],
          ns_use_kernels=False,
        )
        out_rect = gns_rect(G_rect.contiguous())
        record(section, 'shape', passed=out_rect.shape == G_rect.shape, detail=f'{out_rect.shape}')
        record(section, 'no NaN', passed=not out_rect.isnan().any().item())
      except Exception as e:
        record_error(section, 'forward', e)

  except ImportError as e:
    record('gram-NS', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('gram-NS', 'unexpected', e)

  # ------------------------------------------------------------------
  # 7. flash_attn.layers.rotary.apply_rotary_emb
  # ------------------------------------------------------------------

  print('\n--- 7. flash-attn RoPE ---')

  try:
    from flash_attn.layers.rotary import apply_rotary_emb as fa_rope

    section = 'flash-attn RoPE'

    # Our manual RoPE reference
    def manual_rope(x, cos, sin):
      d = x.shape[-1] // 2
      x1, x2 = x[..., :d], x[..., d:]
      y1 = x1 * cos - x2 * sin
      y2 = x1 * sin + x2 * cos
      return torch.cat([y1, y2], -1).to(x.dtype)

    for label, b_size, seq_len, h_dim, hd in [
      ('small', 2, 64, 4, 64),
      ('real', B, L, H, HD),
    ]:
      sub = f'{section} ({label})'
      q = torch.randn(b_size, seq_len, h_dim, hd, device='cuda', dtype=torch.bfloat16)

      # Precompute cos/sin
      inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, hd, 2, dtype=torch.float32, device='cuda') / hd))
      t = torch.arange(seq_len, dtype=torch.float32, device='cuda')
      freqs = torch.outer(t, inv_freq)
      cos_fa = freqs.cos()  # (seq_len, HD//2)
      sin_fa = freqs.sin()  # (seq_len, HD//2)

      # Our reference uses (1, seq_len, 1, HD//2) broadcasting
      cos_our = cos_fa[None, :, None, :]
      sin_our = sin_fa[None, :, None, :]
      ref_q = manual_rope(q, cos_our, sin_our)

      try:
        # flash-attn: x (B, L, H, HD), cos (L, HD//2), sin (L, HD//2)
        fa_q = fa_rope(q.contiguous(), cos_fa, sin_fa, interleaved=False)
        record(sub, 'forward shape', passed=fa_q.shape == ref_q.shape, detail=f'{fa_q.shape}')
        record(sub, 'forward no NaN', passed=not fa_q.isnan().any().item())
        ok, detail = check_close(ref_q, fa_q, atol=1e-2)
        record(sub, 'forward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(sub, 'forward', e)

      # Backward (small only)
      if label == 'small':
        try:
          q_ref = q.clone().requires_grad_(True)
          manual_rope(q_ref, cos_our, sin_our).sum().backward()
          ref_grad = q_ref.grad.clone()

          q_kern = q.clone().contiguous().requires_grad_(True)
          fa_rope(q_kern, cos_fa, sin_fa, interleaved=False).sum().backward()
          kern_grad = q_kern.grad
          record(sub, 'backward produces grad', passed=kern_grad is not None)
          if kern_grad is not None:
            ok, detail = check_close(ref_grad, kern_grad, atol=1e-2)
            record(sub, 'backward correctness', passed=ok, detail=detail)
        except Exception as e:
          record_error(sub, 'backward', e)

  except ImportError as e:
    record('flash-attn RoPE', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('flash-attn RoPE', 'unexpected', e)

  # ------------------------------------------------------------------
  # 8. flash_attn.ops.rms_norm.rms_norm
  # ------------------------------------------------------------------

  print('\n--- 8. flash-attn rms_norm ---')

  try:
    from flash_attn.ops.rms_norm import rms_norm as fa_rms_norm

    for d_label, d_size in [('D=64', 64), ('D=1024', D)]:
      section = f'flash-attn rms_norm ({d_label})'
      N_RMS = 256

      x = torch.randn(N_RMS, d_size, device='cuda', dtype=torch.bfloat16)
      w = torch.ones(d_size, device='cuda', dtype=torch.bfloat16)
      eps = 1e-6

      ref_nn = nn.RMSNorm(d_size, eps=eps).cuda().bfloat16()
      ref_out = ref_nn(x).clone()

      try:
        fa_out = fa_rms_norm(x.contiguous(), w.contiguous(), eps)
        record(section, 'forward shape', passed=fa_out.shape == x.shape, detail=f'{fa_out.shape}')
        record(section, 'forward no NaN', passed=not fa_out.isnan().any().item())
        ok, detail = check_close(ref_out, fa_out, atol=1e-2)
        record(section, 'forward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'forward', e)

      # Backward
      try:
        x_ref = x.clone().requires_grad_(True)
        ref_nn(x_ref).sum().backward()
        ref_grad = x_ref.grad.clone()

        x_kern = x.clone().contiguous().requires_grad_(True)
        fa_rms_norm(x_kern, w.contiguous(), eps).sum().backward()
        kern_grad = x_kern.grad
        record(section, 'backward produces grad', passed=kern_grad is not None)
        if kern_grad is not None:
          ok, detail = check_close(ref_grad, kern_grad, atol=1e-2)
          record(section, 'backward correctness', passed=ok, detail=detail)
      except Exception as e:
        record_error(section, 'backward', e)

  except ImportError as e:
    record('flash-attn rms_norm', 'import', passed=False, detail=str(e))
  except Exception as e:
    record_error('flash-attn rms_norm', 'unexpected', e)

  # ================================================================
  # STAGE 2: Quick Smoke Test (5 iterations, verify no crashes)
  # ================================================================

  print()
  print('=' * 80)
  print(' STAGE 2: Smoke Test (5 iterations each)')
  print('=' * 80)

  SMOKE_ITERS = 5

  def smoke_test(name, fn):
    section = f'smoke:{name}'
    try:
      for i in range(SMOKE_ITERS):
        out = fn()
        torch.cuda.synchronize()
        if isinstance(out, torch.Tensor) and out.isnan().any().item():
          record(section, f'iter {i} NaN check', passed=False, detail='NaN detected')
          return
      record(section, f'{SMOKE_ITERS} iterations', passed=True)
    except Exception as e:
      record_error(section, 'crashed', e)

  # quack CE
  try:
    from quack import cross_entropy as quack_ce

    logits_s = torch.randn(512, V, device='cuda', dtype=torch.bfloat16)
    targets_s = torch.randint(0, V, (512,), device='cuda')
    smoke_test('quack.CE (fwd)', lambda: quack_ce(logits_s.contiguous(), targets_s.contiguous(), reduction='none'))
  except ImportError:
    record('smoke:quack.CE', 'skipped', passed=False, detail='not installed')

  # quack rmsnorm
  try:
    from quack import rmsnorm as quack_rms

    x_s = torch.randn(512, D, device='cuda', dtype=torch.bfloat16)
    w_s = torch.ones(D, device='cuda', dtype=torch.bfloat16)
    smoke_test('quack.rmsnorm', lambda: quack_rms(x_s.contiguous(), w_s.contiguous(), eps=1e-6))
  except ImportError:
    record('smoke:quack.rmsnorm', 'skipped', passed=False, detail='not installed')

  # quack gemm_symmetric
  try:
    from quack.gemm_interface import gemm_symmetric

    X_s = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)
    X_sc = X_s.contiguous()
    X_st = X_sc.mT.contiguous()
    smoke_test('quack.gemm_symmetric', lambda: gemm_symmetric(X_sc, X_st))
  except ImportError:
    record('smoke:quack.gemm_symmetric', 'skipped', passed=False, detail='not installed')

  # quack Linear
  try:
    from quack.linear import Linear as QuackLinear

    lin_s = QuackLinear(D, D, bias=False, dtype=torch.bfloat16).cuda()
    x_ls = torch.randn(512, D, device='cuda', dtype=torch.bfloat16)
    smoke_test('quack.Linear', lambda: lin_s(x_ls.contiguous()))
  except ImportError:
    record('smoke:quack.Linear', 'skipped', passed=False, detail='not installed')

  # quack linear CE
  try:
    from quack.linear_cross_entropy import chunked_linear_cross_entropy

    h_s = torch.randn(512, D, device='cuda', dtype=torch.bfloat16)
    w_s2 = torch.randn(V, D, device='cuda', dtype=torch.bfloat16) * 0.02
    t_s = torch.randint(0, V, (512,), device='cuda')
    smoke_test(
      'quack.linear_CE',
      lambda: chunked_linear_cross_entropy(
        h_s.contiguous(), w_s2.contiguous(), t_s.contiguous(), chunk_size=512, reduction='mean'
      ),
    )
  except ImportError:
    record('smoke:quack.linear_CE', 'skipped', passed=False, detail='not installed')

  # gram-NS
  try:
    from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

    gns_s = GramNewtonSchulz(
      ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
      gram_newton_schulz_reset_iterations=[2],
      ns_use_kernels=False,
    )
    G_s = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)
    smoke_test('gram-NS (no kernels)', lambda: gns_s(G_s.contiguous()))
  except ImportError:
    record('smoke:gram-NS', 'skipped', passed=False, detail='not installed')

  # gram-NS with kernels
  try:
    from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz

    gns_sk = GramNewtonSchulz(
      ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
      gram_newton_schulz_reset_iterations=[2],
      ns_use_kernels=True,
    )
    G_sk = torch.randn(D, D, device='cuda', dtype=torch.bfloat16)

    def gram_k_smoke():
      torch.compiler.cudagraph_mark_step_begin()
      return gns_sk(G_sk.contiguous())

    smoke_test('gram-NS (with kernels)', gram_k_smoke)
  except ImportError:
    record('smoke:gram-NS (kernels)', 'skipped', passed=False, detail='not installed')

  # flash-attn RoPE
  try:
    from flash_attn.layers.rotary import apply_rotary_emb as fa_rope

    q_s = torch.randn(B, L, H, HD, device='cuda', dtype=torch.bfloat16)
    inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, HD, 2, dtype=torch.float32, device='cuda') / HD))
    freqs = torch.outer(torch.arange(L, dtype=torch.float32, device='cuda'), inv_freq)
    cos_s = freqs.cos()
    sin_s = freqs.sin()
    smoke_test('flash-attn RoPE', lambda: fa_rope(q_s.contiguous(), cos_s, sin_s, interleaved=False))
  except ImportError:
    record('smoke:flash-attn RoPE', 'skipped', passed=False, detail='not installed')

  # flash-attn rms_norm
  try:
    from flash_attn.ops.rms_norm import rms_norm as fa_rms_norm

    x_fa = torch.randn(512, D, device='cuda', dtype=torch.bfloat16)
    w_fa = torch.ones(D, device='cuda', dtype=torch.bfloat16)
    smoke_test('flash-attn rms_norm', lambda: fa_rms_norm(x_fa.contiguous(), w_fa.contiguous(), 1e-6))
  except ImportError:
    record('smoke:flash-attn rms_norm', 'skipped', passed=False, detail='not installed')

  # Liger RMSNorm
  try:
    from liger_kernel.transformers import LigerRMSNorm

    rms_l = LigerRMSNorm(D, eps=1e-6).cuda().bfloat16()
    x_l = torch.randn(512, D, device='cuda', dtype=torch.bfloat16)
    smoke_test('Liger RMSNorm', lambda: rms_l(x_l))
  except ImportError:
    record('smoke:Liger RMSNorm', 'skipped', passed=False, detail='not installed')

  # ================================================================
  # SUMMARY
  # ================================================================

  print()
  print('=' * 80)
  print(' VALIDATION SUMMARY')
  print('=' * 80)

  # Group by section
  from collections import defaultdict

  by_section = defaultdict(list)
  for sec, name, status, detail in results:
    by_section[sec].append((name, status, detail))

  total_pass = sum(1 for _, _, s, _ in results if s == 'PASS')
  total_fail = sum(1 for _, _, s, _ in results if s == 'FAIL')
  total = len(results)

  # Print compact table
  print(f'\n  {"Section":<40} {"Test":<35} {"Status":<6} {"Detail"}')
  print(f'  {"-" * 40} {"-" * 35} {"-" * 6} {"-" * 40}')

  for sec in by_section:
    for name, status, detail in by_section[sec]:
      mark = 'PASS' if status == 'PASS' else 'FAIL'
      det = detail[:50] if detail else ''
      print(f'  {sec:<40} {name:<35} {mark:<6} {det}')

  print(f'\n  TOTAL: {total_pass} passed, {total_fail} failed, {total} total')

  # Print failures prominently
  failures = [(s, n, d) for s, n, st, d in results if st == 'FAIL']
  if failures:
    print(f'\n  FAILURES ({len(failures)}):')
    for sec, name, detail in failures:
      print(f'    [{sec}] {name}: {detail}')
  else:
    print('\n  ALL TESTS PASSED')

  # Overall verdict
  print()
  if total_fail == 0:
    print('  VERDICT: READY for benchmark run (modal_kernel_test.py)')
  elif total_fail <= 3:
    print('  VERDICT: MOSTLY READY -- review failures above before benchmark')
  else:
    print('  VERDICT: NOT READY -- fix failures before burning GPU time')
  print()

  return total_fail == 0


# ============================================================================
# Local entry point
# ============================================================================


@app.local_entrypoint()
def main():
  ok = validate.remote()
  if ok:
    print('\nValidation PASSED -- safe to run benchmark')
  else:
    print('\nValidation had FAILURES -- review output above')
