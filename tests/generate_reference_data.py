#!/usr/bin/env python3
"""Generate PyTorch reference forward/backward data for all GPT ops.

Outputs NumPy .npz files that C++ tests can load to verify GPU kernel correctness.
Each op gets its own .npz file with inputs, outputs, and gradients.

Ops covered (matching the C++ implementations):
  1. vec_add, vec_mul, vec_scale         — elementwise kernels
  2. row_max, row_sum, row_mean          — reduction kernels
  3. layernorm_forward / backward        — layer normalization
  4. gelu_forward / backward             — GELU activation (tanh approx)
  5. linear_forward / backward           — Y = X @ W^T + b
  6. embedding_forward / backward        — gather / scatter-add
  7. masked_softmax_forward / backward   — row-wise softmax
  8. cross_entropy_forward / backward    — log-sum-exp loss
  9. attention_forward / backward         — multi-head causal self-attention
  10. block_forward / backward            — full transformer block
  11. gpt_forward / backward              — full GPT model
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reference_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)


def save(name, **arrays):
    """Save arrays to a .npz file."""
    path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez(path, **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in arrays.items()})
    print(f"  Saved {path} ({len(arrays)} arrays)")


# ============================================================================
# 1. Elementwise: vec_add, vec_mul, vec_scale
# ============================================================================
def gen_elementwise():
    print("Generating: elementwise ops")
    N = 128
    a = torch.randn(N)
    b = torch.randn(N)
    alpha = 2.5

    save("vec_add", a=a, b=b, out=a + b)
    save("vec_mul", a=a, b=b, out=a * b)
    save("vec_scale", x=a, alpha=np.float32(alpha), out=a * alpha)
    save("vec_exp", x=a, out=torch.exp(a))


# ============================================================================
# 2. Reductions: row_max, row_sum, row_mean
# ============================================================================
def gen_reductions():
    print("Generating: reductions")
    rows, cols = 8, 32
    x = torch.randn(rows, cols)

    save("row_max", input=x, output=x.max(dim=1).values)
    save("row_sum", input=x, output=x.sum(dim=1))
    save("row_mean", input=x, output=x.mean(dim=1))


# ============================================================================
# 3. LayerNorm forward / backward
# ============================================================================
def gen_layernorm():
    print("Generating: layernorm")
    rows, D = 8, 32
    eps = 1e-5

    x = torch.randn(rows, D, requires_grad=True)
    gamma = torch.randn(D, requires_grad=True)
    beta = torch.randn(D, requires_grad=True)

    # Forward: manual implementation matching our C++ code
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    y = gamma * x_hat + beta

    # Backward
    dy = torch.randn_like(y)
    y.backward(dy)

    save("layernorm_forward",
         x=x, gamma=gamma, beta=beta, eps=np.float32(eps),
         y=y, mean=mean.squeeze(-1), inv_std=inv_std.squeeze(-1))
    save("layernorm_backward",
         dy=dy, x=x, gamma=gamma, beta=beta, eps=np.float32(eps),
         mean=mean.squeeze(-1), inv_std=inv_std.squeeze(-1),
         dx=x.grad, dgamma=gamma.grad, dbeta=beta.grad)


# ============================================================================
# 4. GELU forward / backward (tanh approximation)
# ============================================================================
def gen_gelu():
    print("Generating: gelu")
    N = 256
    x = torch.randn(N, requires_grad=True)

    # Tanh approximation matching C++ implementation
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x ** 3)
    y = 0.5 * x * (1.0 + torch.tanh(inner))

    dy = torch.randn_like(y)
    y.backward(dy)

    save("gelu_forward", x=x, y=y)
    save("gelu_backward", dy=dy, x=x, dx=x.grad)


# ============================================================================
# 5. Linear forward / backward: Y = X @ W^T + b
# ============================================================================
def gen_linear():
    print("Generating: linear")
    N, D_in, D_out = 8, 32, 64

    x = torch.randn(N, D_in, requires_grad=True)
    w = torch.randn(D_out, D_in, requires_grad=True)
    b = torch.randn(D_out, requires_grad=True)

    y = x @ w.T + b

    dy = torch.randn_like(y)
    y.backward(dy)

    save("linear_forward",
         X=x, weight=w, bias=b, Y=y)
    save("linear_backward",
         X=x, dY=dy, weight=w, bias=b,
         dX=x.grad, dW=w.grad, db=b.grad)


# ============================================================================
# 6. Embedding forward / backward
# ============================================================================
def gen_embedding():
    print("Generating: embedding")
    vocab_size, d_model, n = 100, 32, 16

    table = torch.randn(vocab_size, d_model, requires_grad=True)
    indices = torch.randint(0, vocab_size, (n,))

    output = table[indices]

    # Backward: scatter-add
    d_output = torch.randn_like(output)
    output.backward(d_output)

    save("embedding_forward",
         table=table, indices=indices.int(), output=output)
    save("embedding_backward",
         dY=d_output, indices=indices.int(),
         d_table=table.grad)


# ============================================================================
# 7. Masked softmax forward / backward
# ============================================================================
def gen_softmax():
    print("Generating: masked_softmax")
    rows, cols = 16, 16

    x = torch.randn(rows, cols, requires_grad=True)

    # No mask case.
    probs = F.softmax(x, dim=-1)
    dP = torch.randn_like(probs)
    probs.backward(dP)

    save("softmax_forward_nomask",
         input=x, output=probs)
    save("softmax_backward_nomask",
         dP=dP, P=probs, dX=x.grad)

    # Causal mask case.
    x2 = torch.randn(rows, cols, requires_grad=True)
    # Create causal mask: mask[i,j] = 1 if j > i (masked positions)
    mask = torch.triu(torch.ones(rows, cols, dtype=torch.int8), diagonal=1)
    x_masked = x2.clone()
    x_masked = x_masked.masked_fill(mask.bool(), float('-inf'))
    probs2 = F.softmax(x_masked, dim=-1)
    # Replace NaN with 0 for fully masked rows (shouldn't happen with causal)
    probs2 = probs2.nan_to_num(0.0)

    dP2 = torch.randn_like(probs2)
    probs2.backward(dP2)

    save("softmax_forward_causal",
         input=x2, mask=mask, output=probs2)
    save("softmax_backward_causal",
         dP=dP2, P=probs2, dX=x2.grad)


# ============================================================================
# 8. Cross-entropy forward / backward
# ============================================================================
def gen_cross_entropy():
    print("Generating: cross_entropy")
    N, V = 16, 100

    logits = torch.randn(N, V, requires_grad=True)
    targets = torch.randint(0, V, (N,))

    # Forward: our C++ uses log-sum-exp formulation (mean reduction)
    loss = F.cross_entropy(logits, targets, reduction='mean')

    loss.backward()

    save("cross_entropy_forward",
         logits=logits, targets=targets.int(),
         loss=np.float32(loss.item()))
    save("cross_entropy_backward",
         logits=logits, targets=targets.int(),
         d_logits=logits.grad)


# ============================================================================
# 9. Attention forward / backward (multi-head causal self-attention)
# ============================================================================
def gen_attention():
    print("Generating: attention")
    B, T, D, H = 2, 8, 32, 4
    Dh = D // H

    X = torch.randn(B, T, D, requires_grad=True)
    W_qkv = torch.randn(3 * D, D, requires_grad=True)
    b_qkv = torch.randn(3 * D, requires_grad=True)
    W_o = torch.randn(D, D, requires_grad=True)
    b_o = torch.randn(D, requires_grad=True)

    # Step 1: QKV projection
    X_2d = X.reshape(B * T, D)
    qkv = X_2d @ W_qkv.T + b_qkv  # [B*T, 3D]
    qkv_3d = qkv.reshape(B, T, 3 * D)

    # Step 2: Split Q, K, V
    Q, K, V = qkv_3d[:, :, :D], qkv_3d[:, :, D:2*D], qkv_3d[:, :, 2*D:]
    Q = Q.reshape(B, T, H, Dh).permute(0, 2, 1, 3)  # [B, H, T, Dh]
    K = K.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
    V = V.reshape(B, T, H, Dh).permute(0, 2, 1, 3)

    # Steps 3-4: Scores = Q @ K^T * scale
    scale = 1.0 / math.sqrt(Dh)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, T, T]

    # Steps 5-6: Causal mask + softmax
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs = F.softmax(scores, dim=-1)
    probs = probs.nan_to_num(0.0)

    # Step 7: Context = P @ V
    context = torch.matmul(probs, V)  # [B, H, T, Dh]

    # Step 8: Merge heads
    context_merged = context.permute(0, 2, 1, 3).reshape(B, T, D)

    # Step 9: Output projection
    output = context_merged.reshape(B * T, D) @ W_o.T + b_o
    output = output.reshape(B, T, D)

    # Backward
    dO = torch.randn_like(output)
    output.backward(dO)

    save("attention_forward",
         X=X, W_qkv=W_qkv, b_qkv=b_qkv, W_o=W_o, b_o=b_o,
         B=np.int64(B), T=np.int64(T), D=np.int64(D),
         H=np.int64(H), Dh=np.int64(Dh),
         qkv=qkv_3d, Q=Q, K=K, V=V,
         scores=scores, probs=probs, context=context,
         context_merged=context_merged, output=output)
    save("attention_backward",
         dO=dO, X=X,
         dX=X.grad, dW_qkv=W_qkv.grad, db_qkv=b_qkv.grad,
         dW_o=W_o.grad, db_o=b_o.grad)


# ============================================================================
# 10. Transformer block forward / backward
# ============================================================================
def gen_block():
    print("Generating: transformer_block")
    B, T, D, H = 2, 8, 32, 4
    mlp_hidden = 128
    eps = 1e-5

    x = torch.randn(B, T, D, requires_grad=True)

    # LN1 params
    ln1_gamma = torch.ones(D, requires_grad=True)
    ln1_beta = torch.zeros(D, requires_grad=True)

    # Attention params (use .mul_ for in-place to keep leaf status)
    W_qkv = torch.randn(3 * D, D).mul_(0.02).requires_grad_(True)
    b_qkv = torch.zeros(3 * D, requires_grad=True)
    W_o = torch.randn(D, D).mul_(0.02).requires_grad_(True)
    b_o = torch.zeros(D, requires_grad=True)

    # LN2 params
    ln2_gamma = torch.ones(D, requires_grad=True)
    ln2_beta = torch.zeros(D, requires_grad=True)

    # MLP params
    fc1_w = torch.randn(mlp_hidden, D).mul_(0.02).requires_grad_(True)
    fc1_b = torch.zeros(mlp_hidden, requires_grad=True)
    fc2_w = torch.randn(D, mlp_hidden).mul_(0.02).requires_grad_(True)
    fc2_b = torch.zeros(D, requires_grad=True)

    Dh = D // H
    scale = 1.0 / math.sqrt(Dh)

    # ---- Sub-block 1: LN1 → Attention → Residual ----
    # LN1
    ln1_out = F.layer_norm(x, (D,), ln1_gamma, ln1_beta, eps)

    # Attention (manual)
    ln1_2d = ln1_out.reshape(B * T, D)
    qkv = ln1_2d @ W_qkv.T + b_qkv
    qkv = qkv.reshape(B, T, 3 * D)
    Q, K, V = qkv[:,:,:D], qkv[:,:,D:2*D], qkv[:,:,2*D:]
    Q = Q.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
    K = K.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
    V = V.reshape(B, T, H, Dh).permute(0, 2, 1, 3)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs = F.softmax(scores, dim=-1).nan_to_num(0.0)
    ctx = torch.matmul(probs, V)
    ctx_merged = ctx.permute(0, 2, 1, 3).reshape(B, T, D)
    attn_out = (ctx_merged.reshape(B * T, D) @ W_o.T + b_o).reshape(B, T, D)

    # Residual 1
    residual1 = x + attn_out

    # ---- Sub-block 2: LN2 → MLP → Residual ----
    ln2_out = F.layer_norm(residual1, (D,), ln2_gamma, ln2_beta, eps)

    # MLP
    fc1_out = ln2_out.reshape(B * T, D) @ fc1_w.T + fc1_b
    # GELU (tanh approx)
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (fc1_out + coeff * fc1_out ** 3)
    gelu_out = 0.5 * fc1_out * (1.0 + torch.tanh(inner))

    fc2_out = (gelu_out @ fc2_w.T + fc2_b).reshape(B, T, D)

    # Residual 2
    output = residual1 + fc2_out

    # Backward
    d_output = torch.randn_like(output)
    output.backward(d_output)

    save("block_forward",
         x=x,
         ln1_gamma=ln1_gamma, ln1_beta=ln1_beta,
         W_qkv=W_qkv, b_qkv=b_qkv, W_o=W_o, b_o=b_o,
         ln2_gamma=ln2_gamma, ln2_beta=ln2_beta,
         fc1_w=fc1_w, fc1_b=fc1_b, fc2_w=fc2_w, fc2_b=fc2_b,
         B=np.int64(B), T=np.int64(T), D=np.int64(D),
         H=np.int64(H), mlp_hidden=np.int64(mlp_hidden),
         eps=np.float32(eps),
         ln1_out=ln1_out, attn_out=attn_out, residual1=residual1,
         ln2_out=ln2_out, fc1_out=fc1_out, gelu_out=gelu_out,
         fc2_out=fc2_out, output=output)
    save("block_backward",
         d_output=d_output,
         dx=x.grad,
         d_ln1_gamma=ln1_gamma.grad, d_ln1_beta=ln1_beta.grad,
         dW_qkv=W_qkv.grad, db_qkv=b_qkv.grad,
         dW_o=W_o.grad, db_o=b_o.grad,
         d_ln2_gamma=ln2_gamma.grad, d_ln2_beta=ln2_beta.grad,
         d_fc1_w=fc1_w.grad, d_fc1_b=fc1_b.grad,
         d_fc2_w=fc2_w.grad, d_fc2_b=fc2_b.grad)


# ============================================================================
# 11. Full GPT forward / backward
# ============================================================================
def gen_gpt():
    print("Generating: gpt_model")
    B, T, D, H = 1, 8, 32, 4
    V = 100
    max_seq_len = 16
    n_layers = 1
    mlp_hidden = 128
    eps = 1e-5
    Dh = D // H
    scale = 1.0 / math.sqrt(Dh)

    # Inputs
    input_ids = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B * T,))

    # Parameters (all require grad, use .mul_ for in-place to keep leaf status)
    token_emb = torch.randn(V, D).mul_(0.02).requires_grad_(True)
    pos_emb = torch.randn(max_seq_len, D).mul_(0.02).requires_grad_(True)

    ln1_gamma = torch.ones(D, requires_grad=True)
    ln1_beta = torch.zeros(D, requires_grad=True)
    W_qkv = torch.randn(3*D, D).mul_(0.02).requires_grad_(True)
    b_qkv = torch.zeros(3*D, requires_grad=True)
    W_o = torch.randn(D, D).mul_(0.02).requires_grad_(True)
    b_o = torch.zeros(D, requires_grad=True)
    ln2_gamma = torch.ones(D, requires_grad=True)
    ln2_beta = torch.zeros(D, requires_grad=True)
    fc1_w = torch.randn(mlp_hidden, D).mul_(0.02).requires_grad_(True)
    fc1_b = torch.zeros(mlp_hidden, requires_grad=True)
    fc2_w = torch.randn(D, mlp_hidden).mul_(0.02).requires_grad_(True)
    fc2_b = torch.zeros(D, requires_grad=True)
    final_ln_gamma = torch.ones(D, requires_grad=True)
    final_ln_beta = torch.zeros(D, requires_grad=True)
    lm_head_w = torch.randn(V, D).mul_(0.02).requires_grad_(True)

    # Step 1: Token embedding
    embed_out = token_emb[input_ids.flatten()].reshape(B, T, D)

    # Step 2: Add positional embedding
    embed_out = embed_out + pos_emb[:T].unsqueeze(0)

    # Step 3: Transformer block
    # LN1
    block_in = embed_out
    ln1_out = F.layer_norm(block_in, (D,), ln1_gamma, ln1_beta, eps)

    # Attention
    ln1_2d = ln1_out.reshape(B*T, D)
    qkv = ln1_2d @ W_qkv.T + b_qkv
    qkv = qkv.reshape(B, T, 3*D)
    Q, K, Vt = qkv[:,:,:D], qkv[:,:,D:2*D], qkv[:,:,2*D:]
    Q = Q.reshape(B,T,H,Dh).permute(0,2,1,3)
    K = K.reshape(B,T,H,Dh).permute(0,2,1,3)
    Vt = Vt.reshape(B,T,H,Dh).permute(0,2,1,3)
    scores = torch.matmul(Q, K.transpose(-2,-1)) * scale
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs = F.softmax(scores, dim=-1).nan_to_num(0.0)
    ctx = torch.matmul(probs, Vt)
    ctx_merged = ctx.permute(0,2,1,3).reshape(B,T,D)
    attn_out = (ctx_merged.reshape(B*T,D) @ W_o.T + b_o).reshape(B,T,D)
    residual1 = block_in + attn_out

    # LN2 + MLP
    ln2_out = F.layer_norm(residual1, (D,), ln2_gamma, ln2_beta, eps)
    fc1_out = ln2_out.reshape(B*T,D) @ fc1_w.T + fc1_b
    sqrt_2_over_pi = math.sqrt(2.0/math.pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (fc1_out + coeff * fc1_out**3)
    gelu_out = 0.5 * fc1_out * (1.0 + torch.tanh(inner))
    fc2_out = (gelu_out @ fc2_w.T + fc2_b).reshape(B,T,D)
    block_out = residual1 + fc2_out

    # Step 4: Final LN
    final_ln_out = F.layer_norm(block_out, (D,), final_ln_gamma, final_ln_beta, eps)

    # Step 5: LM head
    logits = final_ln_out.reshape(B*T, D) @ lm_head_w.T  # [B*T, V], no bias

    # Loss
    loss = F.cross_entropy(logits, targets, reduction='mean')
    loss.backward()

    save("gpt_forward",
         input_ids=input_ids.int(), targets=targets.int(),
         B=np.int64(B), T=np.int64(T), D=np.int64(D),
         H=np.int64(H), V=np.int64(V),
         max_seq_len=np.int64(max_seq_len),
         n_layers=np.int64(n_layers),
         mlp_hidden=np.int64(mlp_hidden),
         eps=np.float32(eps),
         token_emb=token_emb, pos_emb=pos_emb,
         ln1_gamma=ln1_gamma, ln1_beta=ln1_beta,
         W_qkv=W_qkv, b_qkv=b_qkv, W_o=W_o, b_o=b_o,
         ln2_gamma=ln2_gamma, ln2_beta=ln2_beta,
         fc1_w=fc1_w, fc1_b=fc1_b, fc2_w=fc2_w, fc2_b=fc2_b,
         final_ln_gamma=final_ln_gamma, final_ln_beta=final_ln_beta,
         lm_head_w=lm_head_w,
         embed_out=embed_out, block_out=block_out,
         final_ln_out=final_ln_out, logits=logits,
         loss=np.float32(loss.item()))
    save("gpt_backward",
         d_token_emb=token_emb.grad, d_pos_emb=pos_emb.grad,
         d_ln1_gamma=ln1_gamma.grad, d_ln1_beta=ln1_beta.grad,
         dW_qkv=W_qkv.grad, db_qkv=b_qkv.grad,
         dW_o=W_o.grad, db_o=b_o.grad,
         d_ln2_gamma=ln2_gamma.grad, d_ln2_beta=ln2_beta.grad,
         d_fc1_w=fc1_w.grad, d_fc1_b=fc1_b.grad,
         d_fc2_w=fc2_w.grad, d_fc2_b=fc2_b.grad,
         d_final_ln_gamma=final_ln_gamma.grad,
         d_final_ln_beta=final_ln_beta.grad,
         d_lm_head_w=lm_head_w.grad)


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print(f"Generating reference data in {OUTPUT_DIR}/\n")
    gen_elementwise()
    gen_reductions()
    gen_layernorm()
    gen_gelu()
    gen_linear()
    gen_embedding()
    gen_softmax()
    gen_cross_entropy()
    gen_attention()
    gen_block()
    gen_gpt()
    print(f"\nDone! All reference data saved to {OUTPUT_DIR}/")
