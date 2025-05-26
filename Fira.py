import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

"""Helper: Weight Initialization"""

def init_linear(linear: nn.Linear, activation: str = 'linear'):
    if activation == 'linear':
        nn.init.xavier_uniform_(linear.weight)
    elif activation == 'gelu':
        gain = nn.init.calculate_gain('gelu')
        nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='linear')
        linear.weight.data.mul_(gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


"""Multi‑Head Self‑Attention Module"""

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        for linear in [self.query, self.key, self.value, self.out_proj]:
            init_linear(linear)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
        att = self.dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


"""Mixture‑of‑Experts (MoE) Layer"""

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, dropout_rate=0.1, top_k=0.8, gate_temperature=0.9):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_temperature = gate_temperature
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.load_balancing_loss_coef = 1.0

    def forward(self, x):
        gate_logits = self.gate(x) / self.gate_temperature
        topk_vals, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(gate_logits, float('-inf')).scatter_(-1, topk_idx, topk_vals)
        gate_probs = F.softmax(mask, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=2)
        output = self.dropout(output)
        output = self.post_norm(output)
        avg_gate = gate_probs.mean(dim=(0, 1))
        self.load_balancing_loss = ((avg_gate - 1.0 / self.num_experts) ** 2).sum() * self.load_balancing_loss_coef
        return output


"""Transformer Block with MoE"""

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, num_experts, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, d_ff, num_experts, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.moe(self.ln2(x)))
        return x


"""FIRA Model"""

class FIRA(nn.Module):
    def __init__(self, vocab_size, d_model=1600, n_head=25, num_layers=48, d_ff=3072, num_experts=4, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, num_experts, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        init_linear(self.lm_head)
        self.dropout = nn.Dropout(dropout)
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_head=n_head, num_layers=num_layers, d_ff=d_ff, num_experts=num_experts, max_seq_len=max_seq_len, dropout=dropout)

    def forward(self, input_ids, mask=None):
        B, T = input_ids.size()
        pos_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config["max_seq_len"]:
                input_ids = input_ids[:, -self.config["max_seq_len"]:]
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0).unsqueeze(0)
            logits = self.forward(input_ids, mask)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg["num_layers"], cfg["n_head"], cfg["d_model"] // cfg["n_head"], cfg["max_seq_len"]
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops = flops_per_token * T * fwdbwd_per_iter
        return flops / dt / 312e12

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        extra_args = {'fused': True} if fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @classmethod
    def from_pretrained(cls, model_type, num_experts=4, override_args=None):
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        cfg = hf_model.config
        args = override_args or {}
        model = cls(vocab_size=cfg.vocab_size, d_model=cfg.n_embd, n_head=cfg.n_head,
                    num_layers=cfg.n_layer, d_ff=cfg.n_embd * 4, num_experts=num_experts,
                    max_seq_len=getattr(cfg, 'n_positions', 1024), dropout=args.get('dropout', cfg.resid_pdrop))
        with torch.no_grad():
            model.token_embedding.weight.copy_(hf_model.transformer.wte.weight)
            model.position_embedding.weight.copy_(hf_model.transformer.wpe.weight[:model.position_embedding.weight.shape[0]])
            for i, layer in enumerate(model.layers):
                hf_layer = hf_model.transformer.h[i]
                layer.ln1.weight.copy_(hf_layer.ln_1.weight)
                layer.ln1.bias.copy_(hf_layer.ln_1.bias)
                layer.attn.query.weight.copy_(hf_layer.attn.c_attn.weight[:model.d_model])
                layer.attn.key.weight.copy_(hf_layer.attn.c_attn.weight[model.d_model:2*model.d_model])
                layer.attn.value.weight.copy_(hf_layer.attn.c_attn.weight[2*model.d_model:])
                layer.attn.query.bias.copy_(hf_layer.attn.c_attn.bias[:model.d_model])
                layer.attn.key.bias.copy_(hf_layer.attn.c_attn.bias[model.d_model:2*model.d_model])
                layer.attn.value.bias.copy_(hf_layer.attn.c_attn.bias[2*model.d_model:])
                layer.attn.out_proj.weight.copy_(hf_layer.attn.c_proj.weight)
                layer.attn.out_proj.bias.copy_(hf_layer.attn.c_proj.bias)
                layer.ln2.weight.copy_(hf_layer.ln_2.weight)
                layer.ln2.bias.copy_(hf_layer.ln_2.bias)
                layer.moe.experts[0][0].weight.copy_(hf_layer.mlp.c_fc.weight)
                layer.moe.experts[0][0].bias.copy_(hf_layer.mlp.c_fc.bias)
                layer.moe.experts[0][3].weight.copy_(hf_layer.mlp.c_proj.weight)
                layer.moe.experts[0][3].bias.copy_(hf_layer.mlp.c_proj.bias)
        return model
