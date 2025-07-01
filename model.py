import torch
import torch.nn as nn
import torch.nn.functional as F

class MSTRBlock(nn.Module):
    def __init__(self, input_dim, p=3, L=4, num_heads=6, dropout=0.1):
        super(MSTRBlock, self).__init__()
        self.p = p
        self.L = L
        self.norm = nn.LayerNorm(input_dim)
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        B, T, F_dim = x_norm.size()

        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        multiscale_outputs = []
        for k in range(self.L):
            scale = self.p ** k
            if T < scale:
                continue

            pooled_Q = F.avg_pool1d(Q.transpose(1, 2), scale, scale).transpose(1, 2)
            pooled_K = F.avg_pool1d(K.transpose(1, 2), scale, scale).transpose(1, 2)
            pooled_V = F.avg_pool1d(V.transpose(1, 2), scale, scale).transpose(1, 2)

            windows = pooled_Q.size(1) // self.p
            outputs = []
            for i in range(windows):
                start = i * self.p
                end = start + self.p
                if end > pooled_Q.size(1):
                    break
                q = pooled_Q[:, start:end, :]
                k_ = pooled_K[:, start:end, :]
                v = pooled_V[:, start:end, :]

                q2 = q.transpose(0, 1)  # p x B x F
                k2 = k_.transpose(0, 1)
                v2 = v.transpose(0, 1)
                attn_out, _ = self.mha(q2, k2, v2)
                attn_out = attn_out.transpose(0, 1)
                outputs.append(attn_out)

            if not outputs:
                continue

            scale_output = torch.cat(outputs, dim=1)
            up = F.interpolate(scale_output.transpose(1, 2), size=T, mode='nearest').transpose(1, 2)
            multiscale_outputs.append(F.gelu(up))

        if not multiscale_outputs:
            return residual
        fused = sum(multiscale_outputs)
        out = self.output_proj(fused)
        return residual + out

class MSTRClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, p=3, L=4, num_blocks=4,
                 dropout=0.5, hidden_dims=(256, 128), num_heads=6):
        super(MSTRClassifier, self).__init__()
        blocks = [
            MSTRBlock(input_dim, p=p, L=L, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.classifier(x)




