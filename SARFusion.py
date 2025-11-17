import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---------- utils: window partition/reverse ----------
def pad_to_window_size(x: torch.Tensor, ws: int) -> Tuple[torch.Tensor, Tuple[int,int]]:
    B, C, H, W = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()   # B,Nh,Nw,ws,ws,C
    x = x.view(-1, ws * ws, C)                     # B*nW, M, C
    return x

def window_reverse(windows: torch.Tensor, ws: int, B: int, C: int, H: int, W: int) -> torch.Tensor:
    Nh, Nw = H // ws, W // ws
    x = windows.view(B, Nh, Nw, ws, ws, C)         # B,Nh,Nw,ws,ws,C
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()   # B,C,Nh,ws,Nw,ws
    x = x.view(B, C, H, W)
    return x

# ---------- core: multi-head cross-attention (Opt <- SAR) ----------
class CrossAttention(nn.Module):
    """tokens: [B*, N, C]"""
    def __init__(self, dim: int, num_heads: int = 6, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, opt_tokens: torch.Tensor, sar_tokens: torch.Tensor) -> torch.Tensor:
        Bn, N, C = opt_tokens.shape
        H = self.num_heads

        def reshape_heads(x):
            x = x.view(Bn, N, H, C // H)
            return x.permute(0, 2, 1, 3)  # [Bn,H,N,C/H]

        Q = reshape_heads(self.q(opt_tokens))
        K = reshape_heads(self.k(sar_tokens))
        V = reshape_heads(self.v(sar_tokens))

        attn = (Q @ K.transpose(-2, -1)) * self.scale   # [Bn,H,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ V                                   # [Bn,H,N,C/H]
        out = out.permute(0, 2, 1, 3).contiguous().view(Bn, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ---------- FFN ----------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# ---------- 2D Cross-Attn block with optional windowing + gating ----------
class CrossAttnBlock2D(nn.Module):
    """
    I/O: [B, C, H, W]  (opt & sar features with SAME C,H,W)
    window_size=None -> global; >0 -> windowed
    """
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0,
                 window_size: Optional[int] = 12, gating: bool = True, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ws = 0 if (window_size is None) else int(window_size)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.gating = gating
        if gating:
            self.gate = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, dim), nn.Sigmoid()
            )
            self.gamma = nn.Parameter(torch.tensor(0.1)) 

    @staticmethod
    def _flatten_2d(x_2d: torch.Tensor) -> torch.Tensor:
        return x_2d.permute(0, 2, 3, 1).contiguous().view(x_2d.size(0), -1, x_2d.size(1))

    @staticmethod
    def _unflatten_2d(tokens: torch.Tensor, B: int, C: int, H: int, W: int) -> torch.Tensor:
        return tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        B, C, H, W = opt.shape
        if sar.shape[-2:] != (H, W):
            sar = F.interpolate(sar, size=(H, W), mode="bilinear", align_corners=False)

        if self.ws and (H % self.ws != 0 or W % self.ws != 0):
            opt, (ph1, pw1) = pad_to_window_size(opt, self.ws)
            sar, (ph2, pw2) = pad_to_window_size(sar, self.ws)
            assert (ph1, pw1) == (ph2, pw2)
            Hpad, Wpad = opt.shape[-2:]
        else:
            Hpad, Wpad = H, W

        if self.ws and self.ws > 0:
            opt_tok = window_partition(opt, self.ws)  # [B*nW,M,C]
            sar_tok = window_partition(sar, self.ws)
            opt_ln = self.norm_q(opt_tok)
            sar_ln = self.norm_kv(sar_tok)
            fused = self.attn(opt_ln, sar_ln)
            if self.gating:
                g = self.gate(sar_ln.mean(dim=1))         # [Bn,C]
                fused = opt_tok + self.gamma * (g.unsqueeze(1) * fused)
            else:
                fused = opt_tok + fused
            fused = fused + self.mlp(self.norm_out(fused))
            out = window_reverse(fused, self.ws, B=B, C=C, H=Hpad, W=Wpad)
        else:
            opt_tok = self._flatten_2d(opt)               # [B,HW,C]
            sar_tok = self._flatten_2d(sar)
            opt_ln = self.norm_q(opt_tok)
            sar_ln = self.norm_kv(sar_tok)
            fused = self.attn(opt_ln, sar_ln)
            if self.gating:
                g = self.gate(sar_ln.mean(dim=1))         # [B,C]
                fused = opt_tok + self.gamma * (g.unsqueeze(1) * fused)
            else:
                fused = opt_tok + fused
            fused = fused + self.mlp(self.norm_out(fused))
            out = self._unflatten_2d(fused, B=B, C=C, H=Hpad, W=Wpad)

        if (Hpad, Wpad) != (H, W):
            out = out[:, :, :H, :W]
        return out

# ---------- stack ----------
class CrossAttnFusion(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 6,
                 mlp_ratio: float = 4.0, window_size: Optional[int] = 12,
                 gating: bool = True, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttnBlock2D(dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             window_size=window_size, gating=gating, drop=drop)
            for _ in range(num_layers)
        ])
    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        x = opt
        for blk in self.blocks:
            x = blk(x, sar)
        return x

# ---------- full head: 3-band optical + 2-band SAR ----------
class SARGuidedOpticalHead(nn.Module):
    """
    Optical(3ch) ← SAR(2ch)：
      - 3x3 Conv 
      - Cross-Attn (Opt<-SAR) 
      - 1x1 out 
    """
    def __init__(self, opt_in_ch=3, sar_in_ch=2, dim=96, heads=6, layers=2,
                 mlp_ratio=4.0, window_size=12, gating=True, out_act=None):
        super().__init__()
        self.opt_in = nn.Sequential(
            nn.Conv2d(opt_in_ch, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.sar_in = nn.Sequential(
            nn.Conv2d(sar_in_ch, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.fusion = CrossAttnFusion(dim=dim, num_layers=layers, num_heads=heads,
                                      mlp_ratio=mlp_ratio, window_size=window_size,
                                      gating=gating, drop=0.0)
        self.out_proj = nn.Conv2d(dim, opt_in_ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.1)) 
        self.out_act = out_act 

    def forward(self, opt_rgb: torch.Tensor, sar_2ch: torch.Tensor) -> torch.Tensor:
        B, _, H, W = opt_rgb.shape
        if sar_2ch.shape[-2:] != (H, W):
            sar_2ch = F.interpolate(sar_2ch, size=(H, W), mode="bilinear", align_corners=False)
        f_opt = self.opt_in(opt_rgb)
        f_sar = self.sar_in(sar_2ch)
        f_fused = self.fusion(f_opt, f_sar)          # [B,dim,H,W]
        delta = self.out_proj(f_fused)               # [B,3,H,W]
        out = opt_rgb + self.gamma * delta           
        if self.out_act is not None:
            out = self.out_act(out)
        return out

# ---------- quick test ----------
if __name__ == "__main__":
    B, H, W = 2, 192, 192
    opt = torch.randn(B, 3, H, W)
    sar = torch.randn(B, 2, H, W)

    # model test
    model = SARGuidedOpticalHead(opt_in_ch=3, sar_in_ch=2, dim=96, heads=6, layers=2,
                                 mlp_ratio=4.0, window_size=12, gating=True, out_act=None)
    y = model(opt, sar)
    print("out:", y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params:", f"{n_params/1e6:.2f}M")
