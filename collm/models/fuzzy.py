"""FNN 模糊决策智能体（论文公式 8-11，表 I）。

论文要点:
  - 对 SM 特征 φs(x) ∈ R^{50×32} 的每个特征维度 d 定义高斯隶属函数
        f_d(z_td) = exp(-(z_td - μ_d)^2 / σ_d^2)
    其中模糊均值 μ 与模糊方差 σ² 均为可学习参数（非专家指定）。
  - 隶属度合并为模糊特征矩阵 M ∈ R^{50×64}（每维 2 个隶属函数，表 I：64 个）；
  - 置信度 Q_s = σ(W·φ_s(x) + b) ∈ [0,1]（公式 11，单层线性头），
    以 MSE 逼近置信度标签 Q*_s = 1 - tanh(|y_s - y*| / α)。

设计决策（定稿，REVISIONS #13/#20/#42）:
  - 每特征维 2 个高斯隶属函数 → 32×2 = 64（表 I）；
  - σ 用 log 参数化保证正值；
  - μ/σ 分布感知初始化（按 SM 特征实际分布，防隶属度饱和）；
  - 模糊特征沿时间维聚合（FD001: mean+max+std+last / FD003: mean）后
    进入单层置信度头（公式 11 字面）。
"""
import torch
import torch.nn as nn

from ..config import FuzzyConfig


class GaussianMembership(nn.Module):
    """可学习高斯隶属函数层：f(z) = exp(-(z-μ)²/σ²)。

    每输入维 n_mem 个隶属函数 → 输出 dim = d_in × n_mem。
    """

    def __init__(self, d_in: int, n_mem: int, init_sigma: float = 1.0,
                 init_mu: torch.Tensor | None = None):
        super().__init__()
        self.d_in, self.n_mem = d_in, n_mem
        if init_mu is not None:
            # 分布感知初始化：每维 n_mem 个中心错开分布（REVISIONS #13）
            assert init_mu.shape == (d_in, n_mem), f"init_mu 形状 {init_mu.shape} != {(d_in, n_mem)}"
            self.mu = nn.Parameter(init_mu.clone())
            self.log_sigma = nn.Parameter(
                torch.log(torch.full((d_in, n_mem), float(init_sigma))))
        else:
            self.mu = nn.Parameter(torch.zeros(d_in, n_mem))
            self.log_sigma = nn.Parameter(
                torch.full((d_in, n_mem), float(torch.log(torch.tensor(init_sigma)))))
            with torch.no_grad():
                nn.init.uniform_(self.mu, -2.0, 2.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_in) → (B, T, d_in*n_mem)"""
        sigma = torch.exp(self.log_sigma)          # 恒正
        d = (z.unsqueeze(-1) - self.mu) ** 2 / (sigma ** 2)   # (B, T, d_in, n_mem)
        return torch.exp(-d).reshape(z.shape[0], z.shape[1], -1)


class FuzzyAgent(nn.Module):
    """FNN 决策智能体：φs(x) → 模糊特征 → 置信度 Q_s ∈ [0,1]。"""

    def __init__(self, cfg: FuzzyConfig, init_mu: torch.Tensor | None = None,
                 init_sigma: float | None = None):
        super().__init__()
        self.cfg = cfg
        n_mem_per_dim = cfg.n_membership // cfg.d_input   # 32 维每维 2 个 = 64
        if init_mu is not None:
            self.membership = GaussianMembership(cfg.d_input, n_mem_per_dim,
                                                 init_sigma or 1.0, init_mu)
        else:
            self.membership = GaussianMembership(cfg.d_input, n_mem_per_dim)
        # 时间聚合：'mean'=1 组统计 | 'stats'=mean+max+std+last（FD001）
        self.pool_dim = 4 if cfg.pool_mode == "stats" else 1
        head_in = cfg.n_membership * self.pool_dim
        # 公式 11 字面：σ(W·φs + b)——单层线性头（实测最优，REVISIONS #42）
        self.head = nn.Linear(head_in, 1)

    @staticmethod
    def init_from_features(feat: torch.Tensor, n_mem: int, sigma_scale: float = 1.0):
        """按输入特征分布初始化 μ/σ（设计决策 #20 / REVISIONS #13）。

        feat: (N, T, d) → 时间均值池化 (N, d)。
        每维 n_mem 个隶属函数中心错开：mean + k·std（k 均匀分布在 [-1.5, 1.5]），
        σ 初始 = 各维 std 的均值（核宽匹配分布，不饱和）。
        """
        z = feat.mean(dim=1)                       # (N, d)
        mu_d = z.mean(dim=0)                       # (d,)
        std_d = z.std(dim=0).clamp(min=1e-3)       # (d,)
        ks = torch.linspace(-1.5, 1.5, n_mem)      # 错开中心
        init_mu = mu_d.unsqueeze(1) + std_d.unsqueeze(1) * ks.unsqueeze(0)   # (d, n_mem)
        init_sigma = float((std_d * sigma_scale).mean().item())
        return init_mu, init_sigma

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """时间维聚合：'mean' 或 'stats'（mean+max+std+last）。"""
        if self.cfg.pool_mode == "mean":
            return h.mean(dim=1)
        return torch.cat([h.mean(dim=1), h.amax(dim=1), h.std(dim=1),
                          h[:, -1]], dim=-1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: φs(x) (B, T, d_s) → Q_s: (B,)"""
        m = self.membership(feat)                  # (B, T, 64)
        m = self._pool(m)                          # 时间维聚合
        q = torch.sigmoid(self.head(m))
        return q.squeeze(-1)


def confidence_label(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """论文公式 10/13：Q* = 1 - tanh(|y - y*| / α)。"""
    return 1.0 - torch.tanh((y_pred - y_true).abs() / alpha)
