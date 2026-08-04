"""FNN 模糊决策智能体（论文公式 8-11）。

论文要点:
  - 对 SM 特征 φs(x) ∈ R^{50×32} 的每个特征维度 d 定义高斯隶属函数
        f_d(z_td) = exp(-(z_td - μ_d)^2 / σ_d^2)
    其中模糊均值 μ 与模糊方差 σ² 均为可学习参数（非专家指定）。
  - 隶属度合并为模糊特征矩阵 M ∈ R^{50×32}；
  - 置信度 Q_s = σ(W·φ_s(x) + b) ∈ [0,1]，以 MSE 逼近置信度标签
        Q*_s = 1 - tanh(|y_s - y*| / α)。

设计决策:
  - 每特征维 2 个高斯隶属函数 → 32×2 = 64 个模糊函数（对齐表 I “模糊函数 64 个”）；
  - σ² 用 softplus 参数化保证正值；
  - 模糊特征沿时间维均值池化后进入置信度头（单层线性 + sigmoid）。
"""
import torch
import torch.nn as nn

from ..config import FuzzyConfig


class GaussianMembership(nn.Module):
    """可学习高斯隶属函数层：f(z) = exp(-(z-μ)²/σ²)。

    每输入维 n_mem 个隶属函数 → 输出 dim = d_in × n_mem。
    默认初始化假设输入近标准正态（σ=1 敏感区覆盖 ±2）；
    若输入分布尺度不同（如 SM 特征 std≈2.9），需传 init_mu/init_sigma
    （见 REVISIONS #13：σ=1 与特征尺度不匹配导致隶属度 85.7% 饱和）。
    """

    def __init__(self, d_in: int, n_mem: int, init_sigma: float = 1.0,
                 init_mu: torch.Tensor | None = None):
        super().__init__()
        self.d_in, self.n_mem = d_in, n_mem
        if init_mu is not None:
            # 分布感知初始化：每维 n_mem 个中心错开分布
            assert init_mu.shape == (d_in, n_mem), f"init_mu 形状 {init_mu.shape} != {(d_in, n_mem)}"
            self.mu = nn.Parameter(init_mu.clone())
            self.log_sigma = nn.Parameter(
                torch.log(torch.full((d_in, n_mem), float(init_sigma))))
        else:
            # μ 初始 0，σ 初始 init_sigma（log 空间，exp 保证正）
            self.mu = nn.Parameter(torch.zeros(d_in, n_mem))
            self.log_sigma = nn.Parameter(torch.full((d_in, n_mem), float(torch.log(torch.tensor(init_sigma)))))
            with torch.no_grad():
                nn.init.uniform_(self.mu, -2.0, 2.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_in) → (B, T, d_in*n_mem)"""
        sigma = torch.exp(self.log_sigma)          # 恒正
        # (B, T, d_in, n_mem)
        d = (z.unsqueeze(-1) - self.mu) ** 2 / (sigma ** 2)
        return torch.exp(-d).reshape(z.shape[0], z.shape[1], -1)


class FuzzyAgent(nn.Module):
    """FNN 决策智能体：φs(x) → 模糊特征 → 置信度 Q_s ∈ [0,1]。"""

    def __init__(self, cfg: FuzzyConfig, init_mu: torch.Tensor | None = None,
                 init_sigma: float | None = None):
        super().__init__()
        self.cfg = cfg
        self.pred_dim = 1 if cfg.cat_pred else None   # 拼接预测值 ys（联合分布实验）
        n_mem_per_dim = cfg.n_membership // cfg.d_input
        if init_mu is not None:
            self.membership = GaussianMembership(cfg.d_input, n_mem_per_dim,
                                                 init_sigma or 1.0, init_mu)
        else:
            self.membership = GaussianMembership(cfg.d_input, n_mem_per_dim)
        # 论文公式 11：σ(W·φs + b)；hidden 层增强表达（论文未限层数）
        # 时间聚合：'mean'=1、'stats'=4（mean+max+std+last）、'flatten'=50 步全展平
        if cfg.pool_mode == "flatten":
            self.pool_dim = 50
            # 展平后维度大（50×64=3200）
            head_in = cfg.n_membership if cfg.feat_mode == "fuzzy" else cfg.d_input
            if cfg.hidden:
                # 两层：投影 + MLP（实验）
                self.flatten_proj = nn.Linear(head_in * self.pool_dim, cfg.hidden)
                self.head = nn.Sequential(
                    nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
                    nn.Linear(cfg.hidden, 1))
            else:
                # 论文公式 11 字面：σ(W·flatten(M) + b)——单层
                self.flatten_proj = None
                self.head = nn.Linear(head_in * self.pool_dim + (self.pred_dim or 0), 1)
        else:
            self.pool_dim = 4 if cfg.pool_mode == "stats" else 1   # mean+max+std+last
            head_in = (cfg.n_membership if cfg.feat_mode == "fuzzy" else cfg.d_input) * self.pool_dim
            self.flatten_proj = None
            head_in = head_in + (self.pred_dim or 0)
            if cfg.hidden:
                self.head = nn.Sequential(
                    nn.Linear(head_in, cfg.hidden), nn.GELU(),
                    nn.Linear(cfg.hidden, 1),
                )
            else:
                self.head = nn.Linear(head_in, 1)

    @staticmethod
    def init_from_features(feat: torch.Tensor, n_mem: int, sigma_scale: float = 1.0):
        """按输入特征分布初始化 μ/σ（设计决策 #20）。

        feat: (N, T, d) → 时间均值池化 (N, d)。
        每维 n_mem 个隶属函数中心错开：mean + k·std（k 均匀分布在 [-1.5, 1.5]），
        σ 初始 = 每维 std × sigma_scale（保证核覆盖分布而非饱和）。
        """
        z = feat.mean(dim=1)                       # (N, d)
        mu_d = z.mean(dim=0)                       # (d,)
        std_d = z.std(dim=0).clamp(min=1e-3)       # (d,)
        d = z.shape[1]
        ks = torch.linspace(-1.5, 1.5, n_mem)      # 错开中心
        init_mu = mu_d.unsqueeze(1) + std_d.unsqueeze(1) * ks.unsqueeze(0)   # (d, n_mem)
        init_sigma = float((std_d * sigma_scale).mean().item())
        return init_mu, init_sigma

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """时间维聚合：'mean'/'stats'/'flatten'（算法1 的 F(φs) 输入为完整矩阵）。"""
        if self.cfg.pool_mode == "mean":
            return h.mean(dim=1)
        if self.cfg.pool_mode == "flatten":
            return h.reshape(h.shape[0], -1)
        return torch.cat([h.mean(dim=1), h.amax(dim=1), h.std(dim=1),
                          h[:, -1]], dim=-1)

    def forward(self, feat: torch.Tensor, y_pred: torch.Tensor | None = None) -> torch.Tensor:
        """feat: φs(x) (B, T, d_s) → Q_s: (B,)

        y_pred: SM 预测 ys（论文"R 与 F 学习输入特征和预测结果的联合分布"——
        FNN 输入含预测结果时拼接进置信度头）。
        """
        if self.cfg.feat_mode == "raw":
            h = self._pool(feat)
            if self.cfg.pool_mode == "flatten" and self.flatten_proj is not None:
                h = self.flatten_proj(h)
        else:
            m = self.membership(feat)              # (B, T, n_mem)
            m = self._pool(m)                      # 时间维聚合
            if self.cfg.pool_mode == "flatten" and self.flatten_proj is not None:
                m = self.flatten_proj(m)
            h = m
        if y_pred is not None and self.pred_dim is not None:
            h = torch.cat([h, y_pred.unsqueeze(-1)], dim=-1)
        q = torch.sigmoid(self.head(h))
        return q.squeeze(-1)


def confidence_label(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """论文公式 10：Q* = 1 - tanh(|y - y*| / α)。"""
    return 1.0 - torch.tanh((y_pred - y_true).abs() / alpha)
