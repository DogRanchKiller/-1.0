"""复现 Yeung 等 2024 的深度学习模型（需求文档第 5 节“复现 Yeung 等 2024 的深度学习：怎么做”）。

论文：Yeung、Bunker、Umemoto、Fujii，“Evaluating Soccer Match Prediction Models: A Deep Learning Approach and
Feature Optimization for Gradient-Boosted Trees”（arXiv 2309.14807）。作者代码：GitHub calvinyeungck/
Soccer-Prediction-Challenge-2023（Apache-2.0 许可）。这里按他们的代码重写成我们的模型接口（src/models/base.py），
参数全照原代码（settings.yaml 的 repro_yeung），不调。

一、输入：两队各自“最近 5 场”（Berrar 等 2019 的“近期特征”）
  每队每场记 4 个数：进球、失球、对手实力、主客场（主 1、客 −1）。
  对手实力 = 对手在那场比赛之前、它自己最近 5 场的场均净胜球（对手不到 5 场就算不出，是空的）。
  要预测一场比赛时，把主队、客队最近 5 场排成 5 行（第 1 行是最近一场），每行 8 个数：
    [主队进球, 主队失球, 主队那场的对手实力, 主队那场主客, 客队进球, 客队失球, 客队那场的对手实力, 客队那场主客]
  再加两队的编号，编号经过“嵌入层”变成 1 个可学的数（每队一个，训练时学出来），抄到 5 行上 → 5 × 10 的表。
  不缩放（原代码缩放那几行注释掉了）。
  - 训练：表里有空的比赛不要（原代码 dropna）。
  - 出预测：空的填 0（原代码 fillna(0)）；训练时没见过的队用一个共同的“没见过”编号。

二、网络（每一层都照原代码）
  1. Inception 卷积块（TimesNet 那种）：把 5 × 10 的表当成一张 1 通道的小图，用 1×1、3×3 … 11×11 六种卷积核各卷一次、
     取平均，变成 4 个通道；GELU；再用六种卷积核卷回 1 个通道。形状不变，还是 5 × 10。
  2. 加“位置编码”，再进 Transformer 编码层（1 个头、前馈 1 维、不 dropout）。
  3. 只取编码层输出 5 个位置里的最后一个（原代码 [:, -1, :]），过 1 层全连接（tanh、dropout 0.2），
     再变成 3 个数、softmax → 主胜、平、客胜的概率。
  损失函数 RPS（第 5 节的“排序概率分”，先比“主胜”，再比“主胜或平”）。Adam，学习率 0.01、eps 1e−16，
  每批 100 场、打乱、最后不足 100 的一批丢掉，42 轮，用最后一轮。

三、原代码里两个奇怪的地方，照抄（需求文档写明了）
  - 论文表 7 写全连接 10 层，代码循环写死 range(1)，作者公开的权重文件里也只有 1 层 → 1 层。
  - 位置编码：本来该给“第几场”加标记，原代码按“这一批里第几个样本”算（pos 是批里的序号，i 是第几场）。
    训练时每批 100 个，每个样本拿到的标记不一样；出预测时原代码每次只喂 1 场，pos 都是 0。照这样做。

四、什么时候训练
  回测框架每周调用一次 fit；离上次训练满 retrain_days（91）天才重新训练，训练数据是截止时间前 train_years（5）年
  里两队“最近 5 场”都齐的比赛。两次训练之间，每场赛后只更新“最近 5 场”（update / fit 里的 _process）。
  比赛按开球时间一场一场处理：同一时刻开球的几场先把特征都记下，再一起更新（和 gbdt 一样，免得用到同时开球别场的比分）。
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.models.base import Model, check_result_probs, window_start

RESULT_INDEX = {"H": 0, "D": 1, "A": 2}
N_STATS = 4                          # 每队每场 4 个数：进球、失球、对手实力、主客场


# ======================= 网络 =======================
class InceptionBlock(nn.Module):
    """原代码 Conv_Blocks.Inception_Block_V1（它抄自 thuml/Time-Series-Library）：
    num_kernels 种正方形卷积核（边长 1、3、5…），补边让形状不变，几种结果取平均。"""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int):
        super().__init__()
        self.kernels = nn.ModuleList(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
                                     for i in range(num_kernels))
        for m in self.kernels:                     # 原代码的初始化
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


def positional_encoding(batch: int, window: int, pos0_only: bool = False) -> torch.Tensor:
    """原代码 positional_encoding 的向量化写法，形状 (batch, window, 1)，加到每一列上一样。

    原代码：pos 从 0 数到“这一批有几个样本”，i 从 0 数到“几场”（window）；
    第 pos 个样本第 i 场的标记 = i 是双数时 sin(pos / 100^(2i/window))，单数时 cos(同上)。
    pos0_only：出预测时原代码每次只喂 1 场，pos 都是 0（双数场 0、单数场 1）。"""
    pos = torch.zeros(batch, 1) if pos0_only else torch.arange(batch, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(window, dtype=torch.float32).unsqueeze(0)
    angle = pos / torch.pow(100.0, 2 * i / window)
    pe = torch.where(i.long() % 2 == 0, torch.sin(angle), torch.cos(angle))
    return pe.unsqueeze(-1)


class YeungNet(nn.Module):
    """原代码的 timesnet 类（只留 Transformer 编码器那一支）。"""

    def __init__(self, n_teams: int, cfg: dict):
        super().__init__()
        self.window = int(cfg["window"])
        emb = int(cfg["team_embedding_dim"])
        d = 2 * N_STATS + 2 * emb                  # 每一行的宽度：8 个数 + 两队编号各 emb 个
        ch, nk = int(cfg["inception_channels"]), int(cfg["inception_kernels"])
        self.emb = nn.Embedding(n_teams, emb)
        self.conv = nn.Sequential(InceptionBlock(1, ch, nk), nn.GELU(), InceptionBlock(ch, 1, nk))
        self.encoder = nn.TransformerEncoderLayer(d_model=d, nhead=int(cfg["te_heads"]),
                                                  dim_feedforward=int(cfg["te_feedforward"]),
                                                  dropout=float(cfg["te_dropout"]), batch_first=True)
        self.mlp = nn.ModuleList(nn.Linear(d, d) for _ in range(int(cfg["mlp_layers"])))
        self.dropout = nn.Dropout(float(cfg["mlp_dropout"]))
        self.out = nn.Linear(d, 3)

    def forward(self, recent: torch.Tensor, home: torch.Tensor, away: torch.Tensor,
                pos0_only: bool = False) -> torch.Tensor:
        """recent：(批, 5, 8)；home、away：(批,) 球队编号。返回 (批, 3) 的概率。"""
        b, w = recent.shape[0], self.window
        eh = self.emb(home).unsqueeze(1).expand(b, w, -1)       # 编号抄到 5 行上（原代码 repeat）
        ea = self.emb(away).unsqueeze(1).expand(b, w, -1)
        x = torch.cat([recent, eh, ea], dim=2)                  # (批, 5, 10)
        x = self.conv(x.unsqueeze(1)).squeeze(1)                # 当成 1 通道的小图卷积，形状不变
        x = x + positional_encoding(b, w, pos0_only)
        h = self.encoder(x)[:, -1, :]                           # 只取最后一个位置（原代码 [:, -1, :]）
        for layer in self.mlp:
            h = self.dropout(torch.tanh(layer(h)))
        return F.softmax(self.out(h), dim=1)


def rps_loss(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """原代码的 RPS：0.5 ×（(主胜概率 − 是否主胜)² + (主胜或平概率 − 是否主胜或平)²），对一批取平均。"""
    one = F.one_hot(y, 3).to(p.dtype)
    a = p[:, 0] - one[:, 0]
    return (0.5 * (a ** 2 + (a + p[:, 1] - one[:, 1]) ** 2)).mean()


# ======================= 模型（回测接口） =======================
class YeungModel(Model):
    """把上面的网络包成我们的模型：fit / update / predict（src/models/base.py）。"""

    data = ("比分",)
    usage = "A版输入"

    def __init__(self, name: str, cfg: dict, seed: int):
        self.name, self.cfg, self.seed = name, dict(cfg), int(seed)
        self.window = int(cfg["window"])
        self.first_season = int(cfg["first_season"])
        self.recent: dict[str, deque] = {}           # 队 → 最近 5 场，每场 (进球, 失球, 对手实力, 主客场)，最近的在右边
        self.processed: set[int] = set()
        self.rows: list[tuple] = []                  # 训练表：(开球时间, 主队, 客队, 5×8 的表, 结果 0/1/2)
        self.net: YeungNet | None = None
        self.team_index: dict[str, int] = {}
        self.last_train: pd.Timestamp | None = None
        self.train_log: list[dict] = []

    # ---------------- 最近 5 场 ----------------
    def _snapshot(self, team: str) -> np.ndarray:
        """这队最近 5 场，5 行 × 4 列，第 1 行是最近一场；不到 5 场的行是空的（NaN）。"""
        out = np.full((self.window, N_STATS), np.nan)
        q = self.recent.get(team)
        if q:
            for k, rec in enumerate(reversed(q)):
                out[k] = rec
        return out

    def _strength(self, team: str) -> float:
        """这队“现在”的实力 = 最近 5 场的场均净胜球；不到 5 场算不出（NaN）。给它的对手当“对手实力”。"""
        q = self.recent.get(team)
        if not q or len(q) < self.window:
            return float("nan")
        return sum(gf - ga for gf, ga, _, _ in q) / self.window

    def _features(self, home: str, away: str) -> np.ndarray:
        """一场比赛的 5 × 8 表：左 4 列主队、右 4 列客队（第一部分的排法）。"""
        return np.hstack([self._snapshot(home), self._snapshot(away)])

    def _process(self, rows: pd.DataFrame) -> None:
        """按开球顺序：同一时刻开球的几场先把赛前的表记进训练表，再一场一场把比分算进“最近 5 场”。"""
        if rows is None or rows.empty:
            return
        r = rows[rows["result"].notna() & rows["home_goals"].notna() & ~rows["match_id"].isin(self.processed)
                 & (pd.to_numeric(rows["season"]) >= self.first_season)]
        if r.empty:
            return
        r = r.sort_values(["kickoff_earliest_utc", "match_id"])
        items = list(r[["match_id", "home_team", "away_team", "home_goals", "away_goals", "result",
                        "kickoff_earliest_utc"]].itertuples(index=False, name=None))
        start = 0
        while start < len(items):
            end = start
            while end < len(items) and items[end][6] == items[start][6]:   # 同一个开球时间的一组
                end += 1
            feats = [self._features(it[1], it[2]) for it in items[start:end]]
            for it, f in zip(items[start:end], feats):
                mid, home, away, hg, ag, res, ko = it
                self.processed.add(int(mid))
                self.rows.append((ko, home, away, f, RESULT_INDEX[res]))
                s_home, s_away = self._strength(away), self._strength(home)   # 更新前算：对手在这场之前的实力
                self.recent.setdefault(home, deque(maxlen=self.window)).append((int(hg), int(ag), s_home, 1.0))
                self.recent.setdefault(away, deque(maxlen=self.window)).append((int(ag), int(hg), s_away, -1.0))
            start = end

    def update(self, new_rows: pd.DataFrame, cutoff: str) -> None:
        self._process(new_rows)

    # ---------------- 训练 ----------------
    def _train(self, cutoff: str) -> None:
        since = window_start(cutoff, 365.25 * float(self.cfg["train_years"]))
        keep = [x for x in self.rows if x[0] >= since and not np.isnan(x[3]).any()]   # 原代码 dropna
        bs = int(self.cfg["batch_size"])
        if len(keep) < bs:
            return
        teams = sorted({x[1] for x in keep} | {x[2] for x in keep})
        self.team_index = {t: i for i, t in enumerate(teams)}            # 最后一个编号 len(teams) 给“没见过的队”
        X = torch.tensor(np.stack([x[3] for x in keep]), dtype=torch.float32)
        H = torch.tensor([self.team_index[x[1]] for x in keep])
        A = torch.tensor([self.team_index[x[2]] for x in keep])
        Y = torch.tensor([x[4] for x in keep])
        torch.manual_seed(self.seed)                                     # 初始权重、dropout 每次训练都从同一个种子起
        net = YeungNet(len(teams) + 1, self.cfg)
        opt = torch.optim.Adam(net.parameters(), lr=float(self.cfg["learning_rate"]), eps=float(self.cfg["adam_eps"]))
        gen = torch.Generator().manual_seed(self.seed)                   # 打乱顺序用的
        net.train()
        losses = []
        for _ in range(int(self.cfg["epochs"])):
            perm = torch.randperm(len(keep), generator=gen)
            total, nb = 0.0, 0
            for k in range(0, len(keep) - bs + 1, bs):                   # 不足一批的最后几场丢掉（drop_last）
                idx = perm[k:k + bs]
                loss = rps_loss(net(X[idx], H[idx], A[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, nb = total + loss.item(), nb + 1
            losses.append(total / nb)
        net.eval()
        self.net = net
        self.last_train = pd.Timestamp(cutoff)
        self.train_log.append({"cutoff": cutoff, "rows": len(keep), "teams": len(teams),
                               "first_epoch_rps": round(losses[0], 5), "last_epoch_rps": round(losses[-1], 5)})

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """每周一次：补上没处理的比赛；离上次训练满 retrain_days 天就重新训练。"""
        self._process(history)
        cut = pd.Timestamp(cutoff)
        if self.last_train is None or cut - self.last_train >= pd.Timedelta(days=float(self.cfg["retrain_days"])):
            self._train(cutoff)

    # ---------------- 预测 ----------------
    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        if self.net is None or targets.empty:
            return [None] * len(targets)
        unknown = len(self.team_index)
        X = np.stack([np.nan_to_num(self._features(h, a), nan=0.0)          # 原代码 fillna(0)
                      for h, a in targets[["home_team", "away_team"]].itertuples(index=False, name=None)])
        H = torch.tensor([self.team_index.get(t, unknown) for t in targets["home_team"]])
        A = torch.tensor([self.team_index.get(t, unknown) for t in targets["away_team"]])
        with torch.no_grad():
            p = self.net(torch.tensor(X, dtype=torch.float32), H, A, pos0_only=True).double().numpy()
        return [check_result_probs(q / q.sum()) for q in p]

    def params(self) -> dict:
        return {"seed": self.seed, "训练次数": len(self.train_log),
                "最后一次训练": self.train_log[-1] if self.train_log else None, "设置": self.cfg}
