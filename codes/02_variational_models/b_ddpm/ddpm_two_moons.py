"""
DDPM (Denoising Diffusion Probabilistic Model) 핵심 코드
데이터: two-moons (2D 초승달 두 개, sklearn 제공)

이 코드의 기호는 강의자료
"02.Variational_Perspective_From_VAEs_to_DDPMs_II.pdf" 의 표기와 그대로 맞춰져 있습니다.

    beta_i      : forward 과정의 노이즈 스케줄 beta_i
    alpha_i     : alpha_i = sqrt(1 - beta_i^2)                (식 2.2.9 아래)
    alpha_bar_i : alpha_bar_i = prod_{k=1}^{i} alpha_k         (식 2.2.9)
    x_i         : x_i = alpha_bar_i * x_0 + sqrt(1-alpha_bar_i^2) * eps   (식 2.2.9)
    eps_phi(x_i, i) : 노이즈를 예측하는 신경망 (epsilon-prediction, 식 2.2.10)
    sampling    : x_{i-1} <- (1/alpha_i) * (x_i - (1-alpha_i^2)/sqrt(1-alpha_bar_i^2) * eps_phi(x_i,i))
                            + sigma(i) * eps_i                 (식 2.2.14)

가상의 실제 데이터 생성 코드는 data/data.py 로 공유하고(a_vae, EBM과 동일 데이터 사용),
그래프 저장/시각화 코드는 visualize.py 로 분리했습니다.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

L = 200          # 전체 diffusion 스텝 수 (슬라이드의 L)
N_STEPS = 30000  # 학습 스텝 수
BATCH = 256


def build_schedule(l=L, beta_max=0.5, device=DEVICE):
    """Forward 과정 노이즈 스케줄 beta_i, alpha_i, alpha_bar_i (식 2.2.9)

    beta_max=0.5 정도는 되어야 alpha_bar_L ~ 0 이 되어 x_L이 거의 순수 노이즈(N(0,I))가 됨
    """
    beta = torch.linspace(1e-3, beta_max, l, device=device)  # beta_i, i=1..L
    alpha = torch.sqrt(1.0 - beta ** 2)                      # alpha_i
    # dim=0 필수: cumprod는 dim 기본값이 없고, alpha가 1차원 (l,) 텐서라 dim=0(시퀀스 방향)만 유효
    alpha_bar = torch.cumprod(alpha, dim=0)                  # alpha_bar_i
    return beta, alpha, alpha_bar


def forward_sample(x0, i, alpha_bar):
    """x_i = alpha_bar_i * x0 + sqrt(1-alpha_bar_i^2) * eps  (식 2.2.9)"""
    eps = torch.randn_like(x0)
    ab = alpha_bar[i].unsqueeze(-1)              # (batch,1)
    xi = ab * x0 + torch.sqrt(1 - ab ** 2) * eps
    return xi, eps


class EpsPhi(nn.Module):
    """노이즈 예측 신경망 eps_phi(x_i, i)

    - 아주 단순한 MLP 사용
    - 시간 정보 i는 [0,1] 구간으로 정규화해서 x와 그냥 이어붙임(concat)
    """

    def __init__(self, l=L, hidden=128):
        super().__init__()
        self.l = l
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_i, i):
        # i: (batch,) 정수 텐서 -> [0,1] 스칼라로 정규화
        t = (i.float() / self.l).unsqueeze(-1)
        h = torch.cat([x_i, t], dim=-1)
        return self.net(h)


def train(eps_phi, x0_data, alpha_bar, l=L, n_steps=N_STEPS, batch=BATCH,
          lr=1e-3, device=DEVICE, log_every=1000):
    """학습: L_simple-DDPM(phi) = E_i,x0,eps || eps_phi(x_i,i) - eps ||^2   (식 2.2.10)"""
    optimizer = torch.optim.Adam(eps_phi.parameters(), lr=lr)
    n_data = x0_data.shape[0]
    loss_history = []

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_data, (batch,), device=device)
        x0 = x0_data[idx]

        i = torch.randint(0, l, (batch,), device=device)  # 0-index -> alpha_bar[i]는 alpha_bar_{i+1}에 대응
        x_i, eps = forward_sample(x0, i, alpha_bar)

        eps_pred = eps_phi(x_i, i)
        loss = ((eps_pred - eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    return loss_history


def sigma2(i, alpha_bar, beta, device=DEVICE):
    """sigma^2(i) = (1 - alpha_bar_{i-1}^2) / (1 - alpha_bar_i^2) * beta_i^2   (Lemma 2.2.2)"""
    if i == 0:
        return torch.tensor(0.0, device=device)
    ab_im1 = alpha_bar[i - 1]
    ab_i = alpha_bar[i]
    return (1 - ab_im1 ** 2) / (1 - ab_i ** 2) * beta[i] ** 2


@torch.no_grad()
def sample(eps_phi, alpha, alpha_bar, beta, l=L, n_samples=1000,
           track_trajectory=False, device=DEVICE):
    """x_L ~ N(0,I) 에서 시작해서 x_0 까지 역과정으로 디노이징 (식 2.2.14, Lemma 2.2.2)"""
    x = torch.randn(n_samples, 2, device=device)  # x_L ~ p_prior = N(0,I)
    traj = [x.cpu().numpy()] if track_trajectory else None

    for idx in reversed(range(l)):  # idx: 0..L-1 이 각각 스텝 1..L 에 대응
        i = torch.full((n_samples,), idx, device=device, dtype=torch.long)
        eps_pred = eps_phi(x, i)

        mean = (x - (1 - alpha[idx] ** 2) / torch.sqrt(1 - alpha_bar[idx] ** 2) * eps_pred) / alpha[idx]
        noise = torch.randn_like(x) if idx > 0 else torch.zeros_like(x)
        x = mean + torch.sqrt(sigma2(idx, alpha_bar, beta, device)) * noise

        if track_trajectory:
            traj.append(x.cpu().numpy())

    return x.cpu().numpy(), traj


if __name__ == "__main__":
    from data import load_data  # data/data.py

    x0_data = load_data(device=DEVICE)
    beta, alpha, alpha_bar = build_schedule()
    print(f"beta      | len={len(beta)}, first 5={beta[:5].tolist()}")
    print(f"alpha     | len={len(alpha)}, first 5={alpha[:5].tolist()}")
    print(f"alpha_bar | len={len(alpha_bar)}, first 5={alpha_bar[:5].tolist()}")

    eps_phi = EpsPhi().to(DEVICE)
    loss_history = train(eps_phi, x0_data, alpha_bar)

    x0_generated, trajectory = sample(eps_phi, alpha, alpha_bar, beta,
                                       n_samples=1000, track_trajectory=True)

    from visualize import plot_generation_result, plot_loss_curve

    plot_generation_result(x0_data.cpu().numpy(), x0_generated, trajectory)
    plot_loss_curve(loss_history)
