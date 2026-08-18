"""
EBM (Energy-Based Model) 핵심 코드 — 3.1 Energy-Based Models
데이터: two-moons (2D 초승달 두 개, sklearn 제공) - a_vae, b_ddpm 과 동일한 데이터로 결과를 비교

참고 자료: "03.Score-Based_Perspective_From_EBMs_to_NCSN_I.pdf" 의 3.1절
DDPM(b_ddpm), VAE(a_vae)와 코드 구조를 맞췄습니다. 다른 점은 알고리즘뿐입니다.

    energy E_phi(x)         : 데이터를 에너지 지형으로 표현하는 신경망 (낮을수록 그럴듯한 데이터)
    p_phi(x) := exp(-E_phi(x)) / Z_phi                         (식 3.1.1, Boltzmann 분포)
    score s_phi(x) := grad_x log p_phi(x) = -grad_x E_phi(x)   (Z_phi는 x에 대해 상수라 미분하면 사라짐)
    Tractable Score Matching (Hyvarinen)                        (식 3.2.2)
        L_SM(phi) := E_data[ Tr(grad_x s_phi(x)) + 1/2 ||s_phi(x)||^2 ] + C
        -> 정규화 상수 Z_phi를 몰라도, 오직 데이터 샘플만으로 학습 가능
    Langevin Dynamics 샘플링 (식 3.1.5, discrete-time)
        x_{n+1} = x_n + eta * s_phi(x_n) + sqrt(2*eta) * eps_n,  eps_n ~ N(0,I)
        -> Langevin SDE를 오래 반복하면 x_n의 분포가 p_phi(x)로 수렴 (Boltzmann 분포로의 수렴)

가상의 실제 데이터 생성 코드는 data/data.py 로 공유하고(a_vae, DDPM과 동일 데이터 사용),
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

N_STEPS = 30000  # 학습 스텝 수 (VAE, DDPM과 동일하게 맞춤)
BATCH = 256

N_LANGEVIN_STEPS = 1000  # 샘플링 시 Langevin dynamics 반복 횟수
LANGEVIN_STEP_SIZE = 0.01  # 식 3.1.5의 eta


class EnergyNet(nn.Module):
    """에너지 함수 E_phi(x): R^2 -> R

    - 아주 단순한 MLP 사용 (DDPM의 EpsPhi, VAE와 동일한 크기)
    - 슬라이드 17페이지의 "조건 1) Confining Energy" (||x||->inf 일 때 E->inf) 를 만족시키기 위해
      MLP 출력에 작은 이차항 lambda_conf * ||x||^2 을 더해줌.
      이게 없으면 Langevin dynamics가 에너지 지형 바깥으로 발산할 수 있음 (실제 발산 관찰됨).
    """

    def __init__(self, input_dim=2, hidden=128, lambda_conf=0.01):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        e_raw = self.net(x).squeeze(-1)
        e_confine = self.lambda_conf * (x ** 2).sum(dim=1)
        return e_raw + e_confine


def score_fn(energy_net, x):
    """s_phi(x) = -grad_x E_phi(x)  (score-energy 관계식, 식 3.1.4 근거와 동일)"""
    x = x.requires_grad_(True)
    energy = energy_net(x).sum()
    grad_e = torch.autograd.grad(energy, x, create_graph=True)[0]
    return -grad_e, x


def tractable_score_matching_loss(energy_net, x0):
    """L_SM(phi) := E_data[ Tr(grad_x s_phi(x)) + 1/2 ||s_phi(x)||^2 ]   (식 3.2.2, Hyvarinen's tractable SM)

    - Tr(grad_x s_phi(x)) (divergence 항): 데이터가 많은 곳을 sink(끌어당기는 점)로 만듦
    - 1/2 ||s_phi(x)||^2 (norm 항): 데이터가 많은 곳에서 score를 0(정류점)으로 만듦
    - 데이터 차원이 2뿐이라 Hessian의 대각합(trace)을 for문 2번으로 직접 계산 (Hutchinson trick 없이 exact)
    """
    score, x0 = score_fn(energy_net, x0)

    trace = torch.zeros(x0.shape[0], device=x0.device)
    for i in range(x0.shape[1]):
        grad_i = torch.autograd.grad(score[:, i].sum(), x0, create_graph=True)[0]
        trace = trace + grad_i[:, i]

    norm_term = 0.5 * (score ** 2).sum(dim=1)
    return (trace + norm_term).mean()


def train(energy_net, x0_data, n_steps=N_STEPS, batch=BATCH, lr=1e-3, weight_decay=1e-3,
          grad_clip=1.0, device=DEVICE, log_every=1000):
    """Tractable Score Matching 손실을 최소화 (식 3.2.2)

    - weight_decay, grad_clip: 정규화 없이 순수 ISM(Implicit Score Matching)만 최소화하면,
      loss가 계속 더 음수로 발산하면서(-100 -> -1000 이상) 에너지 지형이 데이터 매니폴드 전체가 아니라
      한두 개의 아주 뾰족하고 깊은 우물로 붕괴하는 현상이 실험적으로 관찰됨(생성 샘플이 한 점으로 뭉침).
      weight decay로 에너지 함수가 너무 뾰족해지는 것을 억제하고, gradient clipping으로
      학습 후반부의 불안정한 큰 업데이트를 막으면 두 초승달 모양이 안정적으로 학습됨.
    """
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=lr, weight_decay=weight_decay)
    n_data = x0_data.shape[0]
    loss_history = []

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_data, (batch,), device=device)
        x0 = x0_data[idx]

        loss = tractable_score_matching_loss(energy_net, x0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy_net.parameters(), grad_clip)
        optimizer.step()
        loss_history.append(loss.item())

        if step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    return loss_history


def langevin_sample(energy_net, n_samples=1000, n_langevin_steps=N_LANGEVIN_STEPS,
                     eta=LANGEVIN_STEP_SIZE, device=DEVICE, init_scale=3.0, track_trajectory=False):
    """discrete-time Langevin dynamics 샘플링 (식 3.1.5)

    x_{n+1} = x_n + eta * s_phi(x_n) + sqrt(2*eta) * eps_n
    """
    x = (torch.rand(n_samples, 2, device=device) * 2 - 1) * init_scale  # 넓게 퍼진 초기점에서 시작
    traj = [x.detach().cpu().numpy()] if track_trajectory else None

    for _ in range(n_langevin_steps):
        with torch.no_grad():
            x_in = x.clone().requires_grad_(True)
        score, x_in = score_fn(energy_net, x_in)
        noise = torch.randn_like(x)
        x = (x_in + eta * score + np.sqrt(2 * eta) * noise).detach()

        if track_trajectory:
            traj.append(x.detach().cpu().numpy())

    return x.detach().cpu().numpy(), traj


if __name__ == "__main__":
    from data import load_data  # data/data.py

    x0_data = load_data(device=DEVICE)

    energy_net = EnergyNet().to(DEVICE)
    loss_history = train(energy_net, x0_data)

    x0_generated, trajectory = langevin_sample(energy_net, n_samples=1000, track_trajectory=True)

    from visualize import plot_generation_result, plot_loss_curve

    plot_generation_result(x0_data.cpu().numpy(), x0_generated, trajectory, energy_net, device=DEVICE)
    plot_loss_curve(loss_history)
