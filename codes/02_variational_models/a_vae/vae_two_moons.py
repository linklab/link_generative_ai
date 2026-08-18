"""
VAE (Variational Autoencoder) 핵심 코드
데이터: two-moons (2D 초승달 두 개, sklearn 제공) - b_ddpm 과 동일한 데이터로 결과를 비교

DDPM(b_ddpm/ddpm_two_moons.py)과 코드 구조를 맞췄습니다. 다른 점은 알고리즘뿐입니다.

    encoder q_phi(z|x)      : 데이터 x를 잠재변수 z의 분포(평균 mu, 로그분산 logvar)로 매핑
    reparameterize          : z = mu + std * eps, eps ~ N(0,I)  (재매개변수화 트릭)
    decoder p_theta(x|z)    : 잠재변수 z로부터 데이터 x를 복원(재구성)
    ELBO 학습 목적함수       : reconstruction loss + KL(q(z|x) || N(0,I))
    sampling                : z ~ N(0,I) 에서 바로 decoder로 x를 생성 (DDPM과 달리 반복 없이 한 번에 생성)

가상의 실제 데이터 생성 코드는 data/data.py 로 공유하고(b_ddpm, EBM과 동일 데이터 사용),
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

LATENT_DIM = 2   # 잠재변수 z의 차원 (데이터가 2D이므로 동일하게 2D로 설정)
N_STEPS = 30000  # 학습 스텝 수 (DDPM과 동일하게 맞춤)
BATCH = 256


class VAE(nn.Module):
    """인코더 q_phi(z|x) + 디코더 p_theta(x|z)

    - 아주 단순한 MLP 사용 (DDPM의 EpsPhi와 동일한 크기)
    - 인코더는 mu, logvar 두 개의 head를 가짐 (분포의 평균과 로그분산)
    """

    def __init__(self, input_dim=2, latent_dim=LATENT_DIM, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """z = mu + std * eps, eps ~ N(0,I)  (재매개변수화 트릭)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def train(vae, x0_data, n_steps=N_STEPS, batch=BATCH, lr=1e-3, device=DEVICE, log_every=1000,
          kl_warmup_steps=5000, beta_max=0.3):
    """ELBO 최소화(음의 ELBO를 손실로): reconstruction loss + beta * KL(q(z|x) || N(0,I))

    - KL warm-up: 학습 초반에는 KL 가중치를 0에서 beta_max까지 선형으로 서서히 올림.
      처음부터 KL을 완전히 반영하면 인코더가 prior 쪽으로 너무 빨리 눌려버려
      (posterior collapse) 잠재공간 일부만 쓰게 되고, 결과적으로 prior에서 샘플링한
      z 상당수가 디코더가 본 적 없는 영역이 되어 생성 품질이 나빠짐.
    - beta_max<1 (beta-VAE): two-moons처럼 저차원 비선형 매니폴드 데이터는 KL 가중치가
      1.0이면 posterior가 한쪽으로 강하게 뭉개져 초승달 한쪽만 복원되는 현상이 실험적으로
      관찰됨 (beta_max=1.0 vs 0.3 비교 시 0.3에서 두 초승달이 모두 잘 복원됨)
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n_data = x0_data.shape[0]
    loss_history = []

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_data, (batch,), device=device)
        x0 = x0_data[idx]

        x_hat, mu, logvar = vae(x0)

        recon_loss = ((x_hat - x0) ** 2).sum(dim=1).mean()
        kl_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1).mean()
        kl_weight = beta_max * min(1.0, step / kl_warmup_steps)
        loss = recon_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append((recon_loss + kl_loss).item())  # 로깅/비교용은 항상 kl_weight=1 기준

        if step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} (recon {recon_loss.item():.4f}, "
                  f"kl {kl_loss.item():.4f}, kl_weight {kl_weight:.2f})")

    return loss_history


@torch.no_grad()
def sample(vae, n_samples=1000, latent_dim=LATENT_DIM, device=DEVICE):
    """z ~ N(0,I) 에서 바로 decoder로 x를 생성 (DDPM처럼 반복적인 디노이징 스텝이 없음)"""
    z = torch.randn(n_samples, latent_dim, device=device)
    x_gen = vae.decode(z)
    return x_gen.cpu().numpy(), z.cpu().numpy()


if __name__ == "__main__":
    from data import load_data  # data/data.py

    x0_data = load_data(device=DEVICE)

    vae = VAE().to(DEVICE)
    loss_history = train(vae, x0_data)

    x0_generated, z_samples = sample(vae, n_samples=1000)

    with torch.no_grad():
        mu_encoded, _ = vae.encode(x0_data)
        mu_encoded = mu_encoded.cpu().numpy()

    from visualize import plot_generation_result, plot_loss_curve

    plot_generation_result(x0_data.cpu().numpy(), x0_generated, mu_encoded, z_samples)
    plot_loss_curve(loss_history)
