"""
EBM 학습/샘플링 결과를 그림으로 저장하는 부가 코드
(핵심 EBM 로직은 ebm_two_moons.py 를 참고)

그래프는 이 파일 기준 plots/ 폴더에 저장됩니다.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def moving_average(values, window=200):
    values = np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


@torch.no_grad()
def _energy_grid(energy_net, device, lim=4.0, n=200):
    """-E_phi(x) (정규화 상수를 뺀 log-density)를 격자 위에서 계산.

    exp(-E)로 바로 시각화하면 데이터에서 먼 영역의 에너지가 매우 커서(=exp가 거의 0)
    동적 범위가 극단적으로 벌어져 대부분이 배경색으로 뭉개짐.
    1~99 백분위수로 색 범위를 clip한 log-density(-E_phi)를 그대로 쓰는 편이
    두 초승달 모양의 저에너지 골짜기를 훨씬 선명하게 보여줌.
    """
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device)
    energy = energy_net(grid).cpu().numpy().reshape(n, n)
    log_density = -energy
    return xx, yy, log_density


def plot_generation_result(x0_data, x0_generated, trajectory, energy_net, device,
                            filename="ebm_two_moons_result.png", n_traj=8):
    """원본 데이터 vs 생성된 샘플, 에너지 지형(energy landscape) + Langevin 궤적 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(x0_data[:, 0], x0_data[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Real data $x_0 \\sim p_{data}$ (two moons)")

    axes[1].scatter(x0_generated[:, 0], x0_generated[:, 1], s=5, alpha=0.5, color="orange")
    axes[1].set_title("EBM generated samples (Langevin dynamics)")

    xx, yy, log_density = _energy_grid(energy_net, device)
    vmin, vmax = np.percentile(log_density, [1, 99])
    axes[2].pcolormesh(xx, yy, log_density, shading="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    traj = np.stack(trajectory)  # (n_langevin_steps+1, n_samples, 2)
    for k in range(n_traj):
        axes[2].plot(traj[:, k, 0], traj[:, k, 1], color="steelblue", alpha=0.6, linewidth=0.8)
    axes[2].scatter(traj[0, :n_traj, 0], traj[0, :n_traj, 1], color="blue", s=15, label="start (init)")
    axes[2].scatter(traj[-1, :n_traj, 0], traj[-1, :n_traj, 1], color="black", s=15, label="end (sample)")
    axes[2].set_title("Energy landscape $-E_\\phi(x)$ (log-density) + Langevin trajectories")
    axes[2].legend()

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"결과 이미지 저장: {save_path}")


def plot_loss_curve(loss_history, filename="ebm_loss_curve.png", window=200):
    """스텝별 loss(Tractable Score Matching) 곡선, 이동평균 함께 표시"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, alpha=0.25, linewidth=0.5, label="raw loss (per step)")
    smoothed = moving_average(loss_history, window=window)
    ax.plot(np.arange(len(smoothed)) + window, smoothed, color="red", linewidth=1.5,
            label=f"moving average (window={window})")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\mathrm{Tr}(\nabla_x s_\phi) + \frac{1}{2}\|s_\phi\|^2$")
    ax.set_title("EBM training loss (Tractable Score Matching, Eq. 3.2.2)")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"결과 이미지 저장: {save_path}")
