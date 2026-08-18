"""
DDPM 학습/샘플링 결과를 그림으로 저장하는 부가 코드
(핵심 DDPM 로직은 ddpm_two_moons.py 를 참고)

그래프는 이 파일 기준 plots/ 폴더에 저장됩니다.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def moving_average(values, window=200):
    values = np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_generation_result(x0_data, x0_generated, trajectory,
                            filename="ddpm_two_moons_result.png", n_traj=8):
    """원본 데이터 vs 생성된 샘플, 디노이징 과정 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(x0_data[:, 0], x0_data[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Real data $x_0 \\sim p_{data}$ (two moons)")

    axes[1].scatter(x0_generated[:, 0], x0_generated[:, 1], s=5, alpha=0.5, color="orange")
    axes[1].set_title("DDPM generated samples $\\hat x_0$")

    # 몇 개 샘플의 디노이징 궤적 (x_L -> x_0)
    traj = np.stack(trajectory)  # (L+1, n_samples, 2)
    for k in range(n_traj):
        axes[2].plot(traj[:, k, 0], traj[:, k, 1], alpha=0.6, linewidth=1)
    axes[2].scatter(traj[0, :n_traj, 0], traj[0, :n_traj, 1], color="red", label="$x_L$ (noise)")
    axes[2].scatter(traj[-1, :n_traj, 0], traj[-1, :n_traj, 1], color="green", label="$x_0$ (denoised)")
    axes[2].set_title("Denoising trajectories $x_L \\to x_0$")
    axes[2].legend()

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"결과 이미지 저장: {save_path}")


def plot_loss_curve(loss_history, filename="ddpm_loss_curve.png", window=200):
    """스텝별 loss 곡선 (원본은 i가 매 스텝 무작위로 바뀌어 진동이 크므로, 이동평균도 함께 표시)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, alpha=0.25, linewidth=0.5, label="raw loss (per step)")
    smoothed = moving_average(loss_history, window=window)
    ax.plot(np.arange(len(smoothed)) + window, smoothed, color="red", linewidth=1.5,
            label=f"moving average (window={window})")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\|\epsilon_\phi(\mathbf{x}_i,i)-\epsilon\|_2^2$")
    ax.set_title("DDPM training loss (Eq. 2.2.10)")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"결과 이미지 저장: {save_path}")
