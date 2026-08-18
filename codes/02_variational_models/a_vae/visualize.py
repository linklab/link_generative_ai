"""
VAE 학습/샘플링 결과를 그림으로 저장하는 부가 코드
(핵심 VAE 로직은 vae_two_moons.py 를 참고)

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


def plot_generation_result(x0_data, x0_generated, mu_encoded, z_samples,
                            filename="vae_two_moons_result.png"):
    """원본 데이터 vs 생성된 샘플, 잠재공간(latent space) 시각화

    DDPM의 "디노이징 궤적" 패널 대신, VAE는 반복 스텝이 없으므로
    잠재공간에서 실제 데이터가 어떻게 인코딩되는지(q(z|x)의 평균)와
    생성에 쓰인 prior 샘플(z~N(0,I))을 함께 보여줌
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(x0_data[:, 0], x0_data[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Real data $x_0 \\sim p_{data}$ (two moons)")

    axes[1].scatter(x0_generated[:, 0], x0_generated[:, 1], s=5, alpha=0.5, color="orange")
    axes[1].set_title("VAE generated samples $\\hat x_0$")

    axes[2].scatter(mu_encoded[:, 0], mu_encoded[:, 1], s=5, alpha=0.3, color="green",
                     label="encoded $\\mu_\\phi(x)$ (real data)")
    axes[2].scatter(z_samples[:, 0], z_samples[:, 1], s=5, alpha=0.3, color="red",
                     label="$z \\sim \\mathcal{N}(0,I)$ (used for generation)")
    axes[2].set_title("Latent space $z$")
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


def plot_loss_curve(loss_history, filename="vae_loss_curve.png", window=200):
    """스텝별 loss(ELBO 음수 = reconstruction + KL) 곡선, 이동평균 함께 표시"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, alpha=0.25, linewidth=0.5, label="raw loss (per step)")
    smoothed = moving_average(loss_history, window=window)
    ax.plot(np.arange(len(smoothed)) + window, smoothed, color="red", linewidth=1.5,
            label=f"moving average (window={window})")
    ax.set_xlabel("training step")
    ax.set_ylabel("reconstruction + KL")
    ax.set_title("VAE training loss (negative ELBO)")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"결과 이미지 저장: {save_path}")
