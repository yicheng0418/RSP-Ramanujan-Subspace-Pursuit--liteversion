import os
import math
import argparse
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def ramanujan_sum_sequence(q: int) -> np.ndarray:
    """
    生成长度为 q 的 Ramanujan sum 基序列 c_q(n), n=0..q-1
    c_q(n) = sum_{1<=a<=q, gcd(a,q)=1} exp(2πi a n / q)

    理论上结果应为实整数，这里取实部。
    """
    n = np.arange(q)
    seq = np.zeros(q, dtype=np.complex128)
    for a in range(1, q + 1):
        if gcd(a, q) == 1:
            seq += np.exp(2j * np.pi * a * n / q)
    return np.real(seq)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.mean(x)
    maxv = np.max(np.abs(x)) + 1e-12
    return x / maxv


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    将一维信号切成帧。
    """
    if len(x) < frame_length:
        x = np.pad(x, (0, frame_length - len(x)))

    num_frames = 1 + (len(x) - frame_length) // hop_length
    frames = np.stack([
        x[i * hop_length:i * hop_length + frame_length]
        for i in range(num_frames)
    ])
    return frames


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    读取音频并重采样到 target_sr，转为单声道。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    y, sr = sf.read(audio_path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    y = normalize_audio(y)
    return y, sr


def ramanujan_periodogram_for_frame(
    frame: np.ndarray,
    q_min: int,
    q_max: int,
    cache: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单帧计算每个候选周期 q 的 Ramanujan 投影分数。
    """
    scores = []
    N = len(frame)

    for q in range(q_min, q_max + 1):
        if q not in cache:
            cache[q] = ramanujan_sum_sequence(q)

        cq = cache[q]
        tiled = np.resize(cq, N)
        score = np.abs(np.dot(frame, tiled)) ** 2
        scores.append(score)

    return np.arange(q_min, q_max + 1), np.array(scores, dtype=np.float64)


def extract_ramanujan_track(
    y: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    fmin: float = 80.0,
    fmax: float = 400.0
) -> dict:
    """
    提取 Ramanujan 主周期轨迹。
    """
    frame_length = int(sr * frame_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)

    frames = frame_signal(y, frame_length, hop_length)

    q_min = max(2, int(sr / fmax))
    q_max = max(q_min + 1, int(sr / fmin))

    cache: dict[int, np.ndarray] = {}
    dominant_q = []
    dominant_score = []
    second_score = []
    all_scores = []

    window = np.hanning(frame_length)

    for frame in frames:
        frame_w = frame * window
        q_values, scores = ramanujan_periodogram_for_frame(frame_w, q_min, q_max, cache)

        idx_sorted = np.argsort(scores)[::-1]
        idx1 = idx_sorted[0]
        idx2 = idx_sorted[1] if len(idx_sorted) > 1 else idx_sorted[0]

        dominant_q.append(q_values[idx1])
        dominant_score.append(scores[idx1])
        second_score.append(scores[idx2])
        all_scores.append(scores)

    dominant_q = np.array(dominant_q, dtype=np.float64)
    dominant_score = np.array(dominant_score, dtype=np.float64)
    second_score = np.array(second_score, dtype=np.float64)
    all_scores = np.vstack(all_scores)

    dominant_f0 = sr / dominant_q
    times = np.arange(len(dominant_q)) * (hop_length / sr)

    return {
        "times": times,
        "dominant_q": dominant_q,
        "dominant_f0": dominant_f0,
        "dominant_score": dominant_score,
        "second_score": second_score,
        "q_min": q_min,
        "q_max": q_max,
        "all_scores": all_scores,
        "frame_length": frame_length,
        "hop_length": hop_length,
    }


def compute_ramanujan_psi(track: dict) -> dict:
    """
    计算 Ramanujan 周期稳定度指数。
    """
    q = track["dominant_q"]
    s1 = track["dominant_score"]
    s2 = track["second_score"]

    if len(q) < 3:
        return {
            "psi_ram": 0.0,
            "mean_delta_q": None,
            "mean_peak_ratio": None,
            "entropy_q": None,
            "score_delta": 0.0,
            "score_peak": 0.0,
            "score_entropy": 0.0,
        }

    eps = 1e-12

    # 主周期相邻变化率
    delta_q = np.abs(np.diff(q)) / (q[:-1] + eps)
    mean_delta_q = float(np.mean(delta_q))

    # 第一峰与第二峰比值
    peak_ratio = s1 / (s2 + eps)
    mean_peak_ratio = float(np.mean(peak_ratio))

    # 主周期分布熵
    q_int = q.astype(int)
    _, counts = np.unique(q_int, return_counts=True)
    p = counts / counts.sum()
    entropy_q = float(-np.sum(p * np.log(p + eps)))

    # 经验分数映射，后续需要你自己校准
    score_delta = 100.0 * np.exp(-((mean_delta_q - 0.015) ** 2) / (2 * 0.01 ** 2))
    score_peak = 100.0 * np.exp(-((mean_peak_ratio - 2.5) ** 2) / (2 * 1.2 ** 2))
    score_entropy = 100.0 * np.exp(-((entropy_q - 1.2) ** 2) / (2 * 0.8 ** 2))

    psi_ram = 0.4 * score_delta + 0.3 * score_peak + 0.3 * score_entropy

    return {
        "psi_ram": float(np.clip(psi_ram, 0, 100)),
        "mean_delta_q": mean_delta_q,
        "mean_peak_ratio": mean_peak_ratio,
        "entropy_q": entropy_q,
        "score_delta": float(np.clip(score_delta, 0, 100)),
        "score_peak": float(np.clip(score_peak, 0, 100)),
        "score_entropy": float(np.clip(score_entropy, 0, 100)),
    }


def save_framewise_csv(track: dict, output_csv: str) -> None:
    """
    保存逐帧结果。
    """
    df = pd.DataFrame({
        "time_sec": track["times"],
        "dominant_q": track["dominant_q"],
        "dominant_f0_hz": track["dominant_f0"],
        "dominant_score": track["dominant_score"],
        "second_score": track["second_score"],
    })
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def plot_result(
    y: np.ndarray,
    sr: int,
    track: dict,
    psi_result: dict,
    output_png: str,
    title: str = "Ramanujan Voice Analysis"
) -> None:
    """
    输出图像：波形 + 主频轨迹
    """
    duration = len(y) / sr
    t_audio = np.linspace(0, duration, len(y), endpoint=False)

    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t_audio, y, linewidth=0.8)
    ax1.set_title(title)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(track["times"], track["dominant_f0"], linewidth=1.2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Dominant F0 (Hz)")
    ax2.grid(True, alpha=0.3)

    summary_text = (
        f"PSI_RAM = {psi_result['psi_ram']:.2f}\n"
        f"mean_delta_q = {psi_result['mean_delta_q']:.6f}\n"
        f"mean_peak_ratio = {psi_result['mean_peak_ratio']:.6f}\n"
        f"entropy_q = {psi_result['entropy_q']:.6f}\n"
        f"score_delta = {psi_result['score_delta']:.2f}\n"
        f"score_peak = {psi_result['score_peak']:.2f}\n"
        f"score_entropy = {psi_result['score_entropy']:.2f}"
    )
    ax2.text(
        0.02, 0.98, summary_text,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def analyze_audio_file(audio_path: str, output_dir: str) -> dict:
    """
    分析单个音频文件并输出结果。
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}_ramanujan_frames.csv")
    png_path = os.path.join(output_dir, f"{base_name}_ramanujan_plot.png")
    summary_path = os.path.join(output_dir, f"{base_name}_ramanujan_summary.txt")

    y, sr = load_audio(audio_path, target_sr=16000)
    track = extract_ramanujan_track(y, sr=sr)
    psi_result = compute_ramanujan_psi(track)

    save_framewise_csv(track, csv_path)
    plot_result(y, sr, track, psi_result, png_path, title=base_name)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_path}\n")
        f.write(f"PSI_RAM: {psi_result['psi_ram']:.6f}\n")
        f.write(f"mean_delta_q: {psi_result['mean_delta_q']}\n")
        f.write(f"mean_peak_ratio: {psi_result['mean_peak_ratio']}\n")
        f.write(f"entropy_q: {psi_result['entropy_q']}\n")
        f.write(f"score_delta: {psi_result['score_delta']:.6f}\n")
        f.write(f"score_peak: {psi_result['score_peak']:.6f}\n")
        f.write(f"score_entropy: {psi_result['score_entropy']:.6f}\n")
        f.write(f"CSV: {csv_path}\n")
        f.write(f"PNG: {png_path}\n")

    return {
        "audio_path": audio_path,
        "csv_path": csv_path,
        "png_path": png_path,
        "summary_path": summary_path,
        **psi_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Ramanujan 周期稳定度分析")
    parser.add_argument("input_audio", help="输入音频文件路径")
    parser.add_argument(
        "--output_dir",
        default="ramanujan_output",
        help="输出目录，默认 ramanujan_output"
    )
    args = parser.parse_args()

    result = analyze_audio_file(args.input_audio, args.output_dir)

    print("分析完成：")
    print(f"输入文件: {result['audio_path']}")
    print(f"PSI_RAM: {result['psi_ram']:.4f}")
    print(f"mean_delta_q: {result['mean_delta_q']}")
    print(f"mean_peak_ratio: {result['mean_peak_ratio']}")
    print(f"entropy_q: {result['entropy_q']}")
    print(f"CSV 输出: {result['csv_path']}")
    print(f"图像输出: {result['png_path']}")
    print(f"摘要输出: {result['summary_path']}")


if __name__ == "__main__":
    main()
