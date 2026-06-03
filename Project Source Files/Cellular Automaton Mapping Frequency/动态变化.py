# ca_continuous_evolution.py
import numpy as np
import os
from enum import Enum
from scipy.io.wavfile import write

class CellStates(Enum):
    OFF = 0
    ON = 1

class TwoDimensionalCellularAutomaton:
    DEFAULT_INIT_ALIVE_PROB = 0.35
    TOROIDAL_BOUNDARY = True

    def __init__(self, height=16, width=16, init_alive_prob=None):
        self.height = height
        self.width = width
        self.init_alive_prob = init_alive_prob or self.DEFAULT_INIT_ALIVE_PROB
        self.grid = self._initialize_grid()
        self.history = [self.grid.copy()]  # 存储所有代的状态

        self._rules = {
            "conway": self._apply_conway_rule,
            "density_based": self._apply_density_based_rule,
        }
        self.current_rule = "conway"

    def _initialize_grid(self):
        return (np.random.rand(self.height, self.width) < self.init_alive_prob).astype(int)

    def step(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                neighbors = self._count_neighbors(i, j)
                new_grid[i, j] = self._apply_current_rule(self.grid[i, j], neighbors)
        self.grid = new_grid
        self.history.append(self.grid.copy())

    def evolve_generations(self, n_generations):
        """演化 n_generations 代，并记录历史"""
        for _ in range(n_generations):
            self.step()

    def _apply_current_rule(self, current_state, neighbors):
        rule_func = self._rules[self.current_rule]
        return rule_func(current_state, neighbors)

    def set_rule(self, rule_name):
        if rule_name in self._rules:
            self.current_rule = rule_name
        else:
            raise ValueError(f"未知规则: {rule_name}")

    def _count_neighbors(self, i, j):
        total = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if self.TOROIDAL_BOUNDARY:
                    ni %= self.height
                    nj %= self.width
                else:
                    if not (0 <= ni < self.height and 0 <= nj < self.width):
                        continue
                total += self.grid[ni, nj]
        return total

    def _apply_conway_rule(self, current_state, neighbors):
        if current_state == CellStates.ON.value:
            return CellStates.ON.value if neighbors in (2, 3) else CellStates.OFF.value
        else:
            return CellStates.ON.value if neighbors == 3 else CellStates.OFF.value

    def _apply_density_based_rule(self, current_state, neighbors):
        return CellStates.ON.value if neighbors >= 4 else CellStates.OFF.value


class ContinuousCAAudioRenderer:
    SAMPLE_RATE = 44100
    BASE_FREQUENCY = 55.0      # A1
    MAX_AMPLITUDE = 0.08       # 更低，因为所有格子同时发声

    def to_continuous_audio(self, ca_history, total_duration=10.0):
        """
        根据 CA 的完整演化历史生成一段连续音频
        :param ca_history: list of grids, shape (H, W)
        :param total_duration: 总时长（秒）
        :return: 单声道音频信号 (float32)
        """
        num_frames = len(ca_history)
        H, W = ca_history[0].shape
        samples_per_frame = int(self.SAMPLE_RATE * total_duration / num_frames)
        total_samples = num_frames * samples_per_frame

        t_full = np.linspace(0, total_duration, total_samples, endpoint=False)
        full_signal = np.zeros(total_samples, dtype=np.float32)

        # 为每个格子生成连续波形
        for i in range(H):
            for j in range(W):
                # 构建该格子的瞬时频率和振幅序列（按帧）
                freq_seq = []
                amp_seq = []
                for grid in ca_history:
                    if grid[i, j] == CellStates.ON.value:
                        # 频率由行 i 决定（指数分布）
                        octaves = 3.0
                        freq = self.BASE_FREQUENCY * (2 ** (i / max(1, H - 1) * octaves))
                        # 振幅由列 j 决定
                        amp = self.MAX_AMPLITUDE * (0.3 + 0.7 * (j / max(1, W - 1)))
                    else:
                        freq = 0.0
                        amp = 0.0
                    freq_seq.append(freq)
                    amp_seq.append(amp)

                freq_seq = np.array(freq_seq)
                amp_seq = np.array(amp_seq)

                # 扩展为每样本的频率/振幅（阶梯状）
                freq_per_sample = np.repeat(freq_seq, samples_per_frame)
                amp_per_sample = np.repeat(amp_seq, samples_per_frame)

                # 使用相位累积生成连续正弦波（避免相位跳变）
                phase = 2 * np.pi * np.cumsum(freq_per_sample) / self.SAMPLE_RATE
                wave = amp_per_sample * np.sin(phase)

                full_signal += wave

        # 全局归一化
        max_val = np.max(np.abs(full_signal)) + 1e-9
        if max_val > 1.0:
            full_signal = full_signal / max_val * 0.95

        return full_signal

def main():
    np.random.seed(2025)  # 可复现

    # 创建并演化 CA
    ca = TwoDimensionalCellularAutomaton(height=12, width=12, init_alive_prob=0.4)
    ca.evolve_generations(n_generations=80)  # 演化 80 代 → 共 81 帧（含初始）

    # 渲染连续音频
    renderer = ContinuousCAAudioRenderer()
    audio_signal = renderer.to_continuous_audio(
        ca_history=ca.history,
        total_duration=12.0  # 总时长 12 秒
    )

    # 保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "/Users/liukun/Desktop/元胞自动机频率/动态变化.wave")
    wav_data = (audio_signal * 32767).astype(np.int16)
    write(output_path, renderer.SAMPLE_RATE, wav_data)

    print(f"✅ 连续演化音频已生成：{output_path}")
    print(f"   时长: 12.0 秒 | 演化代数: {len(ca.history)} 代")
    print(f"   网格: {ca.history[0].shape}")

if __name__ == "__main__":
    main()