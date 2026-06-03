"""
模块一：2D 元胞自动机状态空间	
一个 H x W 的网格，每个格子代表一个“声源单元”或“频率控制单元”
模块二：2CA 规则	
类似 Conway's Game of Life，但可以自定义（比如基于邻居数量决定是否发声、频率高低等）
模块三：
映射到音频参数	将每个格子的状态 → 映射为某个正弦波的 频率、振幅、相位、持续时间
模块四：
音频合成	叠加所有正弦波，生成最终音频信号（使用 numpy + scipy.io.wavfile 或 pydub 等）
"""

# 第一步：导入依赖（先写所有需要的库，避免中间插导入）
import numpy as np
import random
from enum import Enum
from scipy.io.wavfile import write

# 第二步：定义基础枚举（程序的“规矩”，先定好再用）
class CellStates(Enum):
    OFF = 0
    ON = 1

# 第三步：定义二维元胞自动机核心类（按“初始化→核心方法→辅助方法→规则方法”排序）
class TwoDimensionalCellularAutomaton:
    # 固定参数
    DEFAULT_INIT_ALIVE_PROB = 0.35  # 初始激活概率
    TOROIDAL_BOUNDARY = True        # 是否环形边界（首尾相连）

    def __init__(self, height=32, width=32, init_alive_prob=None):
        self.height = height
        self.width = width
        self.init_alive_prob = init_alive_prob or self.DEFAULT_INIT_ALIVE_PROB
        self.grid = self._initialize_grid()

        # 注册演化规则
        self._rules = {
            "conway": self._apply_conway_rule,
            "density_based": self._apply_density_based_rule,
        }
        self.current_rule = "conway"  # 默认使用 Conway 生命游戏规则

    def _initialize_grid(self):
        """生成初始随机网格（0=OFF, 1=ON）"""
        return (np.random.rand(self.height, self.width) < self.init_alive_prob).astype(int)

    def step(self):
        """执行一次演化，生成新网格"""
        new_grid = np.zeros_like(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                neighbors = self._count_neighbors(i, j)
                new_grid[i, j] = self._apply_current_rule(self.grid[i, j], neighbors)
        self.grid = new_grid

    def _apply_current_rule(self, current_state, neighbors):
        """根据当前规则计算新状态"""
        rule_func = self._rules[self.current_rule]
        return rule_func(current_state, neighbors)

    def set_rule(self, rule_name):
        """切换演化规则"""
        if rule_name in self._rules:
            self.current_rule = rule_name
        else:
            raise ValueError(f"未知规则: {rule_name}")

    def _count_neighbors(self, i, j):
        """计算8邻域中 ON 状态的数量（支持环形边界）"""
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

    # —————— 演化规则 ——————
    def _apply_conway_rule(self, current_state, neighbors):
        """经典 Conway's Game of Life 规则"""
        if current_state == CellStates.ON.value:
            return CellStates.ON.value if neighbors in (2, 3) else CellStates.OFF.value
        else:
            return CellStates.ON.value if neighbors == 3 else CellStates.OFF.value

    def _apply_density_based_rule(self, current_state, neighbors):
        """自定义规则：邻居数 ≥4 则激活"""
        return CellStates.ON.value if neighbors >= 4 else CellStates.OFF.value


# 第四步：定义音频合成类（将2D网格映射为正弦波叠加）
class CASineWaveAudioRenderer:
    # 音频参数（可调）
    SAMPLE_RATE = 44100      # 采样率（Hz）
    DURATION = 5.0           # 音频时长（秒）
    BASE_FREQUENCY = 55.0    # 最低频率（Hz），A1
    MAX_AMPLITUDE = 0.15     # 单个正弦波最大振幅（防爆音）

    def to_audio_signal(self, grid):
        """
        将2D网格转换为单声道音频信号（numpy array, float32, [-1, 1]）
        映射逻辑：
          - 行 i → 频率（指数分布，覆盖多个八度）
          - 列 j → 振幅（线性，左弱右强）
          - 仅当 grid[i, j] == ON 时发声
        """
        H, W = grid.shape
        t = np.linspace(0, self.DURATION, int(self.SAMPLE_RATE * self.DURATION), endpoint=False)
        signal = np.zeros_like(t, dtype=np.float32)

        for i in range(H):
            for j in range(W):
                if grid[i, j] == CellStates.ON.value:
                    # 频率：从 BASE_FREQUENCY 到 BASE_FREQUENCY * 2^(octaves)
                    octaves = 3.0  # 覆盖3个八度
                    freq = self.BASE_FREQUENCY * (2 ** (i / max(1, H - 1) * octaves))
                    
                    # 振幅：随列位置增强（避免边缘太弱）
                    amp = self.MAX_AMPLITUDE * (0.3 + 0.7 * (j / max(1, W - 1)))
                    
                    # 生成并叠加正弦波
                    wave = amp * np.sin(2 * np.pi * freq * t)
                    signal += wave

        # 归一化：防止削波（留5%余量）
        max_val = np.max(np.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val * 0.95

        return signal

    def save_to_wav(self, signal, filename="ca_drone_music.wav"):
        """保存为16位单声道WAV文件"""
        wav_data = (signal * 32767).astype(np.int16)
        write(filename, self.SAMPLE_RATE, wav_data)
        return filename


# 第五步：程序入口
def main():
    # 1. 创建2D元胞自动机（16x16网格）
    ca = TwoDimensionalCellularAutomaton(height=16, width=16, init_alive_prob=0.4)

    # 可选：切换规则（默认是 conway）
    # ca.set_rule("density_based")

    # 2. 演化若干代（让图案稳定或形成有趣结构）
    for gen in range(12):
        ca.step()
        print(f"第 {gen + 1} 代演化完成")

    # 3. 创建音频渲染器并生成信号
    renderer = CASineWaveAudioRenderer()
    audio_signal = renderer.to_audio_signal(ca.grid)

    # 4. 保存为WAV文件
    output_file = renderer.save_to_wav(audio_signal, "/Users/liukun/Desktop/元胞自动机频率/元胞自动机频率0.0.py.wav")
    print(f"\n✅ 音频已生成：{output_file}")
    print(f"   网格尺寸: {ca.grid.shape}, 时长: {renderer.DURATION}秒")


# 执行程序
if __name__ == "__main__":
    main()