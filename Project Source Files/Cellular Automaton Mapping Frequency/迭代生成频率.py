# 第一步：导入依赖（仅保留必要库，移除中文相关设置）
import numpy as np
import os
from enum import Enum
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# 第二步：定义细胞状态枚举
class CellStates(Enum):
    OFF = 0
    ON = 1

# 第三步：二维元胞自动机核心类（保持不变）
class TwoDimensionalCellularAutomaton:
    DEFAULT_INIT_ALIVE_PROB = 0.35
    TOROIDAL_BOUNDARY = True

    def __init__(self, height=16, width=16, init_alive_prob=None):
        self.height = height
        self.width = width
        self.init_alive_prob = init_alive_prob or self.DEFAULT_INIT_ALIVE_PROB
        self.grid = self._initialize_grid()

        # 注册规则
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

    def _apply_current_rule(self, current_state, neighbors):
        rule_func = self._rules[self.current_rule]
        return rule_func(current_state, neighbors)

    def set_rule(self, rule_name):
        if rule_name in self._rules:
            self.current_rule = rule_name
        else:
            raise ValueError(f"Unknown rule: {rule_name}")

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

# 第四步：图片导出工具类（改为英文标签，优化字体）
class CAImageExporter:
    def __init__(self, output_dir="CA_Evolution_Images"):
        # 创建图片输出目录（不存在则创建）
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 设置图片大小和字体（使用matplotlib默认英文兼容字体）
        self.fig_size = (8, 8)
        self.title_fontsize = 16
        self.axis_fontsize = 14
        self.tick_fontsize = 12

    def export_grid(self, grid, generation, rule_name):
        """
        Export current generation's grid as an image
        :param grid: 2D numpy array of cell states
        :param generation: current generation number (int)
        :param rule_name: name of the evolution rule (str)
        :return: saved image path (str)
        """
        # Create new plot instance
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot grid (black = ON, white = OFF)
        im = ax.imshow(grid, cmap="binary", interpolation="nearest", aspect="equal")
        
        # Set title (rule + generation)
        ax.set_title(f"2D Cellular Automaton - Rule: {rule_name} (Generation {generation})", 
                     fontsize=self.title_fontsize, pad=20)
        
        # Set axis labels (English)
        ax.set_xlabel("Column Index", fontsize=self.axis_fontsize, labelpad=10)
        ax.set_ylabel("Row Index", fontsize=self.axis_fontsize, labelpad=10)
        
        # Set ticks (show all cell indices)
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Add grid lines for better cell separation
        ax.grid(True, color="gray", linewidth=0.8, alpha=0.6)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save image (PNG format, high resolution)
        img_filename = f"CA_Rule_{rule_name}_Gen_{generation:02d}.png"
        img_path = os.path.join(self.output_dir, img_filename)
        plt.savefig(img_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        
        # Close plot to free memory
        plt.close(fig)
        
        return img_path

# 第五步：音频渲染器（带淡入淡出，增加频率记录功能）
class CASineWaveAudioRenderer:
    SAMPLE_RATE = 44100
    BASE_FREQUENCY = 55.0      # A1
    MAX_AMPLITUDE = 0.12       # Reduce amplitude to avoid clipping

    def to_audio_signal(self, grid, duration=0.5, record_frequencies=False):
        """
        Convert grid state to audio signal with fade in/out
        :param grid: 2D numpy array
        :param duration: audio duration in seconds
        :param record_frequencies: whether to return frequency details (bool)
        :return: numpy array (float32) + optional list of (row, col, frequency)
        """
        H, W = grid.shape
        total_samples = int(self.SAMPLE_RATE * duration)
        if total_samples == 0:
            if record_frequencies:
                return np.array([], dtype=np.float32), []
            return np.array([], dtype=np.float32)
        
        t = np.linspace(0, duration, total_samples, endpoint=False)
        signal = np.zeros_like(t, dtype=np.float32)
        frequency_records = []  # 记录每个活跃元胞的频率信息

        # Generate sine waves for active cells
        for i in range(H):
            for j in range(W):
                if grid[i, j] == CellStates.ON.value:
                    octaves = 3.0
                    # 计算频率：行索引决定八度，列索引决定同一八度内的幅度
                    freq = self.BASE_FREQUENCY * (2 ** (i / max(1, H - 1) * octaves))
                    amp = self.MAX_AMPLITUDE * (0.3 + 0.7 * (j / max(1, W - 1)))
                    wave = amp * np.sin(2 * np.pi * freq * t)
                    signal += wave
                    
                    # 记录频率信息（保留2位小数）
                    if record_frequencies:
                        frequency_records.append((i, j, round(freq, 2)))

        # Add fade in/out to avoid click noise
        fade_duration = 0.02  # 20ms
        fade_samples = min(int(self.SAMPLE_RATE * fade_duration), total_samples // 2)

        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            signal[:fade_samples] *= fade_in
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            signal[-fade_samples:] *= fade_out

        # Normalize single frame to prevent overload
        max_val = np.max(np.abs(signal)) + 1e-9
        if max_val > 1.0:
            signal = signal / max_val * 0.95

        if record_frequencies:
            return signal, frequency_records
        return signal

# 第六步：主程序（增加频率输出功能）
def main():
    # Fix random seed for reproducibility (comment out for randomness)
    np.random.seed(2025)

    # Create cellular automaton instance
    ca = TwoDimensionalCellularAutomaton(height=16, width=16, init_alive_prob=0.4)
    
    # Switch rule (uncomment to use density-based rule)
    # ca.set_rule("density_based")

    # Create image exporter (images saved to "CA_Evolution_Images" folder)
    image_exporter = CAImageExporter(output_dir="CA_Evolution_Images")

    renderer = CASineWaveAudioRenderer()

    # Configuration parameters
    TOTAL_GENERATIONS = 20      # Total number of generations (including initial state)
    DURATION_PER_GEN = 0.4      # Audio duration per generation (seconds)
    SAVE_FREQUENCY_LOG = True   # 是否保存频率日志到文本文件

    print("🔊 Generating evolution audio (with fade in/out)...")
    print("📊 Recording frequency values for active cells...")
    print("🖼️ Exporting images for each generation...")
    audio_frames = []
    all_frequency_logs = []     # 保存所有代的频率日志

    # 初始化频率日志文件
    if SAVE_FREQUENCY_LOG:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(script_dir, "Cell_Frequency_Log.txt")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Cellular Automaton Frequency Log\n")
            f.write(f"Rule: {ca.current_rule}\n")
            f.write(f"Grid Size: {ca.height}x{ca.width}\n")
            f.write(f"Total Generations: {TOTAL_GENERATIONS}\n")
            f.write(f"Base Frequency: {renderer.BASE_FREQUENCY} Hz\n")
            f.write("-" * 60 + "\n\n")

    # Process each generation (including generation 0)
    for gen in range(TOTAL_GENERATIONS):
        # 1. Export current generation image
        img_path = image_exporter.export_grid(ca.grid, generation=gen, rule_name=ca.current_rule)
        
        # 2. Generate audio and record frequencies
        frame, freq_records = renderer.to_audio_signal(
            ca.grid, 
            duration=DURATION_PER_GEN,
            record_frequencies=True  # 启用频率记录
        )
        audio_frames.append(frame)
        
        # 3. 记录并打印频率信息
        active_cell_count = len(freq_records)
        all_frequency_logs.append({
            "generation": gen,
            "active_cells": active_cell_count,
            "frequencies": freq_records
        })
        
        # 打印当前代的频率摘要
        print(f"\n=== Generation {gen} ===")
        print(f"Active Cells Count: {active_cell_count}")
        if active_cell_count > 0:
            print("Active Cells (Row, Column) → Frequency (Hz):")
            # 按频率升序排序输出
            freq_records_sorted = sorted(freq_records, key=lambda x: x[2])
            for row, col, freq in freq_records_sorted:
                print(f"  ({row:2d}, {col:2d}) → {freq:6.2f} Hz")
            # 输出统计信息
            frequencies = [x[2] for x in freq_records]
            print(f"Frequency Range: {min(frequencies):.2f} - {max(frequencies):.2f} Hz")
            print(f"Average Frequency: {np.mean(frequencies):.2f} Hz")
        else:
            print("No active cells → No audio frequency")
        print(f"Image saved to: {img_path}")
        
        # 4. Evolve to next generation (skip after last generation)
        if gen < TOTAL_GENERATIONS - 1:
            ca.step()

    # 5. 保存频率日志到文本文件
    if SAVE_FREQUENCY_LOG:
        with open(log_file_path, "a", encoding="utf-8") as f:
            for log in all_frequency_logs:
                f.write(f"=== Generation {log['generation']} ===\n")
                f.write(f"Active Cells Count: {log['active_cells']}\n")
                if log['active_cells'] > 0:
                    f.write("Row\tColumn\tFrequency (Hz)\n")
                    # 按行、列排序保存
                    freq_records_sorted = sorted(log['frequencies'], key=lambda x: (x[0], x[1]))
                    for row, col, freq in freq_records_sorted:
                        f.write(f"{row}\t{col}\t{freq:.2f}\n")
                    frequencies = [x[2] for x in log['frequencies']]
                    f.write(f"Frequency Range: {min(frequencies):.2f} - {max(frequencies):.2f} Hz\n")
                    f.write(f"Average Frequency: {np.mean(frequencies):.2f} Hz\n")
                else:
                    f.write("No active cells\n")
                f.write("-" * 40 + "\n")
        print(f"\n📄 Frequency log saved to: {log_file_path}")

    # 6. 生成并保存完整音频
    full_audio = np.concatenate(audio_frames)

    # Global normalization for safety
    max_val = np.max(np.abs(full_audio)) + 1e-9
    if max_val > 1.0:
        full_audio = full_audio / max_val * 0.95

    # Save audio file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "/Users/liukun/Desktop/元胞自动机频率/迭代生成频率.wav")
    
    wav_data = (full_audio * 32767).astype(np.int16)
    write(output_path, renderer.SAMPLE_RATE, wav_data)

    total_time = TOTAL_GENERATIONS * DURATION_PER_GEN
    print(f"\n🎉 All tasks completed successfully!")
    print(f"📁 Audio file saved to: {output_path}")
    print(f"🖼️ Images saved to: {image_exporter.output_dir} (Total {TOTAL_GENERATIONS} images)")
    print(f"⏱️ Total audio duration: {total_time:.1f} seconds ({TOTAL_GENERATIONS} gens × {DURATION_PER_GEN} sec/gen)")
    print("🎧 Each audio frame has 20ms fade in/out to eliminate click noise")
    if SAVE_FREQUENCY_LOG:
        print(f"📊 Frequency details saved to text file: {log_file_path}")

# Execution entry
if __name__ == "__main__":
    # Check required libraries
    required_libs = ["numpy", "scipy", "matplotlib"]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"⚠️ Missing required libraries. Please install first:")
        print(f"pip install {' '.join(missing_libs)}")
        exit(1)
    
    main()