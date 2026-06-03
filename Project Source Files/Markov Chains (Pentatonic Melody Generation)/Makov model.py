import numpy as np
import os
import random
import time
from music21 import metadata, note, stream, midi, converter, pitch

# 自动获取桌面路径（适配Windows/macOS）
def get_desktop_path():
    if os.name == 'nt':  # Windows
        return os.path.join(os.path.expanduser('~'), 'Desktop')
    else:  # macOS/Linux
        return os.path.join(os.path.expanduser('~'), 'Desktop')

# 动态随机种子（每次运行不同）
np.random.seed(int(time.time()))
random.seed(int(time.time()))


class CMajorPentatonicMarkovGenerator:
    """C大调五声调式马尔可夫链旋律生成器（带概率文本输出）"""
    def __init__(self):
        # C大调五声调式音高（扩展八度）
        self.pentatonic_pitches = [
            'C3', 'D3', 'E3', 'G3', 'A3',
            'C4', 'D4', 'E4', 'G4', 'A4',
            'C5', 'D5', 'E5', 'G5', 'A5',
            'C6', 'D6', 'E6', 'G6', 'A6'
        ]
        # 扩展音符时长类型
        self.common_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        
        # 构建状态集（音高+时长）
        self.states = [(p, d) for p in self.pentatonic_pitches for d in self.common_durations]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        
        # 初始化概率矩阵
        self.initial_probs = np.zeros(len(self.states))
        self.transition_matrix = np.zeros((len(self.states), len(self.states)))

    def _is_pentatonic_note(self, note_obj):
        """验证音符是否属于C大调五声调式"""
        if not isinstance(note_obj, note.Note):
            return False
        pitch_name = note_obj.pitch.nameWithOctave
        base_pitch = pitch_name[:-1]
        return base_pitch in ['C', 'D', 'E', 'G', 'A']

    def load_desktop_midi_corpus(self):
        """自动加载桌面MIDI文件（过滤生成的文件）"""
        desktop_path = get_desktop_path()
        corpus_dir = os.path.join(desktop_path, '马尔可夫链(五声调式旋律生成)')
        
        if not os.path.exists(corpus_dir):
            raise FileNotFoundError(f"未找到文件夹：{corpus_dir}（请确认文件夹在桌面）")
        
        # 过滤生成的MIDI，只训练原始数据
        all_valid_notes = []
        exclude_files = ["生成的五声调式旋律.mid", "generated_melody.mid"] + [f"五声调式旋律_{time.strftime('%Y%m%d')}_*.mid"]
        for filename in os.listdir(corpus_dir):
            if any(exclude in filename for exclude in exclude_files):
                continue
            if filename.endswith('.mid') or filename.endswith('.midi'):
                midi_path = os.path.join(corpus_dir, filename)
                try:
                    midi_stream = converter.parse(midi_path)
                    notes = midi_stream.flat.notes
                    valid_notes = [n for n in notes if self._is_pentatonic_note(n)]
                    all_valid_notes.extend(valid_notes)
                    print(f"加载成功：{filename} → 提取{len(valid_notes)}个有效音符")
                except Exception as e:
                    print(f"处理{filename}失败：{str(e)}")
        
        if not all_valid_notes:
            raise ValueError("未从MIDI文件中提取到有效五声调式音符")
        return all_valid_notes

    def train(self, training_notes):
        """训练模型（优化概率计算）"""
        # ====================== 1. 计算初始概率 ======================
        initial_count = np.zeros(len(self.states))
        for note in training_notes:
            state = (note.pitch.nameWithOctave, note.duration.quarterLength)
            if state in self.state_to_idx:
                initial_count[self.state_to_idx[state]] += 1
        
        # 概率平滑 + 归一化（总和=1）
        self.initial_probs = (initial_count + 0.1) / (np.sum(initial_count) + 0.1 * len(self.states))

        # ====================== 2. 计算转移概率 ======================
        transition_count = np.zeros((len(self.states), len(self.states)))
        for i in range(len(training_notes)-1):
            curr_note = training_notes[i]
            next_note = training_notes[i+1]
            curr_state = (curr_note.pitch.nameWithOctave, curr_note.duration.quarterLength)
            next_state = (next_note.pitch.nameWithOctave, next_note.duration.quarterLength)
            if curr_state in self.state_to_idx and next_state in self.state_to_idx:
                curr_idx = self.state_to_idx[curr_state]
                next_idx = self.state_to_idx[next_state]
                transition_count[curr_idx][next_idx] += 1
        
        # 转移概率平滑 + 逐行归一化（每行总和=1）
        transition_count += 0.01  # 避免行和为0
        row_sums = transition_count.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.transition_matrix = transition_count / row_sums[:, None]

        # ====================== 3. 输出概率文本 ======================
        self._export_probability_text()

    def _export_probability_text(self):
        """将初始概率和转移概率导出为文本文件并打印关键信息"""
        desktop_path = get_desktop_path()
        corpus_dir = os.path.join(desktop_path, '马尔可夫链(五声调式旋律生成)')
        prob_file_path = os.path.join(corpus_dir, f"马尔可夫链概率分布_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(prob_file_path, 'w', encoding='utf-8') as f:
            # 写入初始概率
            f.write("="*80 + "\n")
            f.write("初始概率分布（音高+时长 → 概率，总和=1）\n")
            f.write("="*80 + "\n")
            # 按概率从高到低排序
            initial_sorted = sorted(
                [(self.idx_to_state[i], self.initial_probs[i]) for i in range(len(self.states))],
                key=lambda x: x[1],
                reverse=True
            )
            # 只写入概率>0.001的状态（减少冗余）
            initial_total = 0.0
            for state, prob in initial_sorted:
                if prob > 0.001:
                    f.write(f"状态：{state[0]}({state[1]}拍) → 概率：{prob:.6f}\n")
                    initial_total += prob
            f.write(f"显示的概率总和：{initial_total:.6f}（剩余为低概率状态）\n")
            f.write(f"初始概率总校验：{np.sum(self.initial_probs):.6f}（应为1）\n")

            # 写入转移概率（只写入非零且概率>0.01的转移）
            f.write("\n" + "="*80 + "\n")
            f.write("转移概率分布（当前状态 → 下一个状态：概率，每行总和=1）\n")
            f.write("="*80 + "\n")
            
            # 只输出概率最高的前20个当前状态的转移
            top_curr_states = np.argsort(self.initial_probs)[-20:][::-1]
            for curr_idx in top_curr_states:
                curr_state = self.idx_to_state[curr_idx]
                trans_probs = self.transition_matrix[curr_idx]
                # 按转移概率排序
                trans_sorted = sorted(
                    [(self.idx_to_state[i], trans_probs[i]) for i in range(len(self.states))],
                    key=lambda x: x[1],
                    reverse=True
                )
                # 写入当前状态
                f.write(f"\n当前状态：{curr_state[0]}({curr_state[1]}拍) → 转移路径：\n")
                # 只写入概率>0.01的转移
                trans_total = 0.0
                for next_state, prob in trans_sorted:
                    if prob > 0.01:
                        f.write(f"  → {next_state[0]}({next_state[1]}拍)：概率={prob:.6f}\n")
                        trans_total += prob
                f.write(f"  该状态转移概率总和：{trans_total:.6f}（剩余为低概率转移）\n")
                f.write(f"  该行总校验：{np.sum(trans_probs):.6f}（应为1）\n")

        # 控制台打印关键概率信息
        print("\n" + "="*80)
        print("【初始概率 TOP10】（概率从高到低）")
        print("="*80)
        initial_sorted = sorted(
            [(self.idx_to_state[i], self.initial_probs[i]) for i in range(len(self.states))],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for i, (state, prob) in enumerate(initial_sorted, 1):
            print(f"{i}. 音高：{state[0]}，时长：{state[1]}拍 → 概率：{prob:.4f}")
        
        print("\n" + "="*80)
        print("【转移概率示例】（C5 1.5拍的TOP5转移路径）")
        print("="*80)
        target_state = ("C5", 1.5)
        if target_state in self.state_to_idx:
            target_idx = self.state_to_idx[target_state]
            trans_probs = self.transition_matrix[target_idx]
            trans_sorted = sorted(
                [(self.idx_to_state[i], trans_probs[i]) for i in range(len(self.states))],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for i, (next_state, prob) in enumerate(trans_sorted, 1):
                print(f"{i}. → 音高：{next_state[0]}，时长：{next_state[1]}拍 → 概率：{prob:.4f}")
        else:
            print("未找到C5 1.5拍的状态，显示概率最高状态的转移")
            top_idx = np.argmax(self.initial_probs)
            top_state = self.idx_to_state[top_idx]
            trans_probs = self.transition_matrix[top_idx]
            trans_sorted = sorted(
                [(self.idx_to_state[i], trans_probs[i]) for i in range(len(self.states))],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print(f"最高概率状态：{top_state[0]}({top_state[1]}拍) 的转移路径：")
            for i, (next_state, prob) in enumerate(trans_sorted, 1):
                print(f"{i}. → 音高：{next_state[0]}，时长：{next_state[1]}拍 → 概率：{prob:.4f}")
        
        print(f"\n完整概率分布已保存到：{prob_file_path}")
        print("="*80 + "\n")

    def generate_melody(self, length=24, temperature=0.8):
        """生成旋律（增加温度参数控制随机性）"""
        if length <= 0:
            raise ValueError("旋律长度需大于0")
        
        # 生成起始状态（加入温度调整）
        initial_probs_adjusted = np.power(self.initial_probs, 1/temperature)
        initial_probs_adjusted /= initial_probs_adjusted.sum()
        start_idx = np.random.choice(len(self.states), p=initial_probs_adjusted)
        melody = [self.idx_to_state[start_idx]]
        
        # 生成后续状态（加入温度调整）
        for _ in range(length-1):
            curr_state = melody[-1]
            curr_idx = self.state_to_idx[curr_state]
            trans_probs = self.transition_matrix[curr_idx]
            
            # 温度调整：让概率分布更分散
            trans_probs_adjusted = np.power(trans_probs, 1/temperature)
            trans_probs_adjusted /= trans_probs_adjusted.sum()
            
            next_idx = np.random.choice(len(self.states), p=trans_probs_adjusted)
            melody.append(self.idx_to_state[next_idx])
        return melody

    def save_and_show_melody(self, melody, output_name=None):
        """保存MIDI（每次生成不同文件名）"""
        # 生成唯一文件名（避免覆盖）
        if output_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"五声调式旋律_{timestamp}.mid"
        
        # 构建乐谱
        score = stream.Score(title="C大调五声调式马尔可夫旋律")
        part = stream.Part()
        for pitch_name, duration in melody:
            part.append(note.Note(pitch_name, quarterLength=duration))
        score.append(part)
        
        # 保存MIDI
        desktop_path = get_desktop_path()
        corpus_dir = os.path.join(desktop_path, '马尔可夫链(五声调式旋律生成)')
        output_path = os.path.join(corpus_dir, output_name)
        mf = midi.translate.streamToMidiFile(score)
        mf.open(output_path, 'wb')
        mf.write()
        mf.close()
        print(f"\n旋律已保存到：{output_path}")
        
        # 打印旋律序列
        print("\n生成的旋律序列（音高+时长）：")
        for i, (p, d) in enumerate(melody, 1):
            print(f"第{i}个音符：{p}（时长：{d}拍）")
        print("\n提示：如需可视化乐谱，可安装MuseScore并配置路径")


def main():
    try:
        generator = CMajorPentatonicMarkovGenerator()
        # 加载训练数据
        print("正在加载桌面MIDI数据集...")
        training_notes = generator.load_desktop_midi_corpus()
        # 训练模型（自动输出概率文本）
        print("\n正在训练马尔可夫链模型...")
        generator.train(training_notes)
        
        # 自定义参数
        while True:
            try:
                length = int(input("\n请输入旋律长度（音符数量，建议16-64）："))
                if length > 0:
                    break
                else:
                    print("长度必须大于0！")
            except ValueError:
                print("请输入有效的数字！")
        
        while True:
            try:
                temperature = float(input("请输入随机性温度（0.5-2.0，越大越随机）："))
                if 0.1 <= temperature <= 5.0:
                    break
                else:
                    print("温度请在0.1-5.0之间！")
            except ValueError:
                print("请输入有效的数字！")
        
        # 生成旋律
        print(f"\n正在生成{length}个音符的旋律（温度={temperature}）...")
        generated_melody = generator.generate_melody(length=length, temperature=temperature)
        # 保存结果
        generator.save_and_show_melody(generated_melody)
    except Exception as e:
        print(f"程序出错：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 依赖安装：pip install music21 numpy
    main()