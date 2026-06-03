import random
from dataclasses import dataclass

import music21


@dataclass(frozen=True) 
class MelodyData: # 封装旋律数据，自动计算旋律总时长（累加音符时长）和总小节数（4/4 拍下时长 //4）
    notes: list # 存储旋律的 “原始音符清单” 列表里的每个元素是 (音高，时长) 格式的元组（
    duration: int = None  # 旋律的 “总时长(拍子)”（自动算） 默认值 None（表示初始化时先不赋值）。
    number_of_bars: int = None  # 旋律的 “总小节数”（自动算）

    def __post_init__(self): # 在初始化后自动计算并赋值 duration（总时长）和 number_of_bars（总小节数），同时绕过不可变类的赋值限制。
        object.__setattr__(self, "duration", sum(duration for _, duration in self.notes) ) # 其中 _, duration 是 “解包元组”：self.notes 里每个元素是 (音高, 时长) 元组（如 ("C5", 1)），_ 是占位符（表示 “不用关心音高”），只取第二个值 duration 累加。
                                                                                # 示例：如果 self.notes = [("C5",1), ("G5",1), ("A5",2)]，求和结果就是 1+1+2=4，即 duration=4。
                                                                                # 最终效果：给当前 MelodyData 实例的 duration 属性赋值为 “所有音符时长的总和”。
       
        object.__setattr__(self, "number_of_bars", self.duration // 4) # self.duration // 4：假设旋律是 4/4 拍（音乐中最常见的拍号，含义是 “每小节有 4 拍”），所以 “总小节数” = 总时长（duration） ÷ 4，用 // 表示 “整数除法”（舍弃余数，确保小节数是整数）。示例：如果 duration=4，则 4//4=1（1 个小节）；如果 duration=48（如代码中《小星星》的总时长），则 48//4=12（12 个小节）。
                                                                       # 最终效果：给当前 MelodyData 实例的 number_of_bars 属性赋值为 “总时长除以 4 的整数结果”。


class GeneticMelodyHarmonizer:  # 使用遗传算法为给定旋律生成和弦伴奏。它通过进化和弦序列种群，基于适应度函数找到最适合该旋律的和弦序列

    def __init__(
        self,
        melody_data, # 包含旋律信息的数据
        chords, # 用于生成序列的可用和弦
        population_size, # 和弦序列种群的规模
        mutation_rate, # 遗传算法中的突变概率
        fitness_evaluator,  # 用于评估适应度的实例
    ):

        self.melody_data = melody_data # 表示“当前要配和弦的旋律数据”。
        self.chords = chords # 表示 “生成和弦序列时可以用的‘和弦库’”
        self.mutation_rate = mutation_rate # 表示 “遗传算法中‘突变’的概率”。
        self.population_size = population_size # 表示 “遗传算法中‘种群’的规模”（即一次同时处理多少个和弦序列）。
        self.fitness_evaluator = fitness_evaluator # 表示 “用来评估和弦序列‘好不好’的工具”（即适应度评估器）。
        self._population = [] # 用于存储 “当前代的所有和弦序列”。[]：初始化为空列表，因为刚开始还没有生成任何和弦序列。

    def generate(self, generations=1000):
      
        self._population = self._initialise_population()
        for _ in range(generations):
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_chord_sequence = (
            self.fitness_evaluator.get_chord_sequence_with_highest_fitness(
                self._population
            )
        )
        return best_chord_sequence


    def _initialise_population(self): # 使用随机和弦序列初始化种群。
        return [
            self._generate_random_chord_sequence()
            for _ in range(self.population_size)
        ]
 
    def _generate_random_chord_sequence(self): # 生成一个随机的和弦序列，和弦的数量与旋律的小节数相同
        return [
            random.choice(self.chords)
            for _ in range(self.melody_data.number_of_bars)
        ]

    def _select_parents(self): # 根据适应度选择用于繁殖的父序列
        fitness_values = [
            self.fitness_evaluator.evaluate(seq) for seq in self._population
        ]
        return random.choices(
            self._population, weights=fitness_values, k=self.population_size
        )

    def _create_new_population(self, parents):
        """
        从提供的父代中生成新的和弦序列种群。
        该方法使用交叉和变异操作创建新一代的和弦序列。对于每对父代和弦序列，会生成两个子代。每个子代都是这对父代经过交叉操作后，再进行可能的变异操作的结果。新种群由所有这些子代组成。

        此方法确保新种群的规模与生成器预定义的种群规模相等。它将父代按对处理，每对父代生成两个子代。
        """
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(
                parents[i], parents[i + 1]
            ), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):# 使用单点交叉将两个父代序列组合成一个新的子代序列。
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

    def _mutate(self, chord_sequence): # 根据突变率对序列中的和弦进行突变。
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence


class FitnessEvaluator: # 	评估和弦序列适配度：从旋律匹配、和弦多样性、和声流畅性、功能和声 4 维度加权计算分数

    def __init__(
        self, melody_data, chord_mappings, weights, preferred_transitions
    ): # 使用旋律、和弦、权重和偏好进行来初始化适应度评估器。

        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_chord_sequence_with_highest_fitness(self, chord_sequences): #  返回适应度分数最高的和弦序列。
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence): # 评估给定给定和弦序列的适应度

        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):# 计算和弦序列与旋律之间的一致性。
                                                    # 此函数评估序列中的每个和弦与旋律的对应应片段的匹配程度。匹配度通过检查旋律中的音符是否存在于同时演奏的和弦中来衡量，对旋律音符与和弦匹配良好的序列给予奖励。
        score, melody_index = 0, 0
        for chord in chord_sequence:
            bar_duration = 0
            while bar_duration < 4 and melody_index < len(
                self.melody_data.notes
            ):
                pitch, duration = self.melody_data.notes[melody_index]
                if pitch[0] in self.chord_mappings[chord]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.melody_data.duration

    def _chord_variety(self, chord_sequence): # 评估序列中使用的和弦多样性。此函数根据序列中出现的独特和弦数量与可用和弦总数的比例计算分数。
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence): #  通过检查连续和弦之间的进行来评估和弦序列的和声流畅性。
                                            # 此函数根据和弦进行与预定义的偏好进行的匹配频率为序列打分。流畅且在音乐上悦耳的进行会获得更高的分数。
        score = 0
        for i in range(len(chord_sequence) - 1):
            next_chord = chord_sequence[i + 1]
            if next_chord in self.preferred_transitions[chord_sequence[i]]:
                score += 1
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):# 基于功能和声原理评估和弦序列。此函数会检查关键和声功能是否存在，例
        score = 0
        if chord_sequence[0] in ["C", "Am"]:
            score += 1
        if chord_sequence[-1] in ["C"]:
            score += 1
        if "F" in chord_sequence and "G" in chord_sequence:
            score += 1
        return score / 3


def create_score(melody, chord_sequence, chord_mappings): # 用给定的旋律和和弦序列创建一个 music21 乐谱。

    # Create a Score object
    score = music21.stream.Score()

    # Create the melody part and add notes to it 创作旋律部分并为其添加音符
    melody_part = music21.stream.Part()
    for note_name, duration in melody:
        melody_note = music21.note.Note(note_name, quarterLength=duration)
        melody_part.append(melody_note)

    # Create the chord part and add chords to it 创作和弦部分并为其添加和弦
    chord_part = music21.stream.Part()
    current_duration = 0  # Track the duration for chord placement 记录和弦放置的时长

    for chord_name in chord_sequence:
        # Translate chord names to note lists 将和弦名称翻译成音符列表
        chord_notes_list = chord_mappings.get(chord_name, [])
        # Create a music21 chord
        chord_notes = music21.chord.Chord(
            chord_notes_list, quarterLength=4
        )  # Assuming 4/4 time signature 假设是4/4拍号
        chord_notes.offset = current_duration
        chord_part.append(chord_notes)
        current_duration += 4  # Increase by 4 beats 增加4拍

    # Append parts to the score 给乐谱添加部分内容
    score.append(melody_part)
    score.append(chord_part)

    return score


def main():

    twinkle_twinkle_melody = [
        ("C5", 1),
        ("C5", 1),
        ("G5", 1),
        ("G5", 1),
        ("A5", 1),
        ("A5", 1),
        ("G5", 2),  # Twinkle, twinkle, little star,
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 1),
        ("D5", 1),
        ("C5", 2),  # How I wonder what you are! 
        ("G5", 1),
        ("G5", 1),
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 2),  # Up above the world so high,
        ("G5", 1),
        ("G5", 1),
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 2),  # Like a diamond in the sky.
        ("C5", 1),
        ("C5", 1),
        ("G5", 1),
        ("G5", 1),
        ("A5", 1),
        ("A5", 1),
        ("G5", 2),  # Twinkle, twinkle, little star,
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 1),
        ("D5", 1),
        ("C5", 2)  # How I wonder what you are!
    ]
    weights = {
        "chord_melody_congruence": 0.4,
        "chord_variety": 0.1,
        "harmonic_flow": 0.3,
        "functional_harmony": 0.2
    }
    chord_mappings = {
        "C": ["C", "E", "G"],
        "Dm": ["D", "F", "A"],
        "Em": ["E", "G", "B"],
        "F": ["F", "A", "C"],
        "G": ["G", "B", "D"],
        "Am": ["A", "C", "E"],
        "Bdim": ["B", "D", "F"]
    }
    preferred_transitions = {
        "C": ["G", "Am", "F"],
        "Dm": ["G", "Am"],
        "Em": ["Am", "F", "C"],
        "F": ["C", "G"],
        "G": ["Am", "C"],
        "Am": ["Dm", "Em", "F", "C"],
        "Bdim": ["F", "Am"]
    }

    # Instantiate objects for generating harmonization 实例化用于生成协调的对象
    melody_data = MelodyData(twinkle_twinkle_melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    # Generate chords with genetic algorithm 用遗传算法生成和弦
    generated_chords = harmonizer.generate(generations=1000)

    # Render to music21 score and show it 渲染为music21乐谱并显示它
    music21_score = create_score(
        twinkle_twinkle_melody, generated_chords, chord_mappings
    )
    # 自定义 MIDI 保存路径（替换为你想保存的路径）
    midi_save_path = "/Users/liukun/Desktop/遗传算法小星星配和声/小星星_伴奏.mid"
    # 保存 MIDI 文件
    music21_score.write('midi', fp=midi_save_path)
    # 打印提示，确认生成成功
    print(f"MIDI 文件已生成！保存路径：{midi_save_path}")

if __name__ == "__main__":
    main()
