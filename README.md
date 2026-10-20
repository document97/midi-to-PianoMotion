# midi-to-PianoMotion

[English](README.en.md)

这是 **BSide: Olivia Lin** 钢琴演奏功能的非官方开源复刻算法，与米哈游无关。

算法可实现输入 MIDI 文件，让三维角色在 Blender 里把它在钢琴上弹出来：

1. **规划**：决定每个音由哪只手、哪根手指弹，每根手指什么时候抬起，以及哪些音因为手够不着只能省略。左右手由规划器自己分配，不需要分好手的谱子。
2. **动作**：生成手臂、手腕、手指、上身倾斜、右脚踩延音踏板和每个琴键的动画，写进带预览音频的 `.blend`。

不用训练数据，也没有神经网络，我做的是较轻量化的纯粹算法。规划是束搜索，动作是用角色自己的骨架和皮肤做反向运动学求解。全程在 CPU 上多进程运行（8 或 16 个进程）。一首 160 秒的曲子，规划十几秒，动作在 16 核上大约 13 分钟解完。

> **演示视频：** [bilibili：林离弹 Daylight（Seredris）](https://www.bilibili.com/video/BV1QEnc6rECK/)

## 初衷

在 BSide: Olivia Lin 里，你可以把一首曲子的 MIDI 交给林离，她会坐在窗边的钢琴前，把它弹给你听。

2026 年 8 月末，官方服务停了。停服前我交给她的最后一首是 Seredris 的 Daylight。她弹了三遍：夜里一遍，白天一遍，黄昏一遍。那三段视频至今还躺在我电脑的缓存里，可我再也没法交给她新的曲子了。

所以就有了这个项目。我想尽可能把那个功能复刻出来。

## 仓库里有什么

只有算法。开发时用的角色、钢琴和场景模型受限于协议无法公开。算法适配模型对使用者来说有一定的难度，建议使用 AI 工具辅助您。

| 文件夹 | 内容 |
|---|---|
| `pianist/planner/` | 指法规划：按同时起音的音组做束搜索，决定用哪只手、哪根手指、何时抬键、省略哪些音。分段并行，拼接结果和串行一致。附独立校验和听感保真度指标 |
| `pianist/global_planning/` | 规划时用来筛选的手型规则（够得着的跨度、固定手掌时的转动范围、手腕移动） |
| `pianist/midi_input/` | MIDI 时钟、延音踏板（CC64）、持音要求 |
| `pianist/performer/` | 动作：每次触键的关键姿势、连贯的手型、手腕路径、手指击键、逐帧手指 IK、手臂、上身、踏板。`pipeline.py` 是入口，`blender_main.py` 把结果写进 Blender |
| `pianist/lib/` | 辅助模块：Blender 端的手臂和拇指驱动、琴键几何、曲线写入，以及预览音频的合成 |
| `pianist/checks/` | `blend_audit.py`：核对写出的 `.blend`，每个琴键、踏板的动画是否和求解结果一致，有没有残留的旧动画，播放设置是否正确（在 Blender 里运行） |
| `pianist/inputs/` | `build_piece.py`（从 MIDI 生成规划输入）、键盘布局和手指可达范围表 |

## 依赖

- Python（需要 `numpy`、`scipy`、`mido`）和 Blender。
- 开发环境是 Windows 11、Python 3.12、Blender 5.2，其他环境没有测试过。

## 开发用的角色模型

为了方便，开发时直接用了《原神》官方发布的 MMD 模型【芙宁娜】，算法的参数都是在她身上调校和验证的。模型受协议限制，不随仓库提供，需要自己获取。

对算法做些适配后，也可以换成自定义的人物模型，但要考虑人物身高。开发中因此遇到过两个问题：

- **手太小**：黑键八度够不到。后来决定把整个角色放大一些（使得赤脚身高约 1.60 m）。放大要以琴凳面中心为基点，否则人会整体升高，大腿顶进琴键底板。
- **腿太短**：坐在琴凳上时膝盖和凳面一样高，脚够不到地板，也踩不到踏板。所以加了一个踏板延长器（`performer/staging.py`），右脚踩它来带动钢琴的延音踏板。

## 用自己的角色

动作是针对具体模型求解的。开发精力有限，还望使用者见谅。如果使用其他的 MMD 模型，需要准备：

- **Blender 场景**，包含：
  - 一个使用 MMD（日文）骨骼名的骨架（如 `左手首`、`右ひじ`、`左親指０`），以及它驱动的网格；
  - 88 个琴键对象，命名为 `key_21` … `key_108`（MIDI 音高）；
  - 琴凳对象 `bench_seat`（凳腿 `bench_leg*`），以及钢琴的延音踏板（见 `performer/staging.py` 的 `PIANO_DAMPER`）。
- **`pianist/hand_calibration.json`**：每根手指的骨骼名、关节轴和方向、中性姿态、关节活动范围，以及指腹上的接触区域（`schema: all-digit-calibration`，由 `performer/kinematics.py`、`performer/rig_export.py` 和 `performer/blender_main.py` 读取）。仓库里没有生成它的工具，需要按自己的模型填写。
- **`pianist/inputs/instrument/pad_reach_conservative.json`**：每根手指的指腹离手腕最远能到多远，规划时用来判断够不够得着。仓库里的是开发模型放大 1.05 倍后的数值，换模型需要按自己的骨架重新测量，仓库里同样没有生成工具。

然后运行：

```bash
cd pianist
# 给 numpy 求解器导出骨架、皮肤点和键盘
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# 从这个骨架采样规划器用的手指范围
python -m performer.hand_ranges
# 从 MIDI 生成一首曲子的规划输入（文件里有分手的音轨时，用 --right/--left 指明）
python inputs/build_piece.py path/to/your_piece.mid your_piece --right 1 --left 2
# 规划 + 动作 + 写入 Blender（输出文件夹里的 performer.blend）
blender -b your_scene.blend --python-exit-code 1 --python performer/blender_main.py -- \
    --inputs inputs/your_piece --output ../output/your_piece --workers 16
```

第一次运行时会为这个骨架计算手指配对表（`performer/reach_table.py`，16 线程约 10 分钟），缓存在 `performer/rig_cache/`。`blender_main.py` 会用 `PATH` 上找到的 Python 求解，也可以用环境变量 `PIANOMOTION_PYTHON` 指定。

预览音频（`planner/audio.py`）用的是 Windows 自带的 General MIDI 音色库（`gm.dls`）。

## 许可证

[AGPL-3.0](LICENSE)
