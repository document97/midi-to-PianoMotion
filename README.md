# midi-to-PianoMotion

[English](README.en.md)

这是 **BSide: Olivia Lin** 钢琴演奏功能的非官方开源复刻算法，与米哈游无关。

算法可实现输入 MIDI 文件，让三维角色在 Blender 里把它在钢琴上弹出来：

1. **规划**：决定每个音由哪只手、哪根手指弹，每根手指什么时候抬起，以及哪些音因为手够不着只能省略。左右手由规划器自己分配，不需要分好手的谱子。
2. **动作**：生成手臂、手腕、手指、上身、头部与眼部微扫视、右脚踩延音踏板和每个琴键的动画，写进带试听音频的 `.blend`。

不用训练数据，也没有神经网络，我做的是较轻量化的纯粹算法。规划是束搜索，动作是用角色自己的骨架和皮肤做反向运动学求解。全部在 CPU 上多进程运行，核心求解全部用 numba 编译成机器码（无 GPU 或纯 numpy 回退）。一首 160 秒的曲子（Daylight），在 16 个进程上约 8 分钟解完。

> **演示视频：** https://www.bilibili.com/video/BV1QEnc6rECK/

## 初衷

在 BSide: Olivia Lin 里，你可以把一首曲子的 MIDI 交给林离，她会坐在窗边的钢琴前，把它弹给你听。

2026 年 8 月末，官方服务停了。停服前我交给她的最后一首是 Seredris 的 Daylight。她弹了三遍：夜里一遍，白天一遍，黄昏一遍。那三段视频至今还躺在我电脑的缓存里，可我再也没法交给她新的曲子了。

所以就有了这个项目。我想尽可能把那个功能复刻出来。

## 效果和局限

- **够不着的音会省略，不硬够。** 一只手能同时按住哪些键，是在角色自己的手上量出来的；键盘最两端的键，上身倾斜、在琴凳上挪动之后仍够不到的，也会省略。省略时优先保住最高和最低的声部，以及更响、更长的音。Daylight 的 1049 个音省略了 15 个；音很密、跨度很大的编曲会省略得多（一首 3135 个音的极端测试曲省略了 18%）。
- **不支持双手交叉的弹法。**
- **只做演奏动作。** 上身会前倾、侧倾、在琴凳上挪动，肩膀会前送，曲子前后双手自动贴合角色大腿静置，头部与眼球会随演奏音区做克制的微扫视（模型含眨眼形态键时自动眨眼）；但不做随音乐起伏的表情、衣服和头发物理，也不渲染视频。
- **支持多角色适配。** 参数最初在开发角色上调校，后续已补齐多角色通用适配开关（见下方环境变量），主要在 Windows 上验证过。

## 仓库里有什么

只有算法。开发时用的角色、钢琴和场景模型受限于协议无法公开。算法适配模型对使用者来说有一定的难度，建议使用 AI 工具辅助您。

| 文件夹 | 内容 |
|---|---|
| `pianist/planner/` | 指法规划：按同时起音的音组做束搜索，决定用哪只手、哪根手指、何时抬键、省略哪些音。附独立校验和听感保真度指标 |
| `pianist/global_planning/` | 规划时用来筛选的手型规则（够得着的跨度、固定手掌时的转动范围、手腕移动） |
| `pianist/midi_input/` | MIDI 时钟、延音踏板（CC64；没有踏板的 MIDI 会按低音的变化补上）、持音要求 |
| `pianist/performer/` | 动作：每次触键的姿势、连贯的手型、手腕路径、手指击键、逐帧手指 IK、手臂、上身、踏板。`pipeline.py` 是求解入口，`blender_main.py` 把结果写进 Blender |
| `pianist/lib/` | Blender 端的手臂、拇指和身体驱动，琴键几何，曲线写入 |
| `pianist/checks/` | `blend_audit.py`：核对写出的 `.blend`，每个琴键、踏板的动画是否和求解结果一致，有没有残留的旧动画，播放设置是否正确（在 Blender 里运行） |
| `pianist/inputs/` | `build_piece.py`（从 MIDI 生成规划输入）、键盘布局和手指可达范围表 |

## 依赖

- **Python**，装有 `numpy`、`scipy`、`mido`、`numba`。**`numba` 是硬性必需依赖**（已移除 GPU 与纯 numpy 慢速回退）：核心求解器全部编译成机器码运行，未安装 `numba` 时程序会在启动阶段直接报错提示 `pip install numba`。第一次运行会先编译，结果缓存在磁盘上，改代码后才重新编译。
- **Blender**。它自带的 Python 只负责读写场景（无需在 Blender 内装 `numba`），求解在上面那个 Python 里进行。
- **试听音频**（输出里的 `preview.wav`；必需，缺少时求解会在最后生成音频这一步报错）：[FluidSynth](https://github.com/FluidSynth/fluidsynth/releases)（Windows 版解压到 `tools/fluidsynth/`）和 Dore Mark 的 Fazioli F308 SF2 音源（法齐奥里三角钢琴，soundfonts4u 制作；原站已关闭，可从 [Hugging Face 镜像](https://huggingface.co/datasets/projectlosangeles/soundfonts4u) 下载 `Dore Mark's Fazioli F308-v2.6.sf2`，放到 `tools/fazioli/`）。放在别处时用环境变量 `PIANOMOTION_FLUIDSYNTH`、`PIANOMOTION_SOUNDFONT` 指定。输出是 WAV，只做一次统一的音量调整。
- 开发环境是 Windows 11、Python 3.12、Blender 5.2。报告里的听感指标（各频段比原曲少多少分贝）需要 General MIDI DLS 音色库：Windows 自动读取系统自带 `gm.dls`，macOS 自动读取 `gs_instruments.dls`，Linux 或自定义路径通过环境变量 `PIANOMOTION_SOUNDBANK` 指定。

## 用自己的角色

动作是针对具体模型求解的。开发精力有限，还望使用者见谅。如果使用其他的 MMD 模型，需要准备：

- **Blender 场景**，包含：
  - 一个使用 MMD（日文）骨骼名的骨架（如 `左手首`、`右ひじ`、`左親指０`、`上半身`、`右足ＩＫ`），以及它驱动的网格。场景里只能有一个这样的骨架；
  - 88 个琴键对象，命名为 `key_21` … `key_108`（MIDI 音高）；
  - 琴凳对象 `bench_seat`（凳腿 `bench_leg*`）、钢琴的延音踏板对象（名字写在 `performer/staging.py` 的 `PIANO_DAMPER`）、地板对象 `floor`（没有时地面高度按 0）。
- **`pianist/hand_calibration.json`**：每根手指的骨骼名、关节轴和方向、中性姿态、关节活动范围，以及指腹上的接触区域（`schema: all-digit-calibration`，由 `performer/kinematics.py`、`performer/rig_export.py` 和 `performer/blender_main.py` 读取）。仓库里没有生成它的工具，需要按自己的模型填写。
- **`pianist/inputs/instrument/` 里的两张表**：`keyboard.json` 是开发场景那台钢琴的琴键位置；`pad_reach_conservative.json` 是每根手指的指腹离手腕最远能到多远，规划时用来判断够不够得着，仓库里的是开发模型放大后的数值。换钢琴、换模型都要按自己的场景重新量，仓库里同样没有生成工具。
- **`performer/rig_export.py` 开头的三个量**：座位 `DEFAULT_SEATING`、整体放大 `CHARACTER_SCALE`、手的放大 `HAND_SCALE`。它们是按开发模型调的，见“开发用的角色模型”。

然后运行：

```bash
cd pianist
# 导出骨架、皮肤点和键盘，供求解使用
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# 从这个骨架采样规划器用的手指范围
python -m performer.hand_ranges
# 从 MIDI 生成一首曲子的规划输入
python inputs/build_piece.py path/to/your_piece.mid your_piece
# 规划 + 动作 + 写入 Blender
blender -b your_scene.blend --python-exit-code 1 --python performer/blender_main.py -- \
    --inputs inputs/your_piece --output ../output/your_piece
# 核对写出的 .blend（可选）
blender -b ../output/your_piece/performer.blend --python checks/blend_audit.py -- ../output/your_piece
```

输出文件夹里有：

| 文件 | 内容 |
|---|---|
| `performer.blend` | 写好动画的场景，带试听音频，按音频同步播放 |
| `preview.wav` | 实际弹出来的声音：省略和提前抬起的音都按规划结果 |
| `plan.json` | 每个音的手、手指、抬键时间，以及省略了哪些音 |
| `report.json` | 省略和缩短的音数、各音区的缺失、动作统计、各阶段耗时 |
| `motion.npz` | 全部动画曲线（每秒 60 个采样） |

常用选项和环境变量：

| 写法 | 作用 |
|---|---|
| `--end 30` | 只解前 30 秒，试跑用 |
| `--skip-solve` | 不重新求解，用输出文件夹里已有的结果重写 `.blend` |
| `--workers N` | 求解的进程数 |
| `inputs/<曲名>/motion.json` | 这首曲子的抬手高度倍数，如 `{"lift": 0.8}` |
| `PIANOMOTION_PYTHON` | 求解用的 Python（必须装有 `numba`）。不设时用 `PATH` 上找到的 `python` |
| `PIANOMOTION_RIG` | 用另一份导出的骨架，默认 `performer/rig_cache/rig.json` |
| `PIANOMOTION_HAND_CALIBRATION` | 指定自定义角色的手部标定 JSON，默认 `pianist/hand_calibration.json` |
| `PIANOMOTION_ENVELOPES` / `PIANOMOTION_RADII` | 指定自定义角色的手指活动包络表 / 保守触键半径表 |
| `PIANOMOTION_CHARACTER_SCALE` / `PIANOMOTION_HAND_SCALE` | 覆盖角色整体缩放 / 手部缩放（导出 rig 与烘焙 `.blend` 时需保持一致） |
| `PIANOMOTION_SOUNDBANK` | 指定听感保真度评估使用的 `.dls` 音色库路径（Win/macOS 默认自动查找） |

进程数默认取 CPU 大核的全部线程：Intel 大小核（P 核 / E 核）和 Arm big.LITTLE 只用性能核，所有核都相同时就用全部线程。进程会绑定到这些核上（Windows、Linux 读系统的核类型信息；macOS 只取性能核数量，不绑定）。

第一次运行时会为这个骨架计算手指配对表（哪两根手指能同时按住哪两个键，`performer/reach_table.py`），16 进程约 20 秒，缓存在 `performer/rig_cache/`，骨架或求解代码变了才重算。

## 开发用的角色模型

为了方便，开发时直接用了《原神》官方发布的 MMD 模型【芙宁娜】，算法的参数都是在她身上调校和验证的。模型受协议限制，不随仓库提供，需要自己获取。

对算法做些适配后，也可以换成自定义的人物模型，但身材会直接影响能弹什么。开发中遇到过这几个问题：

- **人太小**：原模型赤脚约 1.52 m，黑键八度差 3.4 mm 够不到。整个角色放大到 1.05 倍（约 1.60 m，`CHARACTER_SCALE`）。放大要以琴凳面中心为基点，否则人会整体升高，大腿顶进琴键底板。
- **手偏小**：放大后手腕到中指尖只有 15 cm（真人约 17 cm），白键八度会落在键的边上。两只手再单独放大到 1.12 倍（`HAND_SCALE`）。
- **座位**：坐得太近、太高时，大腿会顶进琴键底板，手肘也弯得太紧。现在的 `DEFAULT_SEATING` 更靠后、更低。写入 Blender 时，琴凳面会降到坐着的身体下面，双脚落到地板上，右脚前掌踩在钢琴的延音踏板上（`performer/staging.py`）。

## 许可证

[AGPL-3.0](LICENSE)
