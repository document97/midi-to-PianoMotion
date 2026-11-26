# midi-to-PianoMotion

[English](README.en.md)

这是 **BSide: Olivia Lin** 钢琴演奏功能的非官方开源复刻算法，与米哈游无关。

算法可实现输入 MIDI 文件，让人物在 Blender 里把它在钢琴上弹出来：

1. **规划**：决定每个音由哪只手、哪根手指弹，每根手指什么时候抬起，以及哪些音因为手够不着只能省略。支持单轨与分轨 MIDI：若乐谱已分好左右手轨道（如轨道名为 `Right hand` / `Left hand`，或在导入时通过 `--right` / `--left` 指定），规划器会默认沿用谱面分轨；未分轨（或单轨）的音符则由算法自动分配左右手；同时按角色实测的指蹼张角与和弦手型别扭度做两轮重规划。
2. **动作**：生成手臂、手腕、手指、上身、头部与眼部微扫视、右脚踩延音踏板和每个琴键的动画，写进带试听音频的 `.blend`。

我做的是较轻量化的纯 Python 源码算法（零神经网络、零训练数据）。规划是束搜索，动作是用角色自己的骨架和皮肤做反向运动学求解。全部在 CPU 上多进程运行，核心求解全部采用 fp32 并用 `numba` 编译成机器码（已移除 GPU 与纯 numpy 慢速回退）。全核 4.2 GHz 16 线程下，一首 260 秒左右的曲子（1,400+ 音）约 11 分钟解完。

> **演示视频（第 2 代算法 / `fp32-numba` 分支）：** https://www.bilibili.com/video/BV1pr1jzEEHk/
>
> **演示视频（第 1 代算法 / `fp64` 分支）：** https://www.bilibili.com/video/BV1QEnc6rECK/
>
> **分支说明：** 当前默认展示分支为 **`fp32-numba`**（最新算法：含躯干三维避让与肘部求解、起手/收手腿面自动贴合、手腕二次重瞄与阻尼平滑、头部/眼部微扫视、全量 Numba 编译与多角色环境变量适配）。原第 1 代算法完整保留在 `fp64` 分支。

## 灵感起源

在 BSide: Olivia Lin 里，你可以把一首曲子的 MIDI 交给林离，她学习一段时间后，会坐在窗边的钢琴前，把它弹给你听。

2026 年 8 月末，官方服务停了。停服前我交给她的最后一首是 Seredris 的 Daylight。晌午、黄昏、夜晚三个场景三段视频。那三段视频至今还躺在我电脑的缓存里，可我再也没法交给她新的曲子了。

所以就有了这个项目。我想尽可能把那个功能复刻出来。

## 效果和局限

- **够不着的音会省略，不强求。** 一只手能同时按住哪些键、相邻手指指蹼能张开多大，都是在角色自己的手上量出来的；键盘最两端的键，上身倾斜、在琴凳上挪动之后仍够不到的，也会省略。省略时优先保住最高和最低的声部，以及更响、更长的音。编曲跨度合理时可达全曲 0 漏音；音极密、跨度极大的编曲会按规则省略（早期 3,135 音的极端测试曲省略了 18%）。
- **自然的演奏姿态与物理防穿模。** 手型随按住的跨度在拱手与展平间连续过渡，并按空窗时长区分击键抬手阶梯；上身会随音区前倾、侧倾并在琴凳上挪动，肩膀会在远距离够键时自然前送；曲子前后具备完整的起手抬离与收手回落动作，静置手位会在导出骨架时自动贴合角色大腿曲面（`lap_contact.py`）；躯干三维凸包碰撞会约束肘部与上臂不穿进肋骨，需要时上身自动协同让位（`elbows.py`）；头部与眼球会随演奏音区做克制的微扫视与前庭补偿，模型含眨眼形态键时自动眨眼（`eyes.py`）。
- **不支持双手交叉的弹法。**
- **只做演奏动作。** 未加入感情系统，角色没法理解曲子，不会有随音乐起伏的面部表情；角色的臂、肘与躯干、手指与琴键/邻指以及休息手与大腿皮肤均做了物理碰撞防穿模，其他部分（例如服饰、长发等）大多需要自行解决物理与穿模问题，也不负责渲染视频。
- **支持多角色适配。** 参数最初在开发角色上调校，现已通过环境变量完整开放骨架、手部标定、可达范围表、整体/手部缩放、座位坐标与场景对象名等覆盖项（无需改动源码，见下方说明）；主要在 Windows 上验证过。

## 仓库里有什么

只有算法。开发时用的角色、钢琴和场景模型涉及版权保护部分无法公开。模型适配算法对使用者来说有一定的难度，建议使用 AI 工具辅助您。

| 文件夹 / 模块 | 内容 |
|---|---|
| `pianist/planner/` | 指法规划：按同时起音的音组做束搜索，决定用哪只手、哪根手指、何时抬键、省略哪些音。含指蹼与别扭度定价、独立校验、听感保真度评估与试听音频合成 |
| `pianist/global_planning/` | 规划时用来筛选的手型规则（够得着的跨度、固定手掌时的转动范围、手腕移动） |
| `pianist/midi_input/` | MIDI 时钟、延音踏板（CC64；没有踏板的 MIDI 会按低音的变化补上，保留原谱踏板变化）、持音要求 |
| `pianist/performer/` | 核心动作管线：`pipeline.py` 为求解入口，`blender_main.py` 写入 Blender；含 `pose.py` / `coherence.py` / `motion.py`（触键、和弦手型、手腕路径与逐帧手指 IK）、`fast_settle.py`（Numba 核心求解内核）、`elbows.py` / `elbows_fast.py` / `elbows_render_fast.py`（躯干切片凸包避让与肘部曲线）、`attitude.py`（手腕二次重瞄与阻尼平滑）、`torso.py`（上身前倾/侧倾/琴凳平移与肩部前送）、`lap_contact.py`（大腿静置手位自动拟合）、`eyes.py`（头眼微扫视与眨眼）、`staging.py`（琴凳落脚与踏板驱动） |
| `pianist/lib/` | Blender 端的手臂、拇指和身体驱动，琴键几何，曲线写入 |
| `pianist/checks/` | `blend_audit.py`：核对写出的 `.blend`，每个琴键、踏板的动画是否和求解结果一致，有没有残留的旧动画，播放设置是否正确（在 Blender 里运行） |
| `pianist/inputs/` | `build_piece.py`（从 MIDI 生成规划输入）、键盘布局和手指可达范围表 |

## 依赖

- **Python**，装有 `numpy`、`scipy`、`mido`、`numba`。**`numba` 是硬性必需依赖**：核心求解器全部编译成机器码运行，未安装 `numba` 时程序会在启动阶段直接报错提示 `pip install numba`。第一次运行会先编译，结果缓存在磁盘上，改代码后才重新编译。
- **Blender**。它自带的 Python 只负责读写场景（无需在 Blender 内装 `numba`），求解在上面那个 Python 里进行。
- **试听音频**（输出里的 `preview.wav`；必需，缺少时求解会在最后生成音频这一步报错）：[FluidSynth](https://github.com/FluidSynth/fluidsynth/releases)（Windows 版解压到 `tools/fluidsynth/`）和 Dore Mark 的 Fazioli F308 SF2 音源（法齐奥里三角钢琴，soundfonts4u 制作；原站已关闭，可从 [Hugging Face 镜像](https://huggingface.co/datasets/projectlosangeles/soundfonts4u) 下载 `Dore Mark's Fazioli F308-v2.6.sf2`，放到 `tools/fazioli/`）。放在别处时用环境变量 `PIANOMOTION_FLUIDSYNTH`、`PIANOMOTION_SOUNDFONT` 指定。输出是 WAV，只做一次统一的音量调整。
- **测试环境**：主要在 Windows 11、Python 3.12、Blender 5.2 / 5.3 上验证。`report.json` 中的频段听感指标会自动调用系统自带的 General MIDI DLS 音色库（Windows `gm.dls` / macOS `gs_instruments.dls`），缺少时自动跳过该指标，不影响动作求解与 `preview.wav` 生成。

## 适配自己的角色须知

动作是针对具体模型求解的。MMD 模型作为一种常用的格式我在开发时进行了部分名称的映射适配，也支持通过配置和环境变量适配其他角色。使用前需要准备：

- **Blender 场景**，包含：
  - 一个使用 MMD（日文）骨骼名的骨架（如 `左手首`、`右ひじ`、`左親指０`、`上半身`、`右足ＩＫ`），以及它驱动的网格。场景里只能有一个这样的骨架（如果包含头部/眼球骨骼与眨眼形态键，`rig_export.py` 会自动检测用于眼神与眨眼驱动）；
  - 88 个琴键对象，命名为 `key_21` … `key_108`（MIDI 音高）；
  - 琴凳对象 `bench_seat`（凳腿 `bench_leg*`）、钢琴的延音踏板对象（默认名写在 `performer/staging.py` 的 `PIANO_DAMPER`）、地板对象 `floor`（没有时地面高度按 0）。以上对象名均可通过环境变量覆盖，无需改源码。
- **`pianist/hand_calibration.json`**：每根手指的骨骼名、关节轴和方向、中性姿态、关节活动范围，以及指腹上的接触区域（`schema: all-digit-calibration`，由 `performer/kinematics.py`、`performer/rig_export.py` 和 `performer/blender_main.py` 读取，可用 `PIANOMOTION_HAND_CALIBRATION` 指定自定义文件）。仓库里没有生成它的工具，需要按自己的模型填写。
- **`pianist/inputs/instrument/` 里的两张表**：`keyboard.json` 是开发场景钢琴的琴键位置；`pad_reach_conservative.json` 是每根手指指腹离手腕的保守可达半径；包络表可由 `python -m performer.hand_ranges` 直接从导出的 rig 采样生成（可用 `PIANOMOTION_ENVELOPES` / `PIANOMOTION_ENVELOPES_OUT` / `PIANOMOTION_RADII` 指向自定义角色表）。
- **座位与缩放**：默认使用 `performer/rig_export.py` 中的 `DEFAULT_SEATING`、`CHARACTER_SCALE`、`HAND_SCALE`（见“开发时遇到的问题”），换角色时可直接用环境变量 `PIANOMOTION_SEAT`、`PIANOMOTION_CHARACTER_SCALE`、`PIANOMOTION_HAND_SCALE` 覆盖，无需改动源码（导出 rig 与烘焙 `.blend` 时需保持一致）。

然后运行：

```bash
cd pianist
# 1. 导出骨架、皮肤点、键盘，并自动生成带指纹的躯干碰撞轮廓 (<rig>.trunk.json)、腿面静置手位与眼部信息
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# 2. 从这个骨架采样规划器用的手指活动包络表（可用 PIANOMOTION_ENVELOPES_OUT 指定自定义输出路径）
python -m performer.hand_ranges
# 3. 从 MIDI 生成一首曲子的规划输入（多轨乐谱可用 --right/--left 指定左右手轨号）
python inputs/build_piece.py path/to/your_piece.mid your_piece
# 4. 规划 + 动作求解 + 写入 Blender
blender -b your_scene.blend --python-exit-code 1 --python performer/blender_main.py -- \
    --inputs inputs/your_piece --output ../output/your_piece
# 5. 核对写出的 .blend（可选）
blender -b ../output/your_piece/performer.blend --python checks/blend_audit.py -- ../output/your_piece
```

输出文件夹里有：

| 文件 | 内容 |
|---|---|
| `performer.blend` | 写好动画的场景，带试听音频，按音频同步播放 |
| `preview.wav` | 实际弹出来的声音：省略和提前抬起的音都按规划结果 |
| `plan.json` | 每个音的手、手指、抬键时间，以及省略了哪些音 |
| `report.json` | 省略和缩短的音数、各音区的缺失、动作/肘部/眼部统计、各阶段耗时 |
| `motion.npz` | 全部动画曲线（每秒 60 个采样） |
| `diagnostics.npz` | 逐帧指间间隙、重叠、双手间距等物理诊断曲线 |

### 常用命令行选项

| 写法 | 适用脚本 | 作用 |
|---|---|---|
| `--end 30` | `blender_main.py` / `pipeline.py` | 只解前 30 秒，试跑用 |
| `--skip-solve` | `blender_main.py` | 不重新求解，用输出文件夹里已有的 `motion.npz` 重写 `.blend` |
| `--workers N` | `blender_main.py` / `pipeline.py` | 指定求解的进程数（覆盖默认性能核线程数） |
| `--right T --left T` | `inputs/build_piece.py` | 指定 MIDI 中右手 / 左手轨道编号（不传时仅识别名为 `Right hand`/`Left hand` 的轨，其余交由算法分轨） |
| `--trunk-only` | `performer/rig_export.py` | 仅重导躯干防穿模轮廓 `<rig>.trunk.json`（改完服饰/躯干网格时用） |
| `--beam N` | `performer/pipeline.py` | 束搜索宽度，默认 `48` |
| `--refine-rounds N` | `performer/pipeline.py` | 手腕路径细化检查轮数，默认 `2` |
| `--lift-speed V` / `--motion-cut S` | `performer/pipeline.py` | 调整手腕移动规划速度（m/s）与允许提前抬键的最大可听时长（s） |
| `inputs/<曲名>/motion.json` | 曲目配置 | 这首曲子的抬手高度倍数，如 `{"lift": 0.8}` |

### 环境变量（多角色适配与路径覆盖）

| 环境变量 | 作用 |
|---|---|
| `PIANOMOTION_PYTHON` | 求解用的 Python（必须装有 `numba`）。不设时用 `PATH` 上找到的 `python` |
| `PIANOMOTION_FLUIDSYNTH` / `PIANOMOTION_SOUNDFONT` | 指定 `fluidsynth` 可执行文件与 `.sf2` 钢琴音源路径 |
| `PIANOMOTION_RIG` | 用另一份导出的骨架，默认 `performer/rig_cache/rig.json` |
| `PIANOMOTION_HAND_CALIBRATION` | 指定自定义角色的手部标定 JSON，默认 `pianist/hand_calibration.json` |
| `PIANOMOTION_ENVELOPES` / `PIANOMOTION_ENVELOPES_OUT` | 指定规划器读取的包络表路径 / `hand_ranges` 的输出路径 |
| `PIANOMOTION_RADII` | 指定自定义角色的保守触键半径表 |
| `PIANOMOTION_CHARACTER_SCALE` / `PIANOMOTION_HAND_SCALE` | 覆盖角色整体缩放 / 手部缩放（导出 rig 与烘焙 `.blend` 时需保持一致） |
| `PIANOMOTION_SEAT` | 自定义座位偏移 `x,y,z`（米），默认沿用 `DEFAULT_SEATING` |
| `PIANOMOTION_BENCH` / `PIANOMOTION_BENCH_LEGS` | 覆盖琴凳面对象名（默认 `bench_seat`）与凳腿对象名列表（逗号分隔） |
| `PIANOMOTION_DAMPER` / `PIANOMOTION_FLOOR` | 覆盖钢琴延音踏板对象名与地板对象名（默认 `floor`） |
| `PIANOMOTION_BODY_STILL` | 控制演奏时上身倾侧与挪凳幅度（`0`～`1`，默认 `1` 偏克制，`0` 恢复全幅摆动） |
| `PIANOMOTION_LAP_CONTACT` / `PIANOMOTION_LAP_SLIDE_FOLLOW` | 开关曲首曲尾自动贴腿拟合（默认 `1`）及静置手跟随挪凳比例（默认 `0.91`） |

进程数默认取 CPU 大核的全部线程：Intel 大小核（P 核 / E 核）和 Arm big.LITTLE 只用性能核，所有核都相同时就用全部线程。进程会绑定到这些核上（Windows、Linux 读系统的核类型信息；macOS 只取性能核数量，不绑定）。

第一次运行时会为这个骨架计算手指配对表（哪两根手指能同时按住哪两个键，`performer/reach_table.py`），16 进程约 20–30 秒，缓存在 rig 旁或 `performer/rig_cache/`，骨架或求解代码指纹变化时自动重算。

## 开发时遇到的问题

- **人太小**：开发用模型赤脚约 1.52 m，黑键八度差 3.4 mm 够不到。整个角色放大到 1.05 倍（约 1.60 m，`CHARACTER_SCALE`）。放大要以琴凳面中心为基点，否则人会整体升高，大腿顶进琴键底板。
- **手偏小**：放大后手腕到中指尖只有 15 cm（真人约 17 cm），白键八度会落在键的边上。两只手再单独放大到 1.12 倍（`HAND_SCALE`）。
- **手和人的缩放是独立的**：两个参数供调整直到合适观感（1.65 m 左右比例正常的模型只需通过环境变量将缩放设在 `1.0` 附近即可）。
- **手跨度建议至少八度，留些余量为宜**：真人手掌大概可跨 10 度，不少和弦也是跨八度弹的，将手适当放大可能会降低漏音的数量。
- **座位**：坐得太近、太高时，大腿会顶进琴键底板，手肘也弯得太紧。现在的 `DEFAULT_SEATING` 更靠后、更低。写入 Blender 时，琴凳面会降到坐着的身体下面，双脚落到地板上，右脚前掌踩在钢琴的延音踏板上（`performer/staging.py`）。

## 许可证

[AGPL-3.0](LICENSE)
