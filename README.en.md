# midi-to-PianoMotion

[中文](README.md)

An unofficial open-source recreation of the piano performance feature of **BSide: Olivia Lin**. Not affiliated with miHoYo.

Give it a MIDI file and a 3D character plays it on a piano in Blender:

1. **Planning**: decides which hand and which finger plays each note, when each finger lets go of its key, and which notes have to be left out because the hand cannot reach them. The planner splits the notes between the hands itself; the score does not need to be separated into hands. It also enforces measured finger-web limits and runs two-round chord-shape replanning by measured awkwardness.
2. **Motion**: animates the arms, wrists, fingers, the upper body, subtle head/eye gaze tracking, the right foot on the sustain pedal and every key, and writes it all into a `.blend` with preview audio.

No training data and no neural network: this is a fairly lightweight, purely algorithmic approach. Planning is a beam search; motion is inverse kinematics solved on the character's own skeleton and skin. Everything runs on the CPU in parallel processes in fp32, with all core solves compiled to machine code via `numba` (GPU and pure-NumPy fallbacks have been removed). At 4.2 GHz all-core on 16 threads, a ~260-second piece (1,400+ notes) solves in about 11 minutes.

> **Demo video (v1 algorithm / `main` branch):** https://www.bilibili.com/video/BV1QEnc6rECK/
>
> **Branches:** The default branch is now **`fp32-numba`** (current algorithm: 3D trunk-hull collision & elbow swing solver, automatic lap-rest fitting with opening/closing gestures, two-pass wrist re-aim & damped attitude smoothing, subtle head/eye micro-gaze, mandatory Numba compilation, and multi-character env-var overrides). The original v1 algorithm is preserved intact on `main`.

## Why

In BSide: Olivia Lin you could hand Olivia the MIDI of a piece, and she would sit down at the piano by the window and play it for you.

At the end of August 2026 the official service shut down. The last piece I gave her before it closed was Seredris's Daylight. She played it three times: once at night, once by day, once at dusk. Those three videos are still sitting in a cache folder on my computer, but I can never give her a new piece again.

So this project exists. I want to recreate that feature as faithfully as I can.

## What to expect

- **Notes out of reach are left out, not forced.** Which keys one hand can hold together and how wide neighbouring finger webs can open are measured on the character's own hand; keys at the far ends of the keyboard that stay out of reach after the upper body has leaned and slid along the bench are left out too. The top and bottom voices and the louder, longer notes are kept first. Well-proportioned arrangements achieve 0 omitted notes; dense arrangements with extreme spans omit notes by priority (an early 3,135-note stress test omitted 18%).
- **Natural playing posture & transitions.** Hand shape blends smoothly between arched and flat by chord span, with a strike/crawl lift ladder over free windows; full opening lift and closing return gestures automatically fit the character's own thighs at rig-export time (`lap_contact.py`); 3D sliced trunk hulls keep elbows and arms clear of the ribcage with coordinated torso yield (`elbows.py`); the head and eyes follow the active keyboard register with restrained micro-gaze and VOR compensation, plus automatic blinking when a blink shape key is present (`eyes.py`).
- **Crossed hands are not supported.**
- **Playing motion only.** Facial expressions, cloth/hair physics, and video rendering are not handled.
- **Multi-character adaptation supported.** Parameters were initially tuned on the development character and can now be overridden via environment variables without editing source code; primarily tested on Windows.

## What is in the repository

Only the algorithm. The character, piano and scene models used during development cannot be published because of their licenses. Adapting the algorithm to a model takes some effort; using an AI tool to help is recommended.

| Folder / Module | Contents |
|---|---|
| `pianist/planner/` | Fingering: a beam search over simultaneous note groups choosing hand, finger, release time and omissions, with finger-web & awkwardness pricing, independent validation, cross-platform listening fidelity and preview audio synthesis |
| `pianist/global_planning/` | Hand rules used to screen candidates while planning (reachable spans, rotation range with a fixed palm, wrist travel) |
| `pianist/midi_input/` | MIDI clock, sustain pedal (CC64; a MIDI without any gets a pedal that follows its bass notes while preserving existing score pedal changes), hold requirements |
| `pianist/performer/` | Core motion pipeline: `pipeline.py` (solver entry) and `blender_main.py` (Blender bake); includes `pose.py` / `coherence.py` / `motion.py` (touch poses, chord shapes, wrist paths & per-frame finger IK), `fast_settle.py` (compiled Numba solver kernels), `elbows.py` / `elbows_fast.py` / `elbows_render_fast.py` (3D trunk-hull collision & elbow swing), `attitude.py` (two-pass wrist re-aim & damped yaw/roll smoothing), `torso.py` (lean, bench slide & shoulder protraction), `lap_contact.py` (automatic thigh rest fitting), `eyes.py` (head/eye micro-gaze & blinking), `staging.py` (bench fitting, analytic leg placement & damper pedal linkage), `cores.py` / `codekey.py` (big-core affinity & source-digest caching) |
| `pianist/lib/` | Blender-side arm, thumb and body drivers, key geometry and curve writing |
| `pianist/checks/` | `blend_audit.py`: checks a written `.blend`: whether every key and the pedal animation match the solve, whether stale animation is left over, whether the playback settings are right (runs inside Blender) |
| `pianist/inputs/` | `build_piece.py` (turns a MIDI file into planner inputs, storing only the filename with no local paths), keyboard layout and finger reach tables |

## Requirements

- **Python** with `numpy`, `scipy`, `mido` and `numba`. **`numba` is strictly required** (GPU and pure-NumPy fallbacks have been removed): all core solves run compiled to machine code, and without `numba` the solver stops immediately at startup and tells you to run `pip install numba`. The first run compiles and caches the result on disk until the code changes.
- **Blender**. Its own Python only reads and writes the scene (no `numba` needed inside Blender); the solve runs in the Python above.
- **Preview audio** (`preview.wav` in the output folder; required: without it the solve stops at its last step, the audio): [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) (the Windows build unpacked into `tools/fluidsynth/`) and Dore Mark's Fazioli F308 SF2 (a Fazioli grand, converted by soundfonts4u; that site is gone, the [Hugging Face mirror](https://huggingface.co/datasets/projectlosangeles/soundfonts4u) has `Dore Mark's Fazioli F308-v2.6.sf2`, which goes into `tools/fazioli/`). If they are somewhere else, set `PIANOMOTION_FLUIDSYNTH` and `PIANOMOTION_SOUNDFONT`. The output is WAV with one uniform gain and nothing else.
- Developed on Windows 11, Python 3.12 and Blender 5.2. The listening measure in the report (how many dB each band lacks against the source) uses a General MIDI DLS bank: auto-detected on Windows (`gm.dls`) and macOS (`gs_instruments.dls`), or set via `PIANOMOTION_SOUNDBANK` on Linux / custom setups.

## Using your own character

The motion is solved for a specific model. Development time was limited, so please bear with the rough edges. To use another MMD model you need:

- **A Blender scene** with:
  - a seated-bind armature using MMD (Japanese) bone names (such as `左手首`, `右ひじ`, `左親指０`, `上半身`, `右足`, `右ひざ`, `右足首`, plus optional `頭`, `左目`, `右目`) and the mesh it deforms. The scene may hold only one such armature (legs are driven by an analytic two-bone FK solve and no longer depend on `足ＩＫ`; export/bake also automatically strips unused under-skirt root/foot-IK bones and cleans leftover MMD Empties and armature data-block names);
  - 88 key objects named `key_21` … `key_108` (MIDI pitch);
  - a bench object `bench_seat` (legs `bench_leg*`), the piano's sustain pedal object (named in `PIANO_DAMPER` in `performer/staging.py`) and a floor object `floor` (without one the floor is at height 0).
- **Hand calibration JSON** (defaults to `pianist/hand_calibration.json`, or point `PIANOMOTION_HAND_CALIBRATION` at a custom file): for every finger, the bone names, joint axes and directions, neutral pose, joint ranges and contact patch (`schema: all-digit-calibration`). There is no built-in generator in the repository; build or fill it in for your own model.
- **Finger envelopes & reach tables** (defaults to `pianist/inputs/instrument/`, overridable via `PIANOMOTION_ENVELOPES` / `PIANOMOTION_RADII`): `keyboard.json` holds the development piano's key positions; `pad_reach_conservative.json` holds conservative finger-pad reach radii from the wrist; ROM envelopes are sampled directly from your exported rig via `python -m performer.hand_ranges`.
- **Seat & scaling**: defaults live at the top of `performer/rig_export.py` (`DEFAULT_SEATING`, `CHARACTER_SCALE`, `HAND_SCALE`), and can be overridden for custom characters via `PIANOMOTION_SEAT`, `PIANOMOTION_CHARACTER_SCALE`, and `PIANOMOTION_HAND_SCALE` without editing source code (keep the same env vars across `rig_export.py` and `blender_main.py`).

Then run:

```bash
cd pianist
# export skeleton, skin points, keyboard, fingerprinted trunk outline (<rig>.trunk.json), lap pose & eyes
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# sample the finger ROM envelopes the planner uses from that skeleton
python -m performer.hand_ranges
# make the planner inputs of a piece from MIDI
python inputs/build_piece.py path/to/your_piece.mid your_piece
# planning + motion + writing into Blender
blender -b your_scene.blend --python-exit-code 1 --python performer/blender_main.py -- \
    --inputs inputs/your_piece --output ../output/your_piece
# check the written .blend (optional)
blender -b ../output/your_piece/performer.blend --python checks/blend_audit.py -- ../output/your_piece
```

The output folder holds:

| File | Contents |
|---|---|
| `performer.blend` | the animated scene with the preview audio, played in sync with the audio |
| `preview.wav` | what is actually played: omitted notes and keys let go early as planned |
| `plan.json` | the hand, the finger and the release time of every note, and which notes are omitted |
| `report.json` | omitted and shortened notes, what each register lacks, motion/elbow/eye statistics, seconds per stage |
| `motion.npz` | every animation curve (60 samples per second) |

Options and environment variables:

| Setting | Effect |
|---|---|
| `--end 30` | solve the first 30 seconds only, for a trial run |
| `--skip-solve` | no new solve: write the `.blend` again from the results already in the output folder |
| `--workers N` | number of solver processes |
| `inputs/<piece>/motion.json` | how high the hands lift in this piece, as a factor, e.g. `{"lift": 0.8}` |
| `PIANOMOTION_PYTHON` | the Python that runs the solve (must have `numba` installed). Unset: the `python` found on `PATH` |
| `PIANOMOTION_RIG` | another exported skeleton instead of `performer/rig_cache/rig.json` |
| `PIANOMOTION_HAND_CALIBRATION` | custom character hand calibration JSON (default `pianist/hand_calibration.json`) |
| `PIANOMOTION_ENVELOPES` / `PIANOMOTION_ENVELOPES_OUT` | custom finger ROM envelopes input path / `hand_ranges` output path |
| `PIANOMOTION_RADII` | custom conservative pad reach radii table |
| `PIANOMOTION_CHARACTER_SCALE` / `PIANOMOTION_HAND_SCALE` | override character overall scale / hand scale (must match between `rig_export.py` and `blender_main.py`) |
| `PIANOMOTION_SEAT` | custom seat offset `x,y,z` in metres (defaults to `DEFAULT_SEATING`) |
| `PIANOMOTION_SOUNDBANK` | path to a General MIDI `.dls` bank for the fidelity metric (auto-detected on Windows/macOS) |

By default the process count is every thread of the big cores of the CPU: on Intel hybrid (P-cores / E-cores) and Arm big.LITTLE only the performance cores, on a CPU with one kind of core every thread. The processes are pinned to those cores (Windows and Linux read the core types from the system; macOS only takes the number of performance cores, without pinning).

The first run computes the finger pair table for the skeleton (which two fingers can hold which two keys together, `performer/reach_table.py`): about 20–30 seconds on 16 processes, cached beside the rig or in `performer/rig_cache/` until the skeleton or solver code digest changes.

## The development character

For convenience, development used the official *Genshin Impact* MMD model of **Furina**, and the initial algorithm parameters were tuned and verified on her. The model's license does not allow shipping it with the repository; you need to get it yourself.

With adaptation the algorithm also works with other custom MMD characters (for example, a ~1.65 m character with standard proportions only needs scale env vars near `1.0`, with no code edits), though build and hand length still decide what is playable. These problems came up on the default development model:

- **Too small**: the model is about 1.52 m barefoot and missed black-key octaves by 3.4 mm. The whole character is scaled to 1.05 (about 1.60 m, `CHARACTER_SCALE`). The scale has to be taken about the middle of the bench top; otherwise the whole body rises and the thighs push into the keybed.
- **Small hands**: after that scale the hand still measures only 15 cm from the wrist to the tip of the middle finger (a real hand about 17 cm), and white-key octaves landed on the edges of the keys. Both hands are scaled by a further 1.12 (`HAND_SCALE`).
- **The seat**: seated too close and too high, the thighs push into the keybed and the elbows bend too tightly. The present `DEFAULT_SEATING` is further back and lower. When the result is written into Blender, the bench top is lowered to the seated body, the feet are put on the floor and the ball of the right foot on the sustain pedal of the piano (`performer/staging.py`).

## License

[AGPL-3.0](LICENSE)
