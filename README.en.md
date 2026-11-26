# midi-to-PianoMotion

[中文](README.md)

An unofficial open-source recreation of the piano performance feature of **BSide: Olivia Lin**. Not affiliated with miHoYo.

Give it a MIDI file and a 3D character plays it on a piano in Blender:

1. **Planning**: decides which hand and which finger plays each note, when each finger lets go of its key, and which notes have to be left out because the hand cannot reach them. Both single-track and hand-split MIDI files are supported: if the score already separates hands (tracks named `Right hand` / `Left hand`, or specified via `--right` / `--left` when building inputs), the planner preserves that hand split by default; unassigned / single-track notes are split between the hands automatically by the planner. It also enforces measured finger-web limits and runs two-round chord-shape replanning by measured awkwardness.
2. **Motion**: animates the arms, wrists, fingers, the upper body, subtle head/eye gaze tracking, the right foot on the sustain pedal and every key, and writes it all into a `.blend` with preview audio.

This is a fairly lightweight pure-Python algorithmic solver (no training data, no neural network). Planning is a beam search; motion is inverse kinematics solved on the character's own skeleton and skin. Everything runs on the CPU in parallel processes in fp32, with all core solves compiled to machine code via `numba` (GPU and pure-NumPy fallbacks have been removed). At 4.2 GHz all-core on 16 threads, a ~260-second piece (1,400+ notes) solves in about 11 minutes.

> **Demo video (v2 algorithm / `fp32-numba` branch):** https://www.bilibili.com/video/BV1pr1jzEEHk/
>
> **Demo video (v1 algorithm / `fp64` branch):** https://www.bilibili.com/video/BV1QEnc6rECK/
>
> **Branches:** The default branch is now **`fp32-numba`** (current algorithm: 3D trunk-hull collision & elbow swing solver, automatic lap-rest fitting with opening/closing gestures, two-pass wrist re-aim & damped attitude smoothing, subtle head/eye micro-gaze, mandatory Numba compilation, and multi-character env-var overrides). The original v1 algorithm is preserved intact on `fp64`.

## Inspiration

In BSide: Olivia Lin you could hand Olivia the MIDI of a piece, and after studying it for a while she would sit down at the piano by the window and play it for you.

At the end of August 2026 the official service shut down. The last piece I gave her before it closed was Seredris's Daylight — three scenes across noon, dusk, and night. Those three videos are still sitting in a cache folder on my computer, but I can never give her a new piece again.

So this project exists. I want to recreate that feature as faithfully as I can.

## What to expect

- **Notes out of reach are left out, not forced.** Which keys one hand can hold together and how wide neighbouring finger webs can open are measured on the character's own hand; keys at the far ends of the keyboard that stay out of reach after the upper body has leaned and slid along the bench are left out too. The top and bottom voices and the louder, longer notes are kept first. Well-proportioned arrangements achieve 0 omitted notes; dense arrangements with extreme spans omit notes by priority (an early 3,135-note stress test omitted 18%).
- **Natural playing posture & physical collision avoidance.** Hand shape blends smoothly between arched and flat by chord span, with a strike/crawl lift ladder over free windows; the upper body leans forward/sideways and slides along the bench while shoulders protract naturally for far reaches; full opening lift and closing return gestures automatically fit the character's own thighs at rig-export time (`lap_contact.py`); 3D sliced trunk hulls keep elbows and arms clear of the ribcage with coordinated torso yield (`elbows.py`); the head and eyes follow the active keyboard register with restrained micro-gaze and VOR compensation, plus automatic blinking when a blink shape key is present (`eyes.py`).
- **Crossed hands are not supported.**
- **Playing motion only.** There is no emotion system, so the character does not produce expressive facial reactions to the music. Physical collision checks keep arms/elbows clear of the trunk, fingers clear of neighbouring keys/fingers, and resting hands clear of the thighs; clothing/hair physics and video rendering are not handled.
- **Multi-character adaptation supported.** Parameters were initially tuned on the development character and now expose full environment-variable overrides for custom rigs, hand calibration, reach tables, body/hand scaling, seating, and Blender scene object names (no source edits required; see below). Primarily tested on Windows.

## What is in the repository

Only the algorithm. The character, piano and scene models used during development cannot be published because of copyright and license restrictions. Adapting the algorithm to a custom model takes some effort; using an AI tool to help is recommended.

| Folder / Module | Contents |
|---|---|
| `pianist/planner/` | Fingering: a beam search over groups of notes that start together, choosing the hand, the finger, when to release and which notes to omit. Includes finger-web & awkwardness pricing, independent validation, listening-fidelity measure and preview audio synthesis |
| `pianist/global_planning/` | Hand rules used to screen candidates while planning (reachable spans, rotation range with a fixed palm, wrist travel) |
| `pianist/midi_input/` | MIDI clock, sustain pedal (CC64; a MIDI without any gets a pedal that follows its bass notes while preserving source pedal changes), hold requirements |
| `pianist/performer/` | Core motion pipeline: `pipeline.py` is the solve entry point, `blender_main.py` writes into Blender; includes `pose.py` / `coherence.py` / `motion.py` (touch poses, chord shapes, wrist paths & per-frame finger IK), `fast_settle.py` (Numba solver kernels), `elbows.py` / `elbows_fast.py` / `elbows_render_fast.py` (3D sliced trunk-hull collision & elbow swing), `attitude.py` (two-pass wrist re-aim & damped smoothing), `torso.py` (lean, bench slide & shoulder protraction), `lap_contact.py` (automatic thigh-surface lap rest fitting), `eyes.py` (head/eye micro-gaze & blinking), `staging.py` (bench, feet & pedal rigging) |
| `pianist/lib/` | Blender-side arm, thumb and body drivers, key geometry and curve writing |
| `pianist/checks/` | `blend_audit.py`: checks a written `.blend`: whether every key and the pedal animation match the solve, whether stale animation is left over, whether the playback settings are right (runs inside Blender) |
| `pianist/inputs/` | `build_piece.py` (turns a MIDI file into planner inputs), the keyboard layout and the finger reach tables |

## Requirements

- **Python** with `numpy`, `scipy`, `mido` and `numba`. **`numba` is strictly required**: all core solves run compiled to machine code, and without `numba` the solver stops immediately at startup and tells you to run `pip install numba`. The first run compiles and caches the result on disk until the code changes.
- **Blender**. Its own Python only reads and writes the scene (no `numba` needed inside Blender); the solve runs in the Python above.
- **Preview audio** (`preview.wav` in the output folder; required: without it the solve stops at its last step, the audio): [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) (the Windows build unpacked into `tools/fluidsynth/`) and Dore Mark's Fazioli F308 SF2 (a Fazioli grand, converted by soundfonts4u; that site is gone, the [Hugging Face mirror](https://huggingface.co/datasets/projectlosangeles/soundfonts4u) has `Dore Mark's Fazioli F308-v2.6.sf2`, which goes into `tools/fazioli/`). If they are somewhere else, set `PIANOMOTION_FLUIDSYNTH` and `PIANOMOTION_SOUNDFONT`. The output is WAV with one uniform gain and nothing else.
- **Tested environment**: primarily developed and verified on Windows 11, Python 3.12, and Blender 5.2 / 5.3. The band-deficit metric in `report.json` uses the platform's built-in General MIDI DLS bank (Windows `gm.dls` / macOS `gs_instruments.dls`) and is skipped automatically if unavailable without affecting motion solving or `preview.wav`.

## Adapting your own character

Motion is solved for a specific model. MMD bone-name conventions are mapped out of the box, and other characters can be adapted via config and environment variables. You need:

- **A Blender scene** with:
  - an armature using MMD (Japanese) bone names (such as `左手首`, `右ひじ`, `左親指０`, `上半身`, `右足ＩＫ`) and the mesh it deforms. The scene may hold only one such armature (head/eye bones and blink shape keys are auto-detected by `rig_export.py` when present);
  - 88 key objects named `key_21` … `key_108` (MIDI pitch);
  - a bench object `bench_seat` (legs `bench_leg*`), the piano's sustain pedal object (default in `PIANO_DAMPER` in `performer/staging.py`) and a floor object `floor` (without one the floor is at height 0). All of these object names can be overridden via environment variables without editing source files.
- **`pianist/hand_calibration.json`**: for every finger, the bone names, joint axes and directions, the neutral pose, the joint ranges and the contact area on the finger pad (`schema: all-digit-calibration`, read by `performer/kinematics.py`, `performer/rig_export.py` and `performer/blender_main.py`; override with `PIANOMOTION_HAND_CALIBRATION`). There is no tool in the repository that generates it; fill it in for your own model.
- **Two tables in `pianist/inputs/instrument/`**: `keyboard.json` holds the key positions of the development scene's piano; `pad_reach_conservative.json` holds each finger pad's conservative reach radius from the wrist; ROM envelopes are sampled directly from the exported rig via `python -m performer.hand_ranges` (override with `PIANOMOTION_ENVELOPES` / `PIANOMOTION_ENVELOPES_OUT` / `PIANOMOTION_RADII`).
- **Seating & scaling**: defaults come from `DEFAULT_SEATING`, `CHARACTER_SCALE` and `HAND_SCALE` in `performer/rig_export.py` (see "Problems encountered during development"), and can be overridden via `PIANOMOTION_SEAT`, `PIANOMOTION_CHARACTER_SCALE` and `PIANOMOTION_HAND_SCALE` without editing source code (keep values identical between `rig_export.py` and `blender_main.py`).

Then run:

```bash
cd pianist
# 1. export the skeleton, skin points, keyboard, fingerprinted trunk hull (<rig>.trunk.json), lap poses & eye rig
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# 2. sample the finger ranges the planner uses from that skeleton (set PIANOMOTION_ENVELOPES_OUT for a custom path)
python -m performer.hand_ranges
# 3. make the planner inputs of a piece from MIDI (use --right/--left for explicit hand tracks)
python inputs/build_piece.py path/to/your_piece.mid your_piece
# 4. planning + motion + writing into Blender
blender -b your_scene.blend --python-exit-code 1 --python performer/blender_main.py -- \
    --inputs inputs/your_piece --output ../output/your_piece
# 5. check the written .blend (optional)
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
| `diagnostics.npz` | per-frame physical diagnostics (finger gaps, overlaps, inter-hand clearance) |

### Common CLI options

| Option | Script | Effect |
|---|---|---|
| `--end 30` | `blender_main.py` / `pipeline.py` | solve the first 30 seconds only, for a trial run |
| `--skip-solve` | `blender_main.py` | no new solve: write the `.blend` again from the existing `motion.npz` |
| `--workers N` | `blender_main.py` / `pipeline.py` | number of solver processes (overrides default performance-core thread count) |
| `--right T --left T` | `inputs/build_piece.py` | specify MIDI track indices for right/left hand (otherwise only tracks named `Right hand`/`Left hand` are split; unassigned notes are split by the planner) |
| `--trunk-only` | `performer/rig_export.py` | re-export only the trunk collision hull `<rig>.trunk.json` |
| `--beam N` | `performer/pipeline.py` | beam search width (default `48`) |
| `--refine-rounds N` | `performer/pipeline.py` | wrist-path refinement rounds (default `2`) |
| `--lift-speed V` / `--motion-cut S` | `performer/pipeline.py` | wrist travel speed (m/s) for release timing / max audible early lift (s) |
| `inputs/<piece>/motion.json` | per-piece config | how high the hands lift in this piece, as a factor, e.g. `{"lift": 0.8}` |

### Environment variables (adaptation & overrides)

| Variable | Effect |
|---|---|
| `PIANOMOTION_PYTHON` | the Python that runs the solve (must have `numba` installed). Unset: the `python` found on `PATH` |
| `PIANOMOTION_FLUIDSYNTH` / `PIANOMOTION_SOUNDFONT` | custom paths to the `fluidsynth` binary and `.sf2` piano soundfont |
| `PIANOMOTION_RIG` | another exported skeleton instead of `performer/rig_cache/rig.json` |
| `PIANOMOTION_HAND_CALIBRATION` | custom character hand calibration JSON (default `pianist/hand_calibration.json`) |
| `PIANOMOTION_ENVELOPES` / `PIANOMOTION_ENVELOPES_OUT` | custom envelopes path read by planner / output path for `hand_ranges` |
| `PIANOMOTION_RADII` | custom character conservative pad reach radii table |
| `PIANOMOTION_CHARACTER_SCALE` / `PIANOMOTION_HAND_SCALE` | override character overall scale / hand scale (must match between `rig_export.py` and `blender_main.py`) |
| `PIANOMOTION_SEAT` | custom seating offset `x,y,z` (metres), defaults to `DEFAULT_SEATING` |
| `PIANOMOTION_BENCH` / `PIANOMOTION_BENCH_LEGS` | override bench seat object name (default `bench_seat`) and comma-separated leg names |
| `PIANOMOTION_DAMPER` / `PIANOMOTION_FLOOR` | override sustain pedal object name and floor object name (default `floor`) |
| `PIANOMOTION_BODY_STILL` | scale upper-body lean and bench slide (`0`..`1`, default `1` restrained; `0` restores full travel) |
| `PIANOMOTION_LAP_CONTACT` / `PIANOMOTION_LAP_SLIDE_FOLLOW` | toggle automatic thigh-surface lap rest fitting (default `1`) and resting-hand bench-slide follow ratio (default `0.91`) |

By default the process count is every thread of the big cores of the CPU: on Intel hybrid (P-cores / E-cores) and Arm big.LITTLE only the performance cores, on a CPU with one kind of core every thread. The processes are pinned to those cores (Windows and Linux read the core types from the system; macOS only takes the number of performance cores, without pinning).

The first run computes the finger pair table for the skeleton (which two fingers can hold which two keys together, `performer/reach_table.py`): about 20–30 seconds on 16 processes, cached beside the rig or in `performer/rig_cache/` until the skeleton or solver code fingerprint changes.

## Problems encountered during development

- **Too small**: the development model is about 1.52 m barefoot and missed black-key octaves by 3.4 mm. The whole character is scaled to 1.05 (about 1.60 m, `CHARACTER_SCALE`). The scale has to be taken about the middle of the bench top; otherwise the whole body rises and the thighs push into the keybed.
- **Small hands**: after that scale the hand still measures only 15 cm from the wrist to the tip of the middle finger (a real hand about 17 cm), and white-key octaves landed on the edges of the keys. Both hands are scaled by a further 1.12 (`HAND_SCALE`).
- **Independent body and hand scaling**: both parameters can be tuned independently until the proportions look right (well-proportioned ~1.65 m models can simply keep scales near `1.0` via environment variables).
- **Aim for at least an octave span with margin**: a real human hand spans roughly a 10th and many chords span an octave; scaling the hands modestly helps reduce omitted notes.
- **The seat**: seated too close and too high, the thighs push into the keybed and the elbows bend too tightly. The present `DEFAULT_SEATING` is further back and lower. When the result is written into Blender, the bench top is lowered to the seated body, the feet are put on the floor and the ball of the right foot on the sustain pedal of the piano (`performer/staging.py`).

## License

[AGPL-3.0](LICENSE)
