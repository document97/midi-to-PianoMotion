# midi-to-PianoMotion

[中文](README.md)

An unofficial open-source recreation of the piano performance feature of **BSide: Olivia Lin**. Not affiliated with miHoYo.

Give it a MIDI file and a 3D character plays it on a piano in Blender:

1. **Planning**: decides which hand and which finger plays each note, when each finger lets go of its key, and which notes have to be left out because the hand cannot reach them. The planner splits the notes between the hands itself; the score does not need to be separated into hands.
2. **Motion**: animates the arms, wrists, fingers, the upper body, the right foot on the sustain pedal and every key, and writes it all into a `.blend` with preview audio.

No training data and no neural network: this is a fairly lightweight, purely algorithmic approach. Planning is a beam search; motion is inverse kinematics solved on the character's own skeleton and skin. Everything runs on the CPU in parallel processes, the most expensive solves compiled to machine code with numba. A 160-second piece (Daylight) takes about 8 minutes on 16 processes.

> **Demo video:** https://www.bilibili.com/video/BV1QEnc6rECK/

## Why

In BSide: Olivia Lin you could hand Olivia the MIDI of a piece, and she would sit down at the piano by the window and play it for you.

At the end of August 2026 the official service shut down. The last piece I gave her before it closed was Seredris's Daylight. She played it three times: once at night, once by day, once at dusk. Those three videos are still sitting in a cache folder on my computer, but I can never give her a new piece again.

So this project exists. I want to recreate that feature as faithfully as I can.

## What to expect

- **Notes out of reach are left out, not forced.** Which keys one hand can hold together is measured on the character's own hand; keys at the far ends of the keyboard that stay out of reach after the upper body has leaned and slid along the bench are left out too. The top and bottom voices and the louder, longer notes are kept first. Daylight loses 15 of its 1049 notes; dense arrangements with wide spans lose many more (an extreme test piece of 3135 notes lost 18%).
- **Crossed hands are not supported.**
- **Playing motion only.** The upper body leans forward and sideways and slides along the bench, the shoulders come forward for a far reach, and the hands rest on the lap before and after the piece. There is no expressive movement with the music, clothes and hair are not handled, and nothing is rendered.
- **Tuned on one character and one piano only** (see "The development character"), and run on Windows only.

## What is in the repository

Only the algorithm. The character, piano and scene models used during development cannot be published because of their licenses. Adapting the algorithm to a model takes some effort; using an AI tool to help is recommended.

| Folder | Contents |
|---|---|
| `pianist/planner/` | Fingering: a beam search over groups of notes that start together, choosing the hand, the finger, when to release and which notes to omit. Includes an independent validation and a listening-fidelity measure |
| `pianist/global_planning/` | Hand rules used to screen candidates while planning (reachable spans, rotation range with a fixed palm, wrist travel) |
| `pianist/midi_input/` | MIDI clock, sustain pedal (CC64; a MIDI without any gets a pedal that follows its bass notes), hold requirements |
| `pianist/performer/` | Motion: the pose of each touch, consistent hand shapes, the wrist path, finger strokes, per-frame finger IK, arms, upper body, pedal. `pipeline.py` is the entry point of the solve; `blender_main.py` writes the result into Blender |
| `pianist/lib/` | Blender-side arm, thumb and body drivers, key geometry and curve writing |
| `pianist/checks/` | `blend_audit.py`: checks a written `.blend`: whether every key and the pedal animation match the solve, whether stale animation is left over, whether the playback settings are right (runs inside Blender) |
| `pianist/inputs/` | `build_piece.py` (turns a MIDI file into planner inputs), the keyboard layout and the finger reach tables |

## Requirements

- **Python** with `numpy`, `scipy`, `mido` and `numba`. `numba` is required: the most expensive solves run compiled, about ten times faster than numpy, and without it the program stops and says to install it. The first run compiles, and the result stays on disk until the code changes.
- **Blender**. Its own Python only reads and writes the scene; the solve runs in the Python above.
- **Preview audio** (`preview.wav` in the output folder; required: without it the solve stops at its last step, the audio): [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) (the Windows build unpacked into `tools/fluidsynth/`) and Dore Mark's Fazioli F308 SF2 (a Fazioli grand, converted by soundfonts4u; that site is gone, the [Hugging Face mirror](https://huggingface.co/datasets/projectlosangeles/soundfonts4u) has `Dore Mark's Fazioli F308-v2.6.sf2`, which goes into `tools/fazioli/`). If they are somewhere else, set `PIANOMOTION_FLUIDSYNTH` and `PIANOMOTION_SOUNDFONT`. The output is WAV with one uniform gain and nothing else.
- Developed on Windows 11, Python 3.12 and Blender 5.2; other setups have not been tested. The listening measure in the report (how many dB each band lacks against the source) reads the General MIDI samples that come with Windows (`gm.dls`); on another system that has to be replaced (`SOUNDBANK` in `planner/audio.py`).

## Using your own character

The motion is solved for a specific model. Development time was limited, so please bear with the rough edges. To use another MMD model you need:

- **A Blender scene** with:
  - an armature using MMD (Japanese) bone names (such as `左手首`, `右ひじ`, `左親指０`, `上半身`, `右足ＩＫ`) and the mesh it deforms. The scene may hold only one such armature;
  - 88 key objects named `key_21` … `key_108` (MIDI pitch);
  - a bench object `bench_seat` (legs `bench_leg*`), the piano's sustain pedal object (named in `PIANO_DAMPER` in `performer/staging.py`) and a floor object `floor` (without one the floor is at height 0).
- **`pianist/hand_calibration.json`**: for every finger, the bone names, joint axes and directions, the neutral pose, the joint ranges and the contact area on the finger pad (`schema: all-digit-calibration`, read by `performer/kinematics.py`, `performer/rig_export.py` and `performer/blender_main.py`). There is no tool in the repository that generates it; fill it in for your own model.
- **Two tables in `pianist/inputs/instrument/`**: `keyboard.json` holds the key positions of the development scene's piano; `pad_reach_conservative.json` holds how far each finger pad can reach from the wrist, used while planning to decide what the hand can reach, with the values of the scaled development model. For another piano or another model they have to be measured again in your own scene, and there is no tool for that in the repository either.
- **Three values at the top of `performer/rig_export.py`**: the seat `DEFAULT_SEATING`, the overall scale `CHARACTER_SCALE` and the hand scale `HAND_SCALE`. They are tuned for the development model; see "The development character".

Then run:

```bash
cd pianist
# export the skeleton, skin points and keyboard for the solve
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# sample the finger ranges the planner uses from that skeleton
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
| `report.json` | omitted and shortened notes, what each register lacks, motion statistics, seconds per stage |
| `motion.npz` | every animation curve (60 samples per second) |

Options and environment variables:

| Setting | Effect |
|---|---|
| `--end 30` | solve the first 30 seconds only, for a trial run |
| `--skip-solve` | no new solve: write the `.blend` again from the results already in the output folder |
| `--workers N` | number of solver processes |
| `inputs/<piece>/motion.json` | how high the hands lift in this piece, as a factor, e.g. `{"lift": 0.8}` |
| `PIANOMOTION_PYTHON` | the Python that runs the solve (the one that has numba). Unset: the `python` found on `PATH` |
| `PIANOMOTION_RIG` | another exported skeleton instead of `performer/rig_cache/rig.json` |

By default the process count is every thread of the big cores of the CPU: on Intel hybrid (P-cores / E-cores) and Arm big.LITTLE only the performance cores, on a CPU with one kind of core every thread. The processes are pinned to those cores (Windows and Linux read the core types from the system; macOS only takes the number of performance cores, without pinning).

The first run computes the finger pair table for the skeleton (which two fingers can hold which two keys together, `performer/reach_table.py`): about 20 seconds on 16 processes, cached in `performer/rig_cache/` until the skeleton or the solver code changes.

## The development character

For convenience, development used the official *Genshin Impact* MMD model of **Furina**, and all the algorithm's parameters were tuned and verified on her. The model's license does not allow shipping it with the repository; you need to get it yourself.

With some adaptation the algorithm also works with a custom character, but the build of the character decides what it can play. These problems came up during development:

- **Too small**: the model is about 1.52 m barefoot and missed black-key octaves by 3.4 mm. The whole character is scaled to 1.05 (about 1.60 m, `CHARACTER_SCALE`). The scale has to be taken about the middle of the bench top; otherwise the whole body rises and the thighs push into the keybed.
- **Small hands**: after that scale the hand still measures only 15 cm from the wrist to the tip of the middle finger (a real hand about 17 cm), and white-key octaves landed on the edges of the keys. Both hands are scaled by a further 1.12 (`HAND_SCALE`).
- **The seat**: seated too close and too high, the thighs push into the keybed and the elbows bend too tightly. The present `DEFAULT_SEATING` is further back and lower. When the result is written into Blender, the bench top is lowered to the seated body, the feet are put on the floor and the ball of the right foot on the sustain pedal of the piano (`performer/staging.py`).

## License

[AGPL-3.0](LICENSE)
