# midi-to-PianoMotion

[中文](README.md)

An unofficial open-source recreation of the piano performance feature of **BSide: Olivia Lin**. Not affiliated with miHoYo.

Give it a MIDI file and a 3D character plays it on a piano in Blender:

1. **Planning**: decides which hand and which finger plays each note, when each finger lets go of its key, and which notes have to be left out because the hand cannot reach them. The planner splits the notes between the hands itself; the score does not need to be separated into hands.
2. **Motion**: animates the arms, wrists, fingers, the lean of the upper body, the right foot on the sustain pedal and every key, and writes it all into a `.blend` with preview audio.

No training data and no neural network: this is a fairly lightweight, purely algorithmic approach. Planning is a beam search; motion is inverse kinematics solved on the character's own skeleton and skin. Everything runs on the CPU in parallel processes (8 or 16). For a 160-second piece, planning takes a dozen seconds or so, and the motion about 13 minutes on 16 cores.

> **Demo video:** https://www.bilibili.com/video/BV1QEnc6rECK/

## Why

In BSide: Olivia Lin you could hand Olivia the MIDI of a piece, and she would sit down at the piano by the window and play it for you.

At the end of August 2026 the official service shut down. The last piece I gave her before it closed was Seredris's Daylight. She played it three times: once at night, once by day, once at dusk. Those three videos are still sitting in a cache folder on my computer, but I can never give her a new piece again.

So this project exists. I want to recreate that feature as faithfully as I can.

## What is in the repository

Only the algorithm. The character, piano and scene models used during development cannot be published because of their licenses. Adapting the algorithm to a model takes some effort; using an AI tool to help is recommended.

| Folder | Contents |
|---|---|
| `pianist/planner/` | Fingering: a beam search over groups of notes that start together, choosing the hand, the finger, when to release and which notes to omit. Runs in parallel segments whose joined result equals the serial one. Includes an independent validation and listening-fidelity measures |
| `pianist/global_planning/` | Hand rules used to screen candidates while planning (reachable spans, rotation range with a fixed palm, wrist travel) |
| `pianist/midi_input/` | MIDI clock, sustain pedal (CC64), hold requirements |
| `pianist/performer/` | Motion: the key pose of each touch, consistent hand shapes, the wrist path, finger strokes, per-frame finger IK, arms, upper body, pedal. `pipeline.py` is the entry point; `blender_main.py` writes the result into Blender |
| `pianist/lib/` | Helper modules: Blender-side arm and thumb drivers, key geometry and curve writing, plus the preview-audio synthesis |
| `pianist/checks/` | `blend_audit.py`: checks a written `.blend`: whether every key and the pedal animation match the solve, whether stale animation is left over, whether the playback settings are right (runs inside Blender) |
| `pianist/inputs/` | `build_piece.py` (turns a MIDI file into planner inputs), the keyboard layout and the finger reach tables |

## Requirements

- Python (with `numpy`, `scipy`, `mido`) and Blender.
- Developed on Windows 11, Python 3.12 and Blender 5.2; other setups have not been tested.

## The development character

For convenience, development used the official *Genshin Impact* MMD model of **Furina**, and all the algorithm's parameters were tuned and verified on her. The model's license does not allow shipping it with the repository; you need to get it yourself.

With some adaptation the algorithm also works with a custom character, but the character's height matters. It caused two problems during development:

- **Hands too small**: black-key octaves were out of reach. In the end the whole character was scaled up a little (to about 1.60 m barefoot). The scale has to be taken about the middle of the bench top; otherwise the whole body rises and the thighs push into the keybed.
- **Legs too short**: seated on the bench, the knees are level with the seat and the feet reach neither the floor nor the pedals. So a pedal extender was added (`performer/staging.py`): the right foot presses it, and it moves the piano's sustain pedal.

## Using your own character

The motion is solved for a specific model. Development time was limited, so please bear with the rough edges. To use another MMD model you need:

- **A Blender scene** with:
  - an armature using MMD (Japanese) bone names (such as `左手首`, `右ひじ`, `左親指０`) and the mesh it deforms;
  - 88 key objects named `key_21` … `key_108` (MIDI pitch);
  - a bench object `bench_seat` (legs `bench_leg*`) and the piano's sustain pedal (see `PIANO_DAMPER` in `performer/staging.py`).
- **`pianist/hand_calibration.json`**: for every finger, the bone names, joint axes and directions, the neutral pose, the joint ranges and the contact area on the finger pad (`schema: all-digit-calibration`, read by `performer/kinematics.py`, `performer/rig_export.py` and `performer/blender_main.py`). There is no tool in the repository that generates it; fill it in for your own model.
- **`pianist/inputs/instrument/pad_reach_conservative.json`**: how far each finger pad can reach from the wrist, used while planning to decide what the hand can reach. The values in the repository are for the development model scaled by 1.05; for another model they have to be measured again on its skeleton, and there is no tool for that in the repository either.

Then run:

```bash
cd pianist
# export the skeleton, skin points and keyboard for the numpy solver
blender -b your_scene.blend --python performer/rig_export.py -- --output performer/rig_cache/rig.json
# sample the finger ranges the planner uses from that skeleton
python -m performer.hand_ranges
# make a piece's planner inputs from MIDI (if the file has separate tracks per hand, name them with --right/--left)
python inputs/build_piece.py path/to/your_piece.mid your_piece --right 1 --left 2
# planning + motion + writing into Blender (performer.blend in the output folder)
blender -b your_scene.blend --python-exit-code 1 --python performer/blender_main.py -- \
    --inputs inputs/your_piece --output ../output/your_piece --workers 16
```

The first run computes the finger pair table for the skeleton (`performer/reach_table.py`, about 10 minutes on 16 threads) and caches it in `performer/rig_cache/`. `blender_main.py` runs the solve with the Python found on `PATH`; the environment variable `PIANOMOTION_PYTHON` can name another one.

The preview audio (`planner/audio.py`) uses the General MIDI sound bank that comes with Windows (`gm.dls`).

## License

[AGPL-3.0](LICENSE)
