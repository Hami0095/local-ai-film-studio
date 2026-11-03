# Architecture: from one clip to a full local AI film

Today the studio does one thing reliably: it turns a prompt (and optionally an image) into a validated MP4 using a local GPU. Every later stage is a **local, swappable component** that reads and writes files in a per-project folder. The GPU runs **one model at a time**, and the engine's load/unload cycle enforces that.

```text
idea ──► [1 Screenplay LLM] ──► screenplay.md
                │
                ▼
         [2 Scene/shot breakdown] ──► shotlist.json   ◄── the contract between stages (exists today)
                │
     ┌──────────┼─────────────────────────────┐
     ▼          ▼                             ▼
[3 Character refs]   [6 Dialogue + TTS]   [7 Music]  [8 SFX]
  refs/<char>.png      audio/dialog_*.wav   audio/score.wav  audio/sfx_*.wav
     │
     ▼
[4/5 Wan2.2 TI2V-5B: I2V from refs or T2V]  ──► shots/<id>.mp4   (src/generation/engine.py, exists today)
     │
     ▼
[9 Assembly: ffmpeg concat/xfade + audio mix + loudness] ──► film.mp4   (concat exists today: src/pipeline/shotlist.py)
```

## The shot list contract

`prompts/example_shotlist.json` is the seed format. Later stages add fields to it instead of inventing new formats:

```json
{"id": "s02", "prompt": "...", "image": "refs/astronaut.png", "seed": 7, "duration": 5,
 "characters": ["astronaut"], "dialogue": [{"speaker": "astronaut", "text": "...", "voice": "v1"}],
 "sfx": ["rain", "distant engine hum"], "music_cue": "tension", "transition_in": "xfade:0.5"}
```

`render_shotlist()` already ignores keys it doesn't know, so the format can grow without breaking video rendering.

## Components

Every candidate below must stay local and open-weight. **Check each model's license before adopting it.** Several popular audio models are non-commercial.

| # | Component | Integration point | Notes |
|---|---|---|---|
| 1 | Screenplay LLM | `src/pipeline/screenplay.py`. Prompt goes in, Markdown screenplay comes out | Run a local open-weight LLM through llama.cpp, Ollama or transformers. Unload it before video jobs, because 16 GB is shared |
| 2 | Scene breakdown | Same LLM with a JSON-schema-constrained output that produces `shotlist.json` | Validate against a schema. Each shot is at most 5 s, the TI2V-5B limit, so long scenes become several shots |
| 3 | Character references | `src/pipeline/characters.py`. A local text-to-image model produces `refs/<name>.png` | Fix the seed and description per character. Later: LoRA or IP-Adapter style identity conditioning |
| 4 | Image-to-video | **Exists.** `GenerationRequest(image=...)` uses `WanImageToVideoPipeline` with the same 5B weights | The main tool for consistent characters: start each shot from that character's reference frame |
| 5 | Text-to-video | **Exists.** `VideoEngine.generate()` | Use for establishing shots, environments and B-roll |
| 6 | Dialogue and voice | `src/pipeline/voice.py`. Text plus a voice ID produce a WAV | A local TTS model. Shot duration should follow dialogue length; extend a shot with a last-frame-to-I2V continuation |
| 7 | Music | `src/pipeline/music.py`. A cue description and duration produce a WAV | A local music model, or a royalty-free library |
| 8 | Sound effects | `src/pipeline/sfx.py` | A local text-to-audio model, or a curated SFX library |
| 9 | Assembly | Extend `shotlist.py`: ffmpeg `xfade` transitions, `amix` for the dialogue/music/SFX stems, EBU R128 `loudnorm` | Pure ffmpeg, no GPU. Stream-copy concat already works |
| 10 | Shot continuity | The last frame of shot N becomes the I2V input for shot N+1 | Already possible: `validate_video` exports frames, and the engine accepts `image=` |

## Engineering rules for the pipeline

- **One GPU model at a time.** Stages run as a queue (`load`, run the whole batch, `unload`). Never keep the LLM and Wan resident together on 16 GB.
- **Everything is a file.** Each project gets `projects/<name>/{screenplay.md, shotlist.json, refs/, shots/, audio/, film.mp4}`. That makes every stage resumable and cacheable: skip a shot if its MP4 and metadata JSON already exist with the same parameters.
- **Same seeds and pinned model revisions** make every shot reproducible. The metadata JSON beside each MP4 already records seed, revision and versions.
- **Validate every artefact.** `validate_video` already covers video. Audio stages need the equivalent: duration, clipping and silence detection.
- **Machine A develops** against the tiny backend and small LLM/TTS checkpoints. **Machine B renders.** A future `main.py serve` on Machine B could take shot-list jobs from Machine A over the LAN.
