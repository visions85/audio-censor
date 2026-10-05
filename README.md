# audio-censor

A Linux command-line tool that finds profanity in your film library and remuxes each
file with an extra **"Clean"** audio track where every curse is replaced by a beep.
The original video, audio and subtitle tracks are left untouched. Pick the clean track
in Plex, Jellyfin, Kodi, VLC or mpv when the kids are watching, and the original when
they are not.

Detection combines two sources:

- **Subtitles**: an external `.srt`/`.ass`/`.vtt` next to the file, or an embedded text
  subtitle track. Cheap, catches almost everything, but only knows roughly where in a
  line the word falls.
- **Speech recognition** (optional): [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  transcribes the dialogue with per-word timestamps, so each beep lands on the word
  itself. When both agree, the tighter ASR timing wins.

## Install

Prerequisites: Python 3.11 or newer, pip, and ffmpeg.

```bash
# Ubuntu / Debian / Mint
sudo apt update && sudo apt install python3-pip python3-venv ffmpeg
# Fedora
sudo dnf install python3-pip ffmpeg
# Arch
sudo pacman -S python-pip ffmpeg
```

Then, from a checkout of this repository, install into a virtual environment
(Debian and Ubuntu refuse a system-wide `pip install`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # repeat in each new terminal
pip install -e '.[asr]'            # with speech recognition
pip install -e .                   # subtitles only, much smaller
audio-censor --version
```

Alternatively `sudo apt install pipx` and `pipx install -e '.[asr]'` puts the
`audio-censor` command on your path permanently without activating anything.

Speech recognition runs on the CPU by default (roughly a quarter of real time with
the `small` model, so about 30 minutes for a 2-hour film). With an NVIDIA GPU it
takes a couple of minutes; install faster-whisper's CUDA libraries and set
`device = "cuda"` or pass `--device cuda`.

## Usage

```bash
audio-censor process Movie.mkv                     # scan + remux -> Movie.clean.mkv
audio-censor process -r /media/films --keep-going  # whole library
audio-censor process Movie.mkv --level mild        # censor more (damn, hell, crap ...)
audio-censor process Movie.mkv --no-asr            # subtitles only, fast
audio-censor process Movie.mkv --wave square --frequency 440 --volume=-6dB
audio-censor process Movie.mkv --beep-file ~/sounds/quack.wav
audio-censor process Movie.mkv --mode mute --duck 0.1
```

Clean subtitles come along for free: the words found in the subtitle text are masked
and written to `Movie.clean.en.srt` beside the output and embedded as a subtitle track
titled "Clean". Choose the masking with `--sub-style`:

| style          | result                      |
|----------------|-----------------------------|
| `asterisks`    | `Oh, ****. That is a **** shame.` (default) |
| `first-letter` | `Oh, s***. That is a d*** shame.` |
| `bleep`        | `Oh, [BLEEP]. That is a [BLEEP] shame.` (text via `--sub-replacement`) |
| `remove`       | `Oh, . That is a shame.`    |

`--no-clean-subs` turns it off; `[subtitles]` in the config picks sidecar, embedded or both.

### Scan now, remux later

`process` does everything in one go, but the two halves also run separately, which is
how you'd treat a large library: let the slow speech-recognition pass run overnight,
review or hand-edit what it found, then remux whenever you like, possibly on a
different machine or with a different beep.

```bash
audio-censor scan -r /media/films      # phase one: Movie.censor.json beside each film
audio-censor status -r /media/films    # unscanned / scanned / rendered / nothing to do
audio-censor review Movie.mkv          # print one film's spans
$EDITOR Movie.censor.json              # add, remove or nudge spans
audio-censor render -r /media/films    # phase two: remux from the span files
```

Everything the remux step needs lives beside the film: `Movie.censor.json` holds the
spans, the chosen audio track, the word level and the language decision, and
`Movie.transcript.json` caches the Whisper output so a re-scan with a changed word list
takes seconds. Both commands are resumable: `scan` leaves already-scanned files alone
(`--overwrite` to redo), `render` skips files that are not scanned yet or already have
their clean output, and both carry on past a broken file and print a summary.

Other commands: `info FILE` lists streams so you can pick `--audio-track`,
`words` prints the active word list, `preview-beep` writes the configured beep to a
`.wav` so you can audition it, and `init-config` writes a commented config template.

## Whole libraries and foreign-language films

```bash
audio-censor process -r /media/films /media/shows
```

Directories are walked for video files, anything that already has a `.clean.mkv` is
skipped, errors are reported and the batch carries on (`--stop-on-error` to abort), and
a one-line summary is printed at the end.

Only English dialogue is beeped. The language comes from the audio track's tag; an
untagged track is identified by Whisper from three short clips when the `asr` extra is
installed, and otherwise assumed English (`scan.assume_untagged`). A film whose
dialogue is in another language keeps its audio, but its English subtitles are still
censored: by default to `Movie.en.clean.srt` beside the original (`--standalone-subs
sidecar`), or `--standalone-subs remux` for a `Movie.clean.mkv` with the clean subtitle
track muxed in, or `skip`. `--any-language` beeps everything regardless, and
`--audio-language eng` overrides a wrong tag. Change `languages` in the config to
censor another language, with your own word list.

## Configuration

`audio-censor init-config` writes `~/.config/audio-censor/config.toml`. Every key is
optional; command-line flags override the file.

```toml
level = "moderate"          # strong | moderate | mild   (mild censors the most)

[beep]
mode = "beep"               # beep | mute
wave = "sine"               # sine | square | triangle | sawtooth | noise | file
frequency = 1000
volume = 0.4                # linear 0..1, or "-8dB"
file = "~/sounds/boing.wav" # used when wave = "file"; short files loop to fill the span
channel = "center"          # beep only in the dialogue channel of a 5.1 mix, or "all"
duck = 0.0                  # how much of the original dialogue survives under the beep

[subtitles]
clean = true
style = "asterisks"         # asterisks | first-letter | bleep | remove
sidecar = true              # Movie.clean.en.srt next to the video
embed = true                # plus a "Clean" subtitle track inside the file

[names]
detect = true               # Dick the character vs. dick the insult (see below)

[words]
extra = ["moist"]           # censored at every level
allow = ["dick"]            # never censored anywhere, whatever the context
[words.tiers]
mild = ["heck*"]            # add patterns to a tier
```

Pattern syntax: `word` matches a whole word, `word*` a prefix (`fuck*` covers fucking,
fucker...), `two words` a phrase, `re:...` a regular expression. The built-in list lives
in `audio_censor/data/default_words.toml`, grouped into **strong** (f-word, slurs),
**moderate** (shit, bitch, asshole, goddamn...) and **mild** (damn, hell, crap...).
`--level strong` censors only the first tier; `--level mild` censors all three.

Whisper occasionally writes `f***ing` instead of the word; such tokens are always
treated as hits.

Patterns are deliberately narrow: `dick` and `dicks` rather than `dick*`, so Dickens,
Dickinson and farthing are left alone. If Whisper hears "damn" where the subtitles say
"Hoover Dam", the subtitle wins: `scan.homophones` lists the innocent sound-alikes that
veto an ASR-only hit when they appear in the subtitle line at that moment.

### Names versus swears

A film can have a character called Dick, or a Fagin who trips the `fag*` prefix. The
scanner learns per film which flagged words are names: a word written with a capital
in the middle of a sentence at least twice (`names.min_occurrences`), or used as an
SDH speaker label (`DICK:`), is a name for that film. Each occurrence is then judged on
its own:

| line                              | verdict                                   |
|-----------------------------------|-------------------------------------------|
| `Hey Dick, pass the ball.`        | name, left alone                          |
| `Don't be such a dick about it.`  | swear, beeped                             |
| `You're a Dick, you know that?`   | swear: "a Dick" has a determiner in front |
| `Dick! Over here!`                | sentence-initial, so the film-level verdict decides |

The same classifier runs on Whisper's transcript, which capitalizes proper nouns too,
and an ASR word inside a subtitle line judged to be a name is exempted with it. Clean
subtitles keep the name and mask the insult. The scan log reports what it decided:

```
names detected: Dick (3 capitalized, 1 speaker label(s))
```

`--no-names` or `names.detect = false` turns this off; `names.ignore = ["dick"]`
excludes one word; `words.allow` remains the blunt instrument that always exempts a
word regardless of context.

## How the clean track is built

ffmpeg does all the audio work in one pass, so a two-hour film remuxes in about the
time it takes to re-encode one audio track:

1. The chosen dialogue track is gated to silence (or ducked) inside every span, at
   5 ms resolution.
2. One beep is synthesized per span, faded in and out, and delayed to the span start.
3. For surround tracks the beeps are placed in the center channel only, so music and
   effects in the other channels keep playing.
4. The result is encoded (AAC for stereo, AC-3 for 5.1 by default; override with
   `--codec`) and muxed as an additional audio track titled "Clean (beeped)", marked
   default so players pick it automatically. Pass `--no-default` to keep the original
   as default or `--replace-audio` to drop the originals.
5. The censored subtitle file is muxed in as a "Clean" subtitle track in the same pass.

Output defaults to `<name>.clean.mkv` beside the source. MP4 input works; text
subtitles are converted to SRT for the MKV container.

## Limitations

- Subtitle-only scans (no ASR) estimate the word's position from its place in the
  line and pad by 0.35 s, so beeps are longer than necessary. Spans are marked `~` in
  the table. Install the `asr` extra for tight timing.
- Speech recognition misses words under loud music or heavy accents and sometimes
  hallucinates; the subtitle pass covers most misses, and `scan` + `review` lets you
  correct the rest before rendering.
- Lip movement of course still shows the original word.
