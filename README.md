# audio-censor

A Linux command-line tool that finds profanity in your film library and remuxes each
file with an extra **"Clean"** audio track where every curse is ducked to near silence
(or muted, or replaced by a beep).
The original video, audio and subtitle tracks are left untouched. Pick the clean track
in Plex, Jellyfin, Kodi, VLC or mpv when the kids are watching, and the original when
they are not.

Detection combines two sources:

- **Subtitles**: an external `.srt`/`.ass`/`.vtt` next to the file, or an embedded text
  subtitle track. Cheap, catches almost everything, but only knows roughly where in a
  line the word falls. When a film only has image subtitles (Blu-ray PGS), they are
  read by OCR with [Tesseract](https://github.com/tesseract-ocr/tesseract).
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

Optional, to read image (Blu-ray PGS) subtitles: Tesseract with its English data
(`sudo apt install tesseract-ocr`, `sudo dnf install tesseract tesseract-langpack-eng`,
`sudo pacman -S tesseract tesseract-data-eng`) and numpy (`pip install -e '.[ocr]'`;
the `asr` extra already has it). Without them such films fall back to speech recognition alone.

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
audio-censor process Movie.mkv --in-place          # add the clean track to Movie.mkv itself
audio-censor process -r /media/films --in-place    # whole library
audio-censor process Movie.mkv --level mild        # censor more (damn, hell, crap ...)
audio-censor process Movie.mkv --no-asr            # subtitles only, fast
audio-censor process Movie.mkv --duck 0.03         # duck harder (-30 dB); default 0.1 = -20 dB
audio-censor process Movie.mkv --mode mute
audio-censor process Movie.mkv --mode beep --wave square --frequency 440 --volume=-6dB
audio-censor process Movie.mkv --mode beep --beep-file ~/sounds/quack.wav
```

### One copy of each film: `--in-place`

By default the clean version is a second file, `Movie.clean.mkv`. With `--in-place`
(or `in_place = true` under `[output]`) the clean track is added to the original
instead. A Matroska file cannot have a track inserted without being rewritten, so the
tool renders to `Movie.clean.tmp.mkv` in the same folder, checks it (duration, stream
counts, size), swaps it over the original atomically and records the fact in the span
file. The original tracks are all still inside; only the second copy is gone. A failed
check leaves the original untouched. Notes:

- You need free space equal to the film's size while it runs.
- Non-MKV input (MP4, AVI) becomes `Movie.mkv`, since MKV is the container that holds
  every kind of stream; `--backup` keeps the original as `Movie.orig.mp4`.
- The clean track is marked default, so Plex and Jellyfin play it after a library
  refresh. The original audio is one track switch away.
- A re-run skips files already done; `--overwrite` re-renders and replaces the earlier
  clean track rather than adding another.
- If the file is hard-linked (seeding torrents, for example), the rewrite breaks the
  link: the other name keeps the untouched original.

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

### Films on a network share: `[staging]`

`process` reads a film three times (audio for Whisper, the subtitle track, the remux).
Over NFS or SMB each read is a full transfer. With a local staging folder set, a film on
a network share is copied there once, worked on locally, and only new or changed files
(the clean file, the span file, caches, or the rewritten film with `--in-place`) are
copied back beside the original before the local copy is deleted:

```toml
[staging]
dir = "/mnt/fast/audio-censor"
mode = "network"            # network shares only | always | never
```

`--stage-dir DIR` sets it for one run and `--no-stage` turns it off. A film needs room
for itself and its clean copy; without it the run works over the network as before.
Films skipped for their rating are never copied. If a run fails, the scan results and
caches are still copied back and the local copy is kept, so a retry starts without
fetching the film again.

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
takes seconds. `Movie.ocr.json` does the same for OCR'd image subtitles (a name media
servers do not take for a subtitle file, so the uncensored text never shows up as a track);
`--rescan` redoes both, `--no-ocr` skips OCR. Both commands are resumable: `scan` leaves already-scanned files alone
(`--overwrite` to redo), `render` skips files that are not scanned yet or already have
their clean output, and both carry on past a broken file and print a summary.

Other commands: `info FILE` lists streams so you can pick `--audio-track`,
`words` prints the active word list, `preview-beep` writes the configured beep to a
`.wav` so you can audition it, and `init-config` writes a commented config template.

## Live playback without remuxing (mpv)

The span file is all a player needs, so a film can be censored live:

```bash
sudo apt install mpv
audio-censor play Movie.mkv              # scans first if there is no span file yet
audio-censor play Movie.mkv -- --fs      # anything after -- goes to mpv
audio-censor install-mpv                 # from now on plain `mpv Movie.mkv` censors too
```

`play` launches mpv with the bundled `audio-censor.lua` script, which installs a live
ffmpeg filter from the span file (duck, mute or beep following your `[beep]` config,
same timing precision as the rendered track, seeking included), loads censored
subtitles, and flashes the censored words on screen so you can audit a scan before
rendering. Alt+c toggles censoring during playback. After `install-mpv` the script runs
inside any mpv session and acts on every video that has a `.censor.json` beside it;
defaults go in `~/.config/mpv/script-opts/audio-censor.conf` (`mode=mute`,
`frequency=800`, `osd=no` ...). Custom beep sound files apply to rendered output only;
live playback uses a sine beep.

## Whole libraries and foreign-language films

```bash
audio-censor process -r /media/films /media/shows
```

Directories are walked for video files, anything that already has a `.clean.mkv` is
skipped, errors are reported and the batch carries on (`--stop-on-error` to abort), and
a one-line summary is printed at the end. With a Plex server configured (see below),
`--order rating` scans the best-rated titles first (critic score, else audience score),
`--order added` the newest additions first and `--order watched` the most-played first;
files Plex doesn't know go last.

Only English dialogue is beeped. The language comes from the audio track's tag; an
untagged track is identified by Whisper from three short clips when the `asr` extra is
installed, and otherwise assumed English (`scan.assume_untagged`). A film whose
dialogue is in another language keeps its audio, but its English subtitles are still
censored: by default to `Movie.en.clean.srt` beside the original (`--standalone-subs
sidecar`), or `--standalone-subs remux` for a `Movie.clean.mkv` with the clean subtitle
track muxed in, or `skip`. `--any-language` beeps everything regardless, and
`--audio-language eng` overrides a wrong tag. Change `languages` in the config to
censor another language, with your own word list.

### Skipping G-rated titles

Films rated G, TV-Y, TV-Y7 or TV-G are skipped (`scan.skip_ratings`). The rating is
read from a Kodi or Jellyfin NFO beside the film (`Movie.nfo` / `movie.nfo` with `<mpaa>`
or `<certification>`), the show's `tvshow.nfo` for episodes, iTunes-style container tags,
or a Plex server.
Jellyfin and Kodi write NFOs when "save metadata to media folders" is on. Plex does
not, so for a Plex library the tool can ask the server instead:

```bash
export PLEX_URL=https://plex.example.com:32400
export PLEX_TOKEN=...          # or put both under [plex] in the config, chmod 600
```

Every movie and episode Plex knows is fetched once (cached for six hours,
`--refresh-plex` to force) and matched to your files by path, falling back to the
parent folder plus filename when Plex sees the files under a different mount
(`[plex.path_map]` handles that explicitly). Local NFOs take priority over Plex. Films
with no rating anywhere are scanned normally. `--ignore-rating` overrides the skip.

### Runs killed with free memory (CachyOS and other MGLRU kernels)

Some kernels set `/sys/kernel/mm/lru_gen/min_ttl_ms`, a thrashing guard that makes the
kernel kill a process whenever recently used memory cannot be kept that many
milliseconds. At 100 ms it fires during a remux or a 50 GB copy to a network share,
even with gigabytes free (`dmesg` shows "kswapd0 invoked oom-killer"). With

```toml
[system]
relax_thrash_guard = true
```

`process`, `scan` and `render` set it to 0 while they work (through `sudo -n`, so
passwordless sudo for `tee` is needed) and restore the old value when they finish.
Overlapping runs share it, and a run that was killed outright is cleaned up by the next.

### Plex and Jellyfin play the clean track

Plex remembers the last audio track chosen for each film, and that choice beats the
file's default flag, so a film someone has watched before would keep starting on the
original audio. With `[plex]` configured, every `process` and `render` therefore ends by
refreshing the film in Plex, waiting (up to 90 s) until Plex lists the "Clean" track,
and selecting it. The choice is stored for the Plex account whose token is configured;
other users on the server pick it once themselves. `--no-plex-select` or
`select_clean = false` under `[plex]` turns this off. For films Plex had not picked up
yet, or ones rendered before this existed:

```bash
audio-censor media-select -r /media/films
```

Jellyfin chooses a film's audio by the user's preferred language and then the file's
default flag, so it plays the clean track once it has re-read the file. With
`[jellyfin]` configured (`url`, and an `api_key` from Dashboard -> API Keys, or
`JELLYFIN_URL` / `JELLYFIN_API_KEY`), each render refreshes the film in Jellyfin and
waits until it lists the clean track as default. A Jellyfin user with "remember audio
selections" on who already picked another track keeps that choice.

### Clean audio beside the film, no remux: `--audio-sidecar`

```bash
audio-censor process --audio-sidecar "/media/films/Hook (1991)"
```

writes only the clean track, as `<film>.Clean (muted).eng.default.mka` next to the
film, plus the clean subtitles. The film itself is never rewritten. Jellyfin loads
external audio files with the film and, from the `.default` in the name, plays the
clean one by default. Plex ignores external audio, so this mode is for Jellyfin
libraries. Remux later with a normal `process --overwrite` if Plex needs it too.

## Skipping scenes in Jellyfin

Audio and subtitles can be cleaned; pictures cannot. For a scene that should not be
seen at all, `skip` marks it for Jellyfin players to jump over, without touching the
video:

```bash
audio-censor skip Movie.mkv 1:02:03-1:02:40 --note "pool scene"   # add a scene
audio-censor skip Movie.mkv                                       # list scenes; Jellyfin status
audio-censor skip Movie.mkv --remove 1                            # or --clear for all
```

Each scene becomes a chapter named "Skip" (and the interrupted chapter resumes after
it); `mkvpropedit` rewrites the chapter list in place, so there is no remux. The
original chapters are kept in `Movie.skip.json` and later renders keep the scenes.

One-time Jellyfin setup:

1. Install the **Chapter Segments Provider** plugin (Dashboard -> Plugins -> Catalog) and
   restart Jellyfin.
2. In its settings set the **Commercial** mapping to `^Skip$`. Its default
   (`break|ad|...`, unanchored) also matches ordinary chapter names such as "The Road".
3. In each Jellyfin app: Settings -> Playback -> Media Segments -> Commercial: **Skip**.

Jellyfin builds segments only in its library-wide "Media Segment Scan" task (minutes on a
big library), which `skip` starts; run `skip Movie.mkv` later to see them. A refresh of
the film clears its segments until that scan runs again, so renders of a film with
scenes start the scan too. Plex shows the extra "Skip" chapter but cannot skip it.

## Configuration

`audio-censor init-config` writes `~/.config/audio-censor/config.toml`. Every key is
optional; command-line flags override the file.

```toml
level = "moderate"          # strong | moderate | mild   (mild censors the most)

[beep]
mode = "duck"               # duck | mute | beep
duck = 0.1                  # dialogue level inside a span: 0.1 = -20 dB, 0.03 = -30 dB
duck_channels = "center"    # 5.1/7.1: lower only the dialogue channel, music plays on; or "all"
wave = "sine"               # beep mode: sine | square | triangle | sawtooth | noise | file
frequency = 1000
volume = 0.4                # linear 0..1, or "-8dB"
file = "~/sounds/boing.wav" # used when wave = "file"; short files loop to fill the span
channel = "center"          # beep only in the dialogue channel of a 5.1 mix, or "all"

[output]
in_place = true             # add the clean track to the original file

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

### Words rude in only one sense: `[[context.rules]]`

"Balls", "nuts", "come" or "bitch" are usually innocent. Words named in a context rule
are judged each time by a local AI model served by [Ollama](https://ollama.com)
(`gemma4:12b` by default, under `[context]`). It is shown the subtitle line with its
neighbours, or the transcript around the word, and asked whether the word is used in the
rule's `meaning` or its `innocent` sense:

```toml
[[context.rules]]
words = ["balls", "nuts"]
meaning = "testicles, or crude slang and jokes about them"
innocent = "a literal ball or nut, or meaning crazy"
when_unsure = "keep"        # no AI available: mute only on a fallback phrase
fallback = ["bust* * balls", "in the nuts"]
ask_twice = true            # also ask about the line alone; either "yes" mutes
```

A rule that names a word on the built-in list (`bitch*`, `prick`) takes it over. With
`when_unsure = "mute"` the AI is only asked whether the use is clearly innocent, and
anything unclear stays muted. Answers are cached, so a re-run asks nothing new. Each
film's decisions are recorded in its `.censor.json`; `audio-censor review FILM` lists
them. While a game is running, or Ollama is down, the fallback phrases decide.

When the AI gets one wrong, correct it:

```bash
audio-censor correct "a private [[dick]]" --keep --find /media/films
audio-censor correct "kicked him in the nuts" --mute
audio-censor corrections                 # list, or --remove N
```

A correction decides every use the phrase surrounds, and is shown to the AI as a worked
example for similar lines. Short phrases match more. `--find` lists the films whose last
scan decided otherwise, which then need `process --overwrite`.

Only the rude word of a listed phrase is muted: "son of a [bitch]", "god [damn]".

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

1. The chosen dialogue track is ducked (default, to -20 dB), muted, or gated to silence
   inside every span, at 5 ms resolution. On a 5.1 or 7.1 track only the centre channel,
   where films put the dialogue, is lowered, so music and effects in the other channels
   keep playing (`duck_channels = "all"` or `--duck-channels all` lowers everything).
   Stereo tracks have no separate dialogue channel and are lowered as a whole.
2. In beep mode one beep is synthesized per span, faded in and out, delayed to the span
   start, and for surround tracks placed in the center channel only, so music and
   effects in the other channels keep playing.
3. The result is encoded (AAC for stereo, AC-3 for 5.1 by default; override with
   `--codec`) and muxed as an additional audio track titled "Clean (ducked)", marked
   default so players pick it automatically. Pass `--no-default` to keep the original
   as default or `--replace-audio` to drop the originals.
4. The censored subtitle file is muxed in as a "Clean" subtitle track in the same pass.

Speech recognition listens to the dialogue (centre) channel of a 5.1/7.1 track
rather than a downmix, so music and effects drown out fewer words
(`channel = "mix"` under `[asr]` restores the downmix). Any swear the subtitles have
but the transcript missed is then listened to again on its own: Whisper re-reads just
that subtitle line's few seconds with a wider search. A word heard this way is muted
exactly; otherwise the mute is placed by its position in the line (`recheck = false`
turns the second pass off). Transcripts made from a downmix are redone automatically.

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
