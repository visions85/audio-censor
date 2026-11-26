#!/bin/bash
# Live view of an audio-censor run.
#   tools/watch-run.sh LOGFILE...        everything: copying, OCR, transcription, AI decisions,
#                                        encoding, Plex/Jellyfin (progress updates in place)
#   tools/watch-run.sh --brief LOGFILE...  only films, AI decisions, mutes and results
# With several logs (parallel runs), each line is tagged with its run: [1], [2] ...
brief=0
[ "$1" = "--brief" ] && { brief=1; shift; }
[ $# -ge 1 ] || { echo "usage: watch-run.sh [--brief] LOGFILE..."; exit 1; }
multi=0; [ $# -gt 1 ] && multi=1
names=$(printf '%s\n' "$@")
tail -n +1 -F "$@" 2>/dev/null | gawk -v brief="$brief" -v multi="$multi" -v names="$names" '
BEGIN {
  RS = "\r|\n"
  red = "\033[31m"; green = "\033[32m"; yellow = "\033[33m"; cyan = "\033[36m"
  bold = "\033[1m"; dim = "\033[2m"; off = "\033[0m"; clr = "\r\033[K"
  width = ENVIRON["COLUMNS"] ? ENVIRON["COLUMNS"] - 1 : 150
  n = split(names, files, "\n"); for (k = 1; k <= n; k++) runof[files[k]] = k
  tagcol[1] = "\033[35m"; tagcol[2] = "\033[34m"; tagcol[3] = "\033[33m"; tagcol[4] = "\033[36m"
  tag = ""; r = 1
}
# tail marks which file the next lines come from
/^==> .* <==$/ {
  fname = substr($0, 5, length($0) - 8)
  if (fname in runof) { r = runof[fname]; tag = multi ? tagcol[(r - 1) % 4 + 1] "[" r "]" off " " : "" }
  next
}
# ffmpeg chatter that says nothing useful
/Could not find codec parameters|Lossless check failed|Consider increasing the value|^tail: / { next }
{
  line = $0
  progress = (RT == "\r")
  if (line ~ /^[[:space:]]*$/) next
  if (brief && (progress || line !~ /^\[[0-9]+\/[0-9]+\]|^###|context: |^  (asr|subtitles|asr recheck|clean subtitles|jellyfin|plex):|^ +[0-9]+ +[0-9][0-9]:[0-9][0-9]:|file\(s\):|Traceback|[Ee]rror/)) next
  if (progress) {                                   # a progress update: overwrite in place
    # one status line at the bottom: the latest progress of each run, side by side
    sub(/^[[:space:]]+/, "", line)
    gsub(/  +/, " ", line)
    if (match(line, /time=[0-9:.]+/)) {                 # ffmpeg: just the position and speed
      t = substr(line, RSTART + 5, RLENGTH - 5); sub(/\.[0-9]+$/, "", t)
      sp = match(line, /speed= *[0-9.]+x/) ? substr(line, RSTART, RLENGTH) : ""
      gsub(/ /, "", sp)
      line = "encoding " t " " sp
    }
    prog[r + 0] = line
    showstatus(); fflush(); next
  }
  out = line
  if (line ~ /^\[[0-9]+\/[0-9]+\]/)        out = "\n" bold line off
  else if (line ~ /^###/)                  out = "\n" bold cyan line off
  else if (line ~ /context: MUTE/)         { sub(/^ *context: MUTE /, "", line); out = "  " red "MUTE" off " " line }
  else if (line ~ /context: keep/)         { sub(/^ *context: keep /, "", line); out = "  " green "keep" off " " line }
  else if (line ~ /Traceback|[Ee]rror|failed/) out = red line off
  else if (line ~ /^ +[0-9]+ +[0-9][0-9]:[0-9][0-9]:/) {
    n = split(line, f, /[[:space:]]+/); w = ""
    for (i = 8; i <= n; i++) w = w " " f[i]
    out = "  " yellow "muted" off " " f[3] "  " f[5] "s  " f[7] " " w
  }
  else if (line ~ /^  (asr|subtitles|asr recheck|clean subtitles|jellyfin|plex):|file\(s\):/) out = cyan line off
  if (out ~ /^\n/) { printf "%s\n", clr; out = substr(out, 2) }
  printf "%s%s%s\n", clr, tag, out
  # a finished step ends the progress of that run
  if (line ~ /^\[[0-9]+\/[0-9]+\]|staging: copied|transcribed|encoding clean|wrote |file\(s\)/) delete prog[r + 0]
  showstatus(); fflush()
}
function showstatus(   k, s, each, cnt) {
  cnt = 0; for (k in prog) cnt++
  if (!cnt) { printf "%s", clr; return }
  each = int(width / cnt) - 1; s = ""
  for (k = 1; k <= 4; k++) if (k in prog)
    s = s (s == "" ? "" : " | ") (multi ? tagcol[k] "[" k "]" off " " : "") dim substr(prog[k], 1, each - 5) off
  printf "%s%s", clr, s
}'
