-- audio-censor.lua: beep, mute or duck profanity live in mpv, using the span file
-- written by `audio-censor scan` (<video>.censor.json beside the video).
--
-- Either launch through `audio-censor play Movie.mkv`, or copy this file to
-- ~/.config/mpv/scripts/ (`audio-censor install-mpv`) and it will act on any file
-- that has a span file next to it. Alt+c toggles censoring during playback.
--
-- Options (script-opts, prefix "audio-censor-"):
--   spans=PATH     span file (default: <video>.censor.json)
--   subs=PATH      clean subtitle file to add (default: *.clean*.srt beside the video)
--   mode=beep      beep | mute | duck
--   duck=0.1       dialogue level under a beep / in duck mode
--   frequency=1000 beep frequency (Hz)
--   volume=0.4     beep level, 0..1
--   pad=0          extra seconds either side of each span
--   osd=yes        show the censored words briefly on screen
--   enabled=yes    start enabled

local mp = require 'mp'
local utils = require 'mp.utils'
local msg = require 'mp.msg'
local options = require 'mp.options'

local o = {
    spans = "",
    subs = "",
    mode = "beep",
    duck = 0.1,
    frequency = 1000,
    volume = 0.4,
    pad = 0,
    osd = true,
    enabled = true,
}
options.read_options(o, "audio-censor")

local spans = {}          -- { {start, end, words}, ... } sorted by start
local applied = false
local current = nil       -- index of the span we are inside, for OSD

local function read_file(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local s = f:read("*a")
    f:close()
    return s
end

local function fmt(x)
    return string.format("%.3f", x)
end

local function video_path()
    local p = mp.get_property("path")
    if not p then return nil end
    if p:find("://") then return nil end          -- streams have no sidecars
    return p
end

local function stem_of(path)
    local dir, name = utils.split_path(path)
    return dir, (name:gsub("%.[^.]+$", ""))
end

local function load_spans()
    local path = o.spans
    local video = video_path()
    if path == "" then
        if not video then return nil end
        local dir, stem = stem_of(video)
        path = utils.join_path(dir, stem .. ".censor.json")
    end
    local text = read_file(path)
    if not text then return nil, path end
    local doc = utils.parse_json(text)
    if type(doc) ~= "table" or type(doc.spans) ~= "table" then return nil, path end
    local out = {}
    for _, s in ipairs(doc.spans) do
        local words = ""
        if type(s.words) == "table" then words = table.concat(s.words, ", ") end
        table.insert(out, { math.max(0, s.start - o.pad), s["end"] + o.pad, words })
    end
    table.sort(out, function(a, b) return a[1] < b[1] end)
    return out, path, doc
end

local function enable_expr()
    local parts = {}
    for _, s in ipairs(spans) do
        table.insert(parts, string.format("between(t,%s,%s)", fmt(s[1]), fmt(s[2])))
    end
    return table.concat(parts, "+")
end

-- One libavfilter graph; `t` is the stream timestamp, so seeking stays correct.
local function build_graph()
    local e = enable_expr()
    local level = (o.mode == "duck") and o.duck or 0
    local gate = string.format("volume=volume=%s:enable='%s'", fmt(level), e)
    if o.mode ~= "beep" then
        return gate
    end
    local beep = string.format("aeval=exprs='%s*sin(2*PI*%s*t)',volume=volume=0:enable='not(%s)'",
        fmt(o.volume), fmt(o.frequency), e)
    return string.format("asplit[d][b];[d]%s[dg];[b]%s[bp];[dg][bp]amix=inputs=2:normalize=0", gate, beep)
end

local function remove_filter()
    if applied then
        mp.commandv("af", "remove", "@censor")
        applied = false
    end
end

local function apply_filter()
    remove_filter()
    if not o.enabled or #spans == 0 then return end
    local graph = build_graph()
    -- %n% quoting keeps mpv's option parser away from the quotes and brackets inside.
    local ok = mp.commandv("af", "add", string.format("@censor:lavfi=graph=%%%d%%%s", #graph, graph))
    applied = ok ~= nil and ok ~= false
    if not applied then
        msg.error("could not add the censor filter; is this mpv built with libavfilter?")
    end
end

local function find_clean_subs()
    if o.subs ~= "" then return o.subs end
    local video = video_path()
    if not video then return nil end
    local dir, stem = stem_of(video)
    local files = utils.readdir(dir ~= "" and dir or ".", "files") or {}
    local best = nil
    local exts = { srt = true, ass = true, ssa = true, vtt = true }
    for _, name in ipairs(files) do
        local ext = name:match("%.([^.]+)$")
        if name:sub(1, #stem + 1) == stem .. "." and name:find("%.clean%.") and ext and exts[ext:lower()] then
            best = utils.join_path(dir, name)
            if name:find("%.en%.") or name:find("%.eng%.") then break end
        end
    end
    return best
end

local function add_subs()
    local subs = find_clean_subs()
    if subs and read_file(subs) then
        mp.commandv("sub-add", subs, "select", "Clean")
        msg.info("clean subtitles: " .. subs)
    end
end

local function on_file_loaded()
    current = nil
    local loaded, path, doc = load_spans()
    spans = loaded or {}
    if not loaded then
        if path then msg.info("no span file (" .. path .. "); playing uncensored") end
        remove_filter()
        return
    end
    if doc and doc.rendered_in_place then
        -- the file already carries the clean track as its default audio: no live filter needed
        msg.info("clean track rendered in place; not filtering")
        remove_filter()
        add_subs()
        return
    end
    msg.info(string.format("%d span(s) from %s, mode %s", #spans, path, o.mode))
    apply_filter()
    add_subs()
    if o.osd then
        mp.osd_message(string.format("audio-censor: %d word(s) censored (%s)", #spans, o.mode), 3)
    end
end

local function on_time(_, t)
    if not o.osd or not o.enabled or not t or #spans == 0 then return end
    for i, s in ipairs(spans) do
        if t >= s[1] and t <= s[2] then
            if current ~= i then
                current = i
                mp.osd_message("\226\151\143 " .. (s[3] ~= "" and s[3] or "censored"), math.max(0.8, s[2] - t))
            end
            return
        end
        if s[1] > t then break end
    end
    current = nil
end

local function toggle()
    o.enabled = not o.enabled
    apply_filter()
    mp.osd_message("audio-censor " .. (o.enabled and "on" or "off"), 2)
end

mp.register_event("file-loaded", on_file_loaded)
mp.register_event("end-file", remove_filter)
mp.observe_property("time-pos", "number", on_time)
mp.add_key_binding("Alt+c", "audio-censor-toggle", toggle)
