import json

from audio_censor import context
from audio_censor.config import deep_merge, load_config
from audio_censor.subtitles import censor_text
from audio_censor.wordlist import Matcher, tokenize

RULE = {"words": ["balls", "nuts"], "meaning": "testicles", "fallback": ["bust* * balls", "in the nuts"]}


def _matcher(monkeypatch, answers=None, down=False):
    cfg = deep_merge(load_config(), {"context": {"rules": [RULE]}})
    asked = []

    def fake_ask(self, text, meaning):
        asked.append(text)
        if down:
            return None
        return any(k in text for k in (answers or []))

    monkeypatch.setattr(context.ContextJudge, "_ask", fake_ask)
    return Matcher.from_config(cfg), asked


def test_only_the_rude_sense_is_censored(monkeypatch):
    m, asked = _matcher(monkeypatch, answers=["busted my [[balls]]"])
    lines = ["He's busted my balls all week.", "Where are my golf balls?"]
    m.prepare(lines)
    assert [x.text for x in m.find(tokenize(lines[0]))] == ["balls"]
    assert m.find(tokenize(lines[1])) == []
    assert ">>> where are my golf [[balls]]" in asked[1]   # the neighbouring line is sent too
    assert "busted my balls" in asked[1]


def test_answers_are_cached_and_subtitles_ask_the_same_question(monkeypatch):
    m, asked = _matcher(monkeypatch, answers=["[[nuts]] intact"])
    lines = ["Get off my porch", "with your nuts intact."]
    m.prepare(lines)
    assert m.find(tokenize(lines[1]))
    text, n = censor_text(lines[1], m)          # same line again, as the clean subtitles see it
    assert text == "with your **** intact." and n == 1
    assert len(asked) == 1
    cached = json.loads(context._cache_path().read_text())
    assert list(cached.values()) == [True]


def test_fallback_phrases_when_the_model_is_unavailable(monkeypatch):
    m, _ = _matcher(monkeypatch, down=True)
    for line, expect in (("you busted my balls", ["balls"]), ("kicked him in the nuts", ["nuts"]),
                         ("are you nuts", [])):
        m.prepare([line])
        assert [x.text for x in m.find(tokenize(line))] == expect


def test_a_correction_overrides_the_ai_and_teaches_it(monkeypatch, capsys):
    from audio_censor.cli import main
    from audio_censor.config import default_config_path
    default_config_path().parent.mkdir(parents=True, exist_ok=True)
    default_config_path().write_text('[[context.rules]]\nwords = ["balls", "nuts"]\nmeaning = "testicles"\n')
    m, asked = _matcher(monkeypatch, answers=["[[balls]]"])        # the "AI" mutes every balls
    line = "Throw me some balls."
    m.prepare([line])
    assert m.find(tokenize(line))
    assert main(["correct", "throw me some [[balls]]", "--keep", "--note", "Sandlot"]) == 0
    assert main(["correct", "kicked in the nuts", "--mute"]) == 0           # word found automatically
    assert main(["corrections"]) == 0
    out = capsys.readouterr().out
    assert "keep  throw me some [[balls]]  (Sandlot)" in out and "mute  kicked in the [[nuts]]" in out
    m, asked = _matcher(monkeypatch, answers=["[[balls]]"])
    m.prepare(["Hey, throw me some balls, Smalls."])
    assert m.find(tokenize("Hey, throw me some balls, Smalls.")) == []     # the correction wins, no question
    assert asked == []
    m.prepare(["Grab some balls."])
    m.find(tokenize("Grab some balls."))
    assert "throw me some [[balls]]\nAnswer: no" in m.judge.prompt(m.rules[0])   # and is shown as an example
    assert main(["corrections", "--remove", "1"]) == 0
    assert "throw me" not in capsys.readouterr().out


def test_sound_captions_are_not_judged(monkeypatch):
    m, asked = _matcher(monkeypatch, answers=["[[balls]]"])
    lines = ["[BALLS BOUNCING]", "Watch your balls!"]
    m.prepare(lines)
    assert m.find(tokenize(lines[0])) == [] and asked == []
    assert m.find(tokenize(lines[1]))


def test_a_mute_correction_beats_a_keep(monkeypatch):
    from audio_censor import context
    context.save_corrections([{"phrase": "[[come]] in her", "mute": True}, {"phrase": "wants to [[come]] in", "mute": False}])
    m, asked = _matcher(monkeypatch, answers=[])
    m.rules[0].patterns.append(__import__("audio_censor.wordlist", fromlist=["x"]).compile_pattern("come", "moderate"))
    for line, muted in (("he wants to come in", False), ("he wants to come in her", True), ("come in her", True)):
        m.prepare([line])
        assert bool(m.find(tokenize(line))) is muted, line
    assert asked == []


def test_a_line_anchored_correction(monkeypatch):
    from audio_censor import context
    context.save_corrections([{"phrase": "^[[nuts]]$", "mute": False}])
    m, asked = _matcher(monkeypatch, answers=["[[nuts]]"])          # the "AI" mutes every nuts
    for line, muted in (("Nuts.", False), ("Nuts!", False), ("Kicked in the nuts.", True), ("Nuts to you", True)):
        m.prepare([line])
        assert bool(m.find(tokenize(line))) is muted, line


def test_rule_settings_reach_the_right_fields():
    cfg = deep_merge(load_config(), {"context": {"rules": [
        {"words": ["nuts"], "meaning": "testicles", "innocent": "food", "when_unsure": "keep", "ask_twice": True},
        {"words": ["prick"], "meaning": "insult", "innocent": "a needle", "when_unsure": "mute"}]}})
    a, b = Matcher.from_config(cfg).rules
    assert (a.meaning, a.innocent, a.unsure_mute, a.ask_twice) == ("testicles", "food", False, True)
    assert (b.innocent, b.unsure_mute, b.ask_twice) == ("a needle", True, False)
