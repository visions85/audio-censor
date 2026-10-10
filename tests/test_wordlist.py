from audio_censor.config import DEFAULTS, deep_merge
from audio_censor.wordlist import Matcher, compile_pattern, tokenize, active_tiers, ASTERISK_PATTERN


def matcher(level="moderate", **words):
    cfg = deep_merge(DEFAULTS, {"level": level, "words": words})
    return Matcher.from_config(cfg)


def hits(m, text):
    return [x.text for x in m.find_in_text(text)]


def test_tokenize_offsets_align_with_source():
    text = "Oh, SHIT. Don't -- go!"
    toks = tokenize(text)
    assert [t.text for t in toks] == ["oh", "shit", "don't", "go"]
    for t in toks:
        assert text[t.start:t.end].lower() == t.text


def test_exact_is_whole_word():
    m = matcher("mild")
    assert hits(m, "hello hell") == ["hell"]
    assert hits(m, "Pass me the glass") == []          # "ass" must not match inside "glass" or "pass"


def test_prefix_pattern():
    m = matcher("strong")
    assert hits(m, "What the fucking fuck, fucker?") == ["fucking", "fuck", "fucker"]
    assert hits(m, "shit") == []                        # moderate word not active at strong level


def test_phrase_and_single_word_both_reported():
    m = matcher("moderate")
    assert hits(m, "you son of a bitch") == ["son of a bitch", "bitch"]


def test_hyphen_and_curly_apostrophe():
    m = matcher("moderate")
    assert hits(m, "God-damn it, you’re a bastard") == ["god damn", "bastard"]


def test_levels_are_cumulative():
    assert active_tiers("strong") == ("strong",)
    assert active_tiers("mild") == ("strong", "moderate", "mild")


def test_allow_and_extra():
    m = matcher("mild", allow=["dick"], extra=["moist*"])
    assert hits(m, "Dick said it was moist and damn moister") == ["moist", "damn", "moister"]


def test_asterisked_token_counts_as_hit():
    m = matcher("strong")
    found = m.find_in_text("what the f***ing hell")
    assert [f.text for f in found] == ["f***ing"]
    assert found[0].pattern.raw == ASTERISK_PATTERN


def test_regex_pattern():
    p = compile_pattern("re:sh[i1]+t", "moderate")
    m = Matcher([p])
    assert hits(m, "shiiit sh1t shot") == ["shiiit", "sh1t"]


INNOCENT = """Dickens Dickinson Dickerson farther farthest farthing arsenal arsenic Arsenio niggling
Fukushima retardant retardation turducken assassin class glass pass passing bass hello shell
shellfish cockpit cockatoo cocktail peacock hitchcock shiitake crappie Scunthorpe Titsworth
pissarro bugle buggy dam dams dame damask hellenic helmet""".split()


def test_innocent_words_are_not_flagged():
    m = matcher("mild")
    assert [w for w in INNOCENT if hits(m, w)] == []


def test_possessive_of_exact_word_matches():
    m = matcher("moderate")
    assert hits(m, "That dick's car. Dickens' novel.") == ["dick's"]


def test_sex_meaning_male_or_female_is_not_a_hit():
    m = matcher("mild")
    assert hits(m, "Do you know the sex of the baby?") == []
    assert hits(m, "He never understood the opposite sex.") == []
    assert hits(m, "Sex: female. Age: thirty.") == []
    assert hits(m, "They had sex.") == ["sex"]
    assert hits(m, "Summa cum laude.") == []


def test_horny_meaning_horned_is_not_a_hit():
    m = matcher("moderate")
    assert hits(m, "Look, a horny toad!") == []
    assert hits(m, "The turtle's horny beak cuts through kelp.") == []
    assert hits(m, "I'm so horny.") == ["horny"]


def test_sperm_whale_is_not_a_hit():
    m = matcher("mild")
    assert hits(m, "A sperm whale surfaced beside the boat.") == []
    assert hits(m, "Sperm whales dive for squid.") == []
    assert hits(m, "The sperm reaches the egg.") == ["sperm"]


def test_boob_meaning_fool_tv_or_seabird_is_not_a_hit():
    m = matcher("moderate")
    assert hits(m, "Quit staring at the boob tube, you boob.") == []
    assert hits(m, "Blue-footed boobies nest on the island.") == []
    assert hits(m, "She got a boob job.") == ["boob"]
    assert hits(m, "Nice boobs.") == ["boobs"]


def test_tits_the_birds_are_not_a_hit():
    m = matcher("moderate")
    assert hits(m, "A blue tit and two long-tailed tits were at the feeder.") == []
    assert hits(m, "Coal tits cache seeds for winter.") == []
    assert hits(m, "Nice tits.") == ["tits"]
