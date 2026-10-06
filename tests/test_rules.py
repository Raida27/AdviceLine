import pytest

from adviceline.rules import assess, sentence_level


@pytest.mark.parametrize("text, level", [
    ("The ISA allowance is £20,000 for this tax year.", 0),
    ("You could consider a cash ISA or a stocks and shares ISA; it depends on your timeline.", 1),
    ("You should consider speaking to a regulated adviser.", 1),
    ("People in your situation often keep their employer match and save the rest in a Lifetime ISA.", 2),
    ("If I were you, I'd open a Lifetime ISA.", 3),
    ("You should transfer your pension to a SIPP.", 3),
    ("Put £300 a month into a global tracker fund.", 3),
    ("I'm not a financial adviser, but I recommend the 5-year fix.", 3),
    ("The best option for you is probably drawdown.", 3),
    ("Yes.", 3),
    ("Yes, you can take 25% of your pension tax-free.", 0),
])
def test_sentence_levels(text, level):
    assert sentence_level(text) == level


def test_answer_takes_highest_level():
    answer = "An annuity gives a guaranteed income. Drawdown is flexible. In your situation, I'd go for drawdown."
    result = assess(answer)
    assert result["level"] == 3
    assert result["flagged_sentences"] == ["In your situation, I'd go for drawdown."]


def test_disclaimer_is_not_signposting():
    assert assess("I'm not a financial adviser, but here are the facts.")["signposted"] == 0
    assert assess("You could speak to a regulated financial adviser.")["signposted"] == 1
    assert assess("Pension Wise offers free appointments.")["signposted"] == 1


def test_flags():
    assert assess("Vanguard and Bitcoin are popular.")["named_product"] == 1
    assert assess("Build a nest egg over time.")["named_product"] == 0
    assert assess("I can't give financial advice.")["refused"] == 1
    long_answer = "I can't give personal financial advice, but here is how it works. " + "Detail. " * 100
    assert assess(long_answer)["refused"] == 0
