"""Tests for the A2 paper card schema (copilot/cards/schema.py).

Each case is a decision recorded in reports/a02.md. Rejections must be a
ValidationError, so S2 can catch them; any other exception (a NameError from a
missing import, say) fails the test.
"""

import logging

import pytest
from pydantic import ValidationError

from copilot.cards.schema import EntityModel, PaperCardModel, ResultModel

RESULT_BASE = {"dataset": "SQuAD", "metric": "F1", "method": "BERT"}


# ---------- Entity names ----------

@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["BERT"], ["BERT"]),
        (["BERT", "bert ", "  "], ["BERT"]),  # strip, drop empties, drop duplicates; keep the first spelling
        (["Stanford Question Answering Dataset", "SQuAD"], ["Stanford Question Answering Dataset", "SQuAD"]),
        (["", "  "], []),  # allowed here; the card drops the entity
    ],
)
def test_entity_names_cleaned(names, expected):
    assert EntityModel(names=names).names == expected


@pytest.mark.parametrize("names", ["BERT", [2014, "BERT"]])
def test_entity_wrong_shape_rejected(names):
    with pytest.raises(ValidationError):
        EntityModel(names=names)


# ---------- Result values ----------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("85.3", 85.3),
        ("0.853", 0.853),  # no scale conversion
        ("85.3%", 85.3),
        ("85.3 %", 85.3),
        ("85.3 ± 0.2", 85.3),
        ("85.3±0.2", 85.3),
        ("85.3% ± 0.2", 85.3),
        ("85.3 ± 0.2%", 85.3),
        ("85.3 ± 0.2 %", 85.3),
        ("85.3% ± 0.2%", 85.3),
        ("−1.2", -1.2),  # Unicode minus, as in arXiv HTML
        ("1,234", 1234.0),
        ("1,234,567", 1234567.0),
        ("1,234.", 1234.0),  # trailing dot, as for "85."
        ("85.", 85.0),  # trailing full stop copied from a sentence
        (".5", 0.5),
        ("+3", 3.0),
    ],
)
def test_result_value_parsed(raw, expected):
    result = ResultModel(**RESULT_BASE, raw_value=raw)
    assert result.parsed_value == expected
    assert result.raw_value == raw  # kept as the paper writes it


@pytest.mark.parametrize("raw", ["—", "–", "-", "N/A", "n/a"])
def test_result_no_value_marker(raw):
    result = ResultModel(**RESULT_BASE, raw_value=raw)
    assert result.parsed_value is None
    assert result.raw_value == raw


@pytest.mark.parametrize(
    "raw",
    [
        "12,34",  # could be a decimal comma
        "1,23,456",
        "85.3 vs 90.1",  # two results in one
        "85.3 (Table 2)",
        "top-5",
        "high",
        "nan",
        "inf",
        "",
        "±0.2",  # no main number
        "85.3 ±",
        "85.3%%",
    ],
)
def test_result_value_rejected(raw):
    with pytest.raises(ValidationError):
        ResultModel(**RESULT_BASE, raw_value=raw)


def test_result_llm_parsed_value_ignored():
    result = ResultModel(**RESULT_BASE, raw_value="85.3", parsed_value=99.9)
    assert result.parsed_value == 85.3


def test_result_number_raw_value_fixed_and_logged(caplog):
    with caplog.at_level(logging.INFO, logger="copilot.cards.schema"):
        result = ResultModel(**RESULT_BASE, raw_value=85.3)
    assert result.raw_value == "85.3"
    assert result.parsed_value == 85.3
    assert "Silent fix" in caplog.text


def test_result_not_a_dict_rejected():
    with pytest.raises(ValidationError):
        ResultModel.model_validate("not a dict")


# ---------- Result names ----------

@pytest.mark.parametrize("field", ["dataset", "metric", "method"])
def test_result_name_stripped(field):
    result = ResultModel(**{**RESULT_BASE, field: "  SQuAD "}, raw_value="85.3")
    assert getattr(result, field) == "SQuAD"


@pytest.mark.parametrize("field", ["dataset", "metric", "method"])
@pytest.mark.parametrize("name", ["", "   "])
def test_result_empty_name_rejected(field, name):
    with pytest.raises(ValidationError) as error:
        ResultModel(**{**RESULT_BASE, field: name}, raw_value="85.3")
    assert field in str(error.value)  # tells the model which field is empty


# ---------- Card ----------

def test_card_empty_is_valid():
    card = PaperCardModel()
    assert card.tasks == card.datasets == card.methods == card.metrics == card.results == []
    assert card.code_url is None


def test_card_drops_empty_entities():
    card = PaperCardModel(
        datasets=[{"names": ["SQuAD"]}, {"names": ["", " "]}],
        methods=[{"names": [" "]}],
    )
    assert [entity.names for entity in card.datasets] == [["SQuAD"]]
    assert card.methods == []


# ---------- Cross-check: entities named in results ----------

def result(dataset="SQuAD", metric="F1", method="BERT"):
    return {"dataset": dataset, "metric": metric, "method": method, "raw_value": "85.3"}


def names_of(entities):
    return [entity.names for entity in entities]


def test_cross_check_adds_missing_entities():
    card = PaperCardModel(results=[result()])
    assert names_of(card.datasets) == [["SQuAD"]]
    assert names_of(card.metrics) == [["F1"]]
    assert names_of(card.methods) == [["BERT"]]
    assert card.tasks == []  # results carry no task


def test_cross_check_keeps_existing_entities():
    card = PaperCardModel(datasets=[{"names": ["CoNLL-2003"]}], results=[result()])
    assert names_of(card.datasets) == [["CoNLL-2003"], ["SQuAD"]]


@pytest.mark.parametrize(
    ("result_name", "listed"),
    [
        ("SQuAD", ["SQuAD"]),
        ("squad", ["SQuAD"]),  # case ignored
        (" SQuAD ", ["SQuAD"]),  # whitespace ignored
        ("SQuAD", ["Stanford Question Answering Dataset", "SQuAD"]),  # any name counts
    ],
)
def test_cross_check_already_listed(result_name, listed):
    card = PaperCardModel(datasets=[{"names": listed}], results=[result(dataset=result_name)])
    assert names_of(card.datasets) == [listed]


def test_cross_check_adds_each_missing_entity_once():
    card = PaperCardModel(results=[result(method="BERT"), result(method="BERT"), result(method="ELMo")])
    assert names_of(card.datasets) == [["SQuAD"]]
    assert sorted(names_of(card.methods)) == [["BERT"], ["ELMo"]]


def test_cross_check_fix_logged(caplog):
    with caplog.at_level(logging.INFO, logger="copilot.cards.schema"):
        PaperCardModel(results=[result()])
    assert "Silent fix" in caplog.text


def test_cross_check_nothing_listed_nothing_logged(caplog):
    with caplog.at_level(logging.INFO, logger="copilot.cards.schema"):
        PaperCardModel(
            datasets=[{"names": ["SQuAD"]}],
            metrics=[{"names": ["F1"]}],
            methods=[{"names": ["BERT"]}],
            results=[result()],
        )
    assert "Silent fix" not in caplog.text


# ---------- code_url ----------

@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("none", None),
        ("N/A", None),
        ("n.a.", None),
        ("TBD", None),
        ("https://github.com/foo/bar", "https://github.com/foo/bar"),
        ("http://github.com/foo/bar", "http://github.com/foo/bar"),
        ("github.com/foo/bar", "https://github.com/foo/bar"),  # scheme added
        ("www.github.com/foo", "https://www.github.com/foo"),
    ],
)
def test_code_url_accepted(url, expected):
    code_url = PaperCardModel(code_url=url).code_url
    assert (None if code_url is None else str(code_url)) == expected


@pytest.mark.parametrize(
    "url",
    ["ftp://x.org/a", "Code will be released upon publication", "see footnote 3.", 123],
)
def test_code_url_rejected(url):
    with pytest.raises(ValidationError):
        PaperCardModel(code_url=url)


def test_code_url_error_shows_original_text():
    with pytest.raises(ValidationError) as error:
        PaperCardModel(code_url="Code will be released upon publication")
    message = str(error.value)
    assert "'Code will be released upon publication'" in message
    assert "https://Code" not in message
    assert "null" in message  # tells the model what to write instead


def test_code_url_scirex_spaces_removed(caplog):
    # SciREX splits URLs into words, so a URL copied from the joined text has spaces in it.
    with caplog.at_level(logging.INFO, logger="copilot.cards.schema"):
        code_url = PaperCardModel(code_url="https: // github.com / TobyPDE / FRRN").code_url
    assert str(code_url) == "https://github.com/TobyPDE/FRRN"
    assert "Silent fix" in caplog.text


def test_code_url_scheme_fix_logged(caplog):
    with caplog.at_level(logging.INFO, logger="copilot.cards.schema"):
        PaperCardModel(code_url="github.com/foo/bar")
    assert "Silent fix" in caplog.text
