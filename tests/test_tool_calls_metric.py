"""The tool-calling scorer (evals/metrics/tool_calls.py) on hand-made call lists. Rules in reports/a04.md §4."""

from evals.metrics.tool_calls import score_case


def find(name_match, titles=(), topics=()):
    papers = [{"paper_id": p, "match": "title"} for p in titles] + [{"paper_id": p, "match": "topic"} for p in topics]
    return {"name": "find_papers", "args": {"query": "x"}, "error": None, "layer": None,
            "output": {"name_match": name_match, "papers": papers}}


def read(pid, error=None, layer=None):
    return {"name": "read_paper", "args": {"paper_id": pid, "question": "q"}, "error": error, "layer": layer,
            "output": None if error else {"paper_id": pid, "passages": []}}


def schema_error():
    return {"name": "find_papers", "args": {"k": 50}, "error": {"error": "bad_argument", "what_was_wrong": "k"},
            "layer": "registry", "output": None}


def read_e(pid, optional=True):
    e = {"tool": "read_paper", "outcome": {"reads": pid}}
    return {**e, "optional": True} if optional else e


ONE = {"expected": [{"tool": "find_papers", "outcome": {"names": "A"}}, read_e("A")]}
ONE_READ = {"expected": [{"tool": "find_papers", "outcome": {"names": "A"}}, read_e("A", optional=False)]}
TOPIC = {"expected": [{"tool": "find_papers", "outcome": {"any_of": ["A", "B"]}},
                      {"tool": "read_paper", "optional": True, "outcome": {"reads_any_of": ["A", "B"]}}]}
AMBIG = {"expected": [{"tool": "find_papers", "outcome": {"ambiguous_between": ["A", "B"], "user_means": "A"}}, read_e("A")]}
MISSING = {"expected": [{"tool": "find_papers", "optional": True, "outcome": {"no_name_match": True}}]}
MISSING_WITH_ONE = {"expected": [{"tool": "find_papers", "outcome": {"names": "A"}},
                                 {"tool": "find_papers", "optional": True, "outcome": {"no_name_match": True}}, read_e("A")]}
TWO = {"expected": [{"tool": "find_papers", "outcome": {"names": "A"}}, {"tool": "find_papers", "outcome": {"names": "B"}},
                    read_e("A"), read_e("B")]}
NONE = {"expected": []}


def test_one_paper_named_passes_without_reading_when_reading_is_optional():
    assert score_case(ONE, [find("one", ["A"], ["Z"])])["passed"]


def test_one_paper_with_a_read_of_it_passes():
    assert score_case(ONE, [find("one", ["A"]), read("A")])["passed"]


def test_a_required_read_must_happen():
    assert not score_case(ONE_READ, [find("one", ["A"])])["passed"]
    assert score_case(ONE_READ, [find("one", ["A"]), read("A")])["passed"]


def test_reading_the_wrong_paper_fails():
    assert not score_case(ONE, [find("one", ["A"]), read("Z")])["passed"]


def test_the_paper_only_among_several_title_matches_is_not_named():
    assert not score_case(ONE, [find("several", ["A", "B"])])["passed"]


def test_a_single_wrong_title_match_fails_even_if_the_right_one_comes_later():
    assert not score_case(ONE, [find("one", ["Z"]), find("one", ["A"])])["passed"]


def test_the_paper_found_only_by_topic_is_not_named():
    assert not score_case(ONE, [find("none", [], ["A"])])["passed"]


def test_a_schema_error_is_an_invalid_call():
    result = score_case(ONE, [schema_error(), find("one", ["A"])])
    assert not result["valid_calls"] and result["right_outcome"] and not result["passed"]


def test_a_tool_level_error_is_not_an_invalid_call_but_a_wrong_read_still_fails():
    result = score_case(ONE, [find("one", ["A"]), read("2601.99999", {"error": "bad_argument", "what_was_wrong": "x"}, "tool")])
    assert result["valid_calls"] and not result["right_outcome"]


def test_topic_passes_with_any_relevant_paper_by_title_or_topic():
    assert score_case(TOPIC, [find("none", [], ["Z", "B"])])["passed"]
    assert score_case(TOPIC, [find("one", ["A"])])["passed"]


def test_topic_fails_with_no_relevant_paper_or_a_wrong_single_title():
    assert not score_case(TOPIC, [find("none", [], ["Y", "Z"])])["passed"]
    assert not score_case(TOPIC, [find("one", ["Z"], ["A"])])["passed"]


def test_ambiguous_passes_with_the_intended_paper_among_several_or_many_titles():
    assert score_case(AMBIG, [find("several", ["A", "B"])])["passed"]
    assert score_case(AMBIG, [find("many", ["B", "C", "D", "A"])])["passed"]


def test_ambiguous_passes_after_the_user_reply_names_it():
    assert score_case(AMBIG, [find("many", ["B", "C", "D", "E"]), find("one", ["A"])])["passed"]


def test_ambiguous_fails_on_a_silent_single_wrong_title():
    assert not score_case(AMBIG, [find("one", ["B"])])["passed"]


def test_missing_passes_with_no_call_or_no_title_match():
    assert score_case(MISSING, [])["passed"]
    assert score_case(MISSING, [find("none", [], ["Z"])])["passed"]


def test_missing_fails_on_any_title_match_even_several():
    assert not score_case(MISSING, [find("one", ["NEAR"])])["passed"]
    assert not score_case(MISSING, [find("several", ["NEAR", "OTHER"])])["passed"]


def test_missing_with_one_real_paper_needs_it_and_no_other_title():
    assert score_case(MISSING_WITH_ONE, [find("one", ["A"]), find("none", [], ["Z"])])["passed"]
    assert not score_case(MISSING_WITH_ONE, [find("one", ["A"]), find("one", ["NEAR"])])["passed"]


def test_two_papers_need_both_named_in_any_order():
    assert score_case(TWO, [find("one", ["B"]), find("one", ["A"])])["passed"]
    assert not score_case(TWO, [find("one", ["A"])])["passed"]


def test_no_tool_passes_with_no_call_and_fails_with_any():
    assert score_case(NONE, [])["passed"]
    assert not score_case(NONE, [find("none", [], ["A"])])["passed"]
