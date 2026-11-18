"""Score one tool-calling case (evals/data/tool_calls_cases.jsonl) from the calls a model made.

Rules (reports/a04.md §4), for the find + read tools:
* valid calls: every call passed the registry's checks (a known tool, arguments that fit the
  schema strictly). A tool's own error, like an unknown paper_id, is an outcome, not an invalid call.
* right outcome, judged on what the calls returned, never on the argument text:
  - names X: some find_papers call matched exactly one title, and it was X;
  - any_of [...]: some find_papers call returned at least one of these papers (title or topic);
  - ambiguous_between + user_means X: X was among the title matches of a find_papers call that
    matched several or many titles, or was matched alone (after the scripted user's reply);
  - no_name_match (optional): no find_papers call matched a title the case doesn't expect;
  - reads X: some read_paper call read X; optional when the abstract already answers;
  - a call to a tool the case doesn't expect fails it, so any call fails a no-tool case;
  - a single title match for a paper the case doesn't expect fails it (a silent wrong pick);
  - reading a paper the case doesn't expect fails it.
* passed: valid calls and right outcome.

A call is a dict {"name", "args", "error" (the structured error or None), "layer" (where an
error came from: "registry" or "tool"), "output" (the parsed tool result, or None)}.
"""


def finds(calls: list[dict]) -> list[dict]:
    return [c["output"] for c in calls if c["name"] == "find_papers" and isinstance(c["output"], dict)]


def title_matched(found: dict) -> list[str]:
    return [p["paper_id"] for p in found["papers"] if p["match"] == "title"]


def named_alone(calls: list[dict]) -> list[str]:
    """Papers a find_papers call matched as the one title match."""
    return [title_matched(f)[0] for f in finds(calls) if f["name_match"] == "one"]


def reads(calls: list[dict]) -> list[str]:
    """Paper IDs a read_paper call was asked for (without a version suffix)."""
    return [str(c["args"].get("paper_id", "")).split("v")[0] for c in calls if c["name"] == "read_paper"]


def score_case(case: dict, calls: list[dict]) -> dict:
    reasons = []

    invalid = [c for c in calls if c["error"] is not None and c["layer"] == "registry"]
    valid = not invalid
    for c in invalid:
        reasons.append(f"invalid call {c['name']}: {c['error']['what_was_wrong']}")

    expected = case["expected"]
    outcome = True

    unexpected = [c["name"] for c in calls if c["name"] not in {e["tool"] for e in expected}]
    if unexpected:
        outcome = False
        reasons.append(f"called {sorted(set(unexpected))}, which this case doesn't expect")

    meant, readable = set(), set()
    for e in expected:
        o = e["outcome"]
        meant |= {o[key] for key in ("names", "user_means") if key in o}
        readable |= {o["reads"]} if "reads" in o else set(o.get("reads_any_of", []))
    if any("any_of" in e["outcome"] for e in expected):
        meant |= {pid for e in expected for pid in e["outcome"].get("any_of", [])}

    wrong = [pid for pid in named_alone(calls) if pid not in meant]
    if any(e["outcome"].get("no_name_match") for e in expected):
        wrong += [pid for f in finds(calls) for pid in title_matched(f) if pid not in meant]
    if wrong:
        outcome = False
        reasons.append(f"matched the title of {sorted(set(wrong))}, which isn't the paper meant")

    wrong_reads = [pid for pid in reads(calls) if pid not in readable]
    if wrong_reads:
        outcome = False
        reasons.append(f"read {sorted(set(wrong_reads))}, which this case doesn't expect")

    found_any = {p["paper_id"] for f in finds(calls) for p in f["papers"]}
    for e in expected:
        o = e["outcome"]
        if "names" in o and o["names"] not in named_alone(calls):
            outcome = False
            reasons.append(f"never matched {o['names']} as the one paper")
        if "any_of" in o and not (found_any & set(o["any_of"])):
            outcome = False
            reasons.append("no relevant paper among the results")
        if "user_means" in o:
            reached = o["user_means"] in named_alone(calls) or any(
                o["user_means"] in title_matched(f) for f in finds(calls) if f["name_match"] in ("several", "many"))
            if not reached:
                outcome = False
                reasons.append(f"never reached the intended paper {o['user_means']}")
        if "reads" in o and not e.get("optional") and o["reads"] not in reads(calls):
            outcome = False
            reasons.append(f"never read {o['reads']}")

    return {"valid_calls": valid, "right_outcome": outcome, "passed": valid and outcome, "reasons": reasons}
