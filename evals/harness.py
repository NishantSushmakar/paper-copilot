from pydantic import BaseModel
from typing import Optional
from evals.systems.qasper import system_dict,SystemOutputModel
from evals.adapters.qasper import QASPERQuestion
from evals.metrics.qasper_f1 import compare_scores
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from langsmith import traceable, get_current_run_tree
from copilot.infra.jsonl import append_jsonl


class IdentityModel(BaseModel):
    question_id:str
    question_type : str
    system_name:str 

class OutcomeModel(BaseModel): 
    f1_score: float
    prediction: str

class FailureModel(BaseModel): 
    failure_code: str
    failure_message:str 

class ResponseModel(BaseModel):
    latency_ms: float
    thinking: Optional[str]
    prompt_tokens: int 
    completion_tokens: int
    finish_reason : str
    cache_hit : bool


class ResultModel(BaseModel):

    identity: IdentityModel
    outcome: OutcomeModel
    response: Optional[ResponseModel]=None
    failure: Optional[FailureModel]=None

def get_input(input:dict):

    question_text = input["question"].question
    
    answer_list = []
    for answer in input["question"].answers : 
        answer_list.append(answer.text)

    return {
        "question_text": question_text,
        "gold_answers": answer_list
    }


def get_output(output):

    
    failure = None

    if output.failure is not None:
        failure = {"failure_code":output.failure.failure_code,
                   "failure_message":output.failure.failure_message}

    response = {
        "prediction": output.outcome.prediction,
        "f1": output.outcome.f1_score,
        "failure": failure
    }

    return response 

@traceable(run_type="chain",name="Q/A System",process_inputs=get_input,process_outputs=get_output)
def run_one_question(system_name:str,question:QASPERQuestion): 

    system = system_dict[system_name]
    error_response = None
    f1_score = 0 
    gold_set_list = [a.text for a in question.answers]
    system_response = SystemOutputModel.model_validate({
        "prediction":"",
        "response": None
    })

    run = get_current_run_tree()

    if run is not None: 
        run.add_metadata({
            "question_id": question.question_id,
            "question_type": question.question_type,
            "system_name": system_name
        })

    try: 
        system_response = system(question=question)
        f1_score = compare_scores(system_response.prediction,gold_set_list)
    
    except Exception as e:
        error_msg = str(e)
        error_code = type(e).__name__

        error_response = {
            "failure_code":error_code,
            "failure_message":error_msg
        }
        
       
    

    llm_response = None

    if system_response.response is not None: 
        llm_response = {
            "latency_ms" : system_response.response.latency.total_seconds() * 1000,
            "thinking": system_response.response.thinking,
            "prompt_tokens":system_response.response.prompt_tokens,
            "completion_tokens": system_response.response.completion_tokens,
            "finish_reason": system_response.response.finish_reason,
            "cache_hit": system_response.response.cache_hit   
        }


    result = {
        "identity": {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "system_name": system_name 
        },
        "outcome": {
            "f1_score": f1_score,
            "prediction": system_response.prediction
        },
        "response": llm_response,
        "failure": error_response
    }
    
    return ResultModel.model_validate(result)


def run_all(system_name: str, questions: list[QASPERQuestion], results_path: Path) -> list[ResultModel]:
    """Run one system over every question, writing each row to disk as soon as it is done."""
    results_path = Path(results_path)
    if results_path.exists():
        # Appending to an old file would mix two runs; each run gets a fresh path.
        raise FileExistsError(f"{results_path} already exists")

    rows = []
    for question in questions:
        row = run_one_question(system_name, question)
        append_jsonl(results_path, row)
        rows.append(row)
    return rows


def _rate(count: int, total: int) -> Optional[float]:
    # None, not 0.0, when nothing was measured (fake systems have no LLM responses).
    return count / total if total else None


def summarise(rows: list[ResultModel]) -> dict:
    f1_by_type = defaultdict(list)
    for row in rows:
        f1_by_type[row.identity.question_type].append(row.outcome.f1_score)

    failures = [row.failure for row in rows if row.failure is not None]
    responses = [row.response for row in rows if row.response is not None]

    return {
        "n": len(rows),
        "mean_f1": mean(row.outcome.f1_score for row in rows) if rows else None,
        "by_type": {
            question_type: {"n": len(scores), "mean_f1": mean(scores)}
            for question_type, scores in sorted(f1_by_type.items())
        },
        "errors": len(failures),
        "error_types": dict(Counter(f.failure_code for f in failures)),
        "n_with_response": len(responses),
        "truncation_rate": _rate(sum(r.finish_reason == "length" for r in responses), len(responses)),
        "cache_hit_rate": _rate(sum(r.cache_hit for r in responses), len(responses)),
        "mean_latency_ms": mean(r.latency_ms for r in responses) if responses else None,
        "total_prompt_tokens": sum(r.prompt_tokens for r in responses) if responses else None,
        "total_completion_tokens": sum(r.completion_tokens for r in responses) if responses else None,
    }


