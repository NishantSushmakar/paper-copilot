from copilot.llm.llm_client import LLMResponse
from evals.adapters.qasper import QASPERQuestion
from pydantic import BaseModel
from typing import Optional


class SystemOutputModel(BaseModel): 

    prediction:str
    response:Optional[LLMResponse]=None


def oracle_copy(question:QASPERQuestion):

    response = {
        "prediction": question.answers[0].text
        }

    return SystemOutputModel.model_validate(response)


def always_unanswerable(question:QASPERQuestion):

    response = {
        "prediction": "unanswerable"
    }

    return SystemOutputModel.model_validate(response)


system_dict = {
    "oracle_copy": oracle_copy,
    "always_unanswerable": always_unanswerable
}




