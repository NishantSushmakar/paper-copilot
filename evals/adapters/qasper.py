from pydantic import BaseModel, Field
from typing import Literal, Optional
from pathlib import Path
import json


q_type = Literal["extractive", "abstractive", "yes_no", "unanswerable"]

class Answer(BaseModel):
    answer_type : q_type
    text : str = Field(min_length=1)
    evidence : list[str]
    annotation_id : str

class Section(BaseModel): 
    section_name : Optional[str]
    paragraphs: list[str] 

class QASPERQuestion(BaseModel):
    question_id : str
    paper_id : str
    question: str
    title:str
    abstract : str
    question_type : q_type
    answers: list[Answer] = Field(min_length=1)
    paper_text : list[Section] 



def load_raw_answer(raw_answer, annotation_id):

    a_type = None 
    a_text = None 

    if len(raw_answer["extractive_spans"])>0: 
        a_type = "extractive"
        a_text = ", ".join(x for x in raw_answer["extractive_spans"])
    elif len(raw_answer["free_form_answer"])>0:
        a_type = "abstractive"
        a_text = raw_answer["free_form_answer"]
    elif raw_answer["yes_no"] is not None:
        a_type = "yes_no"
        a_text = "yes" if raw_answer["yes_no"] else "no"
    elif raw_answer["unanswerable"] : 
        a_type = "unanswerable"
        a_text = "unanswerable"



    answer = {

            "answer_type"  : a_type,
            "text" : a_text,
            "evidence": raw_answer["evidence"],
            "annotation_id" : annotation_id


    }

    return Answer.model_validate(answer)


def answer_type_choice(answer_list:list[Answer]):

    type_set = set()

    for answer in answer_list: 
        type_set.add(answer.answer_type)

    if "extractive" in type_set:
        return "extractive"
    elif "abstractive" in type_set:
        return "abstractive"
    elif "yes_no" in type_set:
        return "yes_no"
    else: 
        return "unanswerable"
    


def load_split(split_name:str): 

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    data_path = root_dir/"data/qasper"/f"qasper-{split_name}-v0.3.json"


    with open(data_path,'r') as f: 
        data = json.load(f) 

    question_list = []

    for key , value in data.items(): 
        section_list = []
        for sec in value["full_text"]: 
            section_list.append(Section.model_validate(sec))

        for question in value["qas"]: 
            answer_list = []
            for answer in question["answers"]: 

                    final_answer = load_raw_answer(raw_answer=answer["answer"],annotation_id=answer["annotation_id"])
                    answer_list.append(final_answer)

            
            answer_type = answer_type_choice(answer_list)

            question_model = {

                "question_id" : question["question_id"] ,
                "paper_id" : key,
                "question": question["question"],
                "title": value["title"],
                "abstract": value["abstract"],
                "question_type": answer_type,
                "answers": answer_list,
                "paper_text": section_list

                 }

            question_list.append(QASPERQuestion.model_validate(question_model))

    return question_list

