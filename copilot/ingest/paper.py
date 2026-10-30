from typing import Optional,List,Literal

from pydantic import BaseModel,AwareDatetime

class SectionModel(BaseModel):

    heading :Optional[str]=None
    paragraphs: List[str]

class PaperModel(BaseModel):
    source: Literal["qasper","arxiv"]
    id : str
    version: int 
    title : str 
    authors : Optional[List[str]]=None
    abstract: str
    sections: List[SectionModel]
    fetched_at: AwareDatetime

