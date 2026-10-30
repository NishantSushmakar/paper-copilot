from pydantic import BaseModel, model_validator, Field ,AwareDatetime, ConfigDict
from typing import Literal, List


class ArxivEventModel(BaseModel):
    model_config = ConfigDict(strict=True)
    
    event_type: Literal["paper_published","paper_updated"]
    source: Literal["arxiv"]
    id:str 
    version:int=Field(gt=0)
    title: str
    abstract:str 
    authors: List[str]= Field(min_length=1)
    categories: List[str]
    published: AwareDatetime
    updated: AwareDatetime
    pdf_url: str
    polled_at: AwareDatetime

    @model_validator(mode="after")
    def check_disagreement(self):

        if (self.event_type=="paper_published") and (self.version>1):
            raise ValueError(f"Published Paper cannot have Version:{self.version}")
        elif(self.event_type=="paper_updated") and (self.version==1):
            raise ValueError("Updated Paper cannot have Version:1")



        return self


    
