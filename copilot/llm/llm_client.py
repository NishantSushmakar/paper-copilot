import httpx
from copilot.llm.slots import get_chat_slot,config
from pydantic import BaseModel
from datetime import timedelta
from typing import Optional


class LLMResponse(BaseModel):
    text: str
    thinking: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    model : str
    latency: timedelta
    cache_hit: bool 


def chat_with_llm(slot:str,messages:list,temperature:float,max_tokens:int,thinking:bool)->LLMResponse:

    model_config = get_chat_slot(config=config,name=slot)

    data = {
        'model':model_config.model,
        'messages': messages,
        'reasoning_effort': 'none' if not thinking else 'high',
        'temperature': temperature,
        'stream': False,
        'max_tokens': max_tokens
    }

    ENDPOINT_URL = str(model_config.base_url) + '/chat/completions'
        
    response = httpx.post(ENDPOINT_URL,json=data,timeout=120)

    if not response.is_success: 
        raise RuntimeError(f"API Request failed {response.text} with code {response.status_code}")
        

    results = response.json()

    output = {
        'text' : results['choices'][0]['message']['content'],
        'thinking': results['choices'][0]['message'].get('reasoning'),
        'prompt_tokens':results['usage']['prompt_tokens'],
        'completion_tokens':results['usage']['completion_tokens'],
        'finish_reason': results['choices'][0]['finish_reason'],
        'model': model_config.model,
        'latency': response.elapsed.total_seconds(),
        'cache_hit': False

    }

    return LLMResponse.model_validate(output)

     

if __name__ == "__main__":


    messages=[{'role':'user','content':'tell me how many r in strawberry'}]

    print(chat_with_llm(slot='MED',messages=messages,temperature=0,max_tokens=5,thinking=False))




