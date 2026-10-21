import httpx
import json


OLLAMA_ENDPOINT = 'http://localhost:11434/v1/chat/completions' 

data = {
   
   'model':'qwen3:8b',
   'messages':[
    {
        'role':'user',
        'content':'Hi good morning how are you!! just say thank you',
        
    }
   ],
   'reasoning_effort':'none',
   'temperature': 0,
   'stream':False

}

r = httpx.post(OLLAMA_ENDPOINT,json=data,timeout=30)


print(json.dumps(r.json(),indent=2))