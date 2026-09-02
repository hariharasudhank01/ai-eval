"""
This Script focus only on Ollama Models
"""
import ollama
from pydantic import BaseModel


SYSTEM_PROMPT = """
Role:
- Synthetic data generater
Task:
- Generate Synthetic data based on the data provided by the user
Action:
- Make sure you generate similar data to what the iser has given and not answer the question as part of Synthetic data
- Make sure follow the use template, if it has more cases match the equal no of cases
- Make sure you have every out put in list with no specila character just space so that it can be easily split using " " or \n
- Make sure you keep the data short within 2 sentence and not more than 70 words.
- Make sure you dont use any personal information reference like name, age, address etc. There is a PII check running in the pipeline to flag it
- Make sure you dont deviate from the core business problem that data is addressing
- Make sure to understand the data, relate it with the industry it is trying to connect and generate data accordingly
- Try to understand the possible cases covered and try ot generate data similar to that, make sure you're not biased and miss less importane data.
"""

class Response(BaseModel):
    created_at: str
    content: str

def run(user_prompt, selected_model):
    response = ollama.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        format=Response.model_json_schema(),
        options={
            'think': False,
            'num_predict': -1
        }
    )

    resp = Response.model_validate_json(response.message.content)
    return resp

