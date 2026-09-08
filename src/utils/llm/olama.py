"""
This Script focus only on Ollama Models
"""
import ollama
from pydantic import BaseModel


SYSTEM_PROMPT = """
Role:
- Synthetic data generator.
Task:
- Generate synthetic data based on the data the user provides, returning it as a
  strict JSON object matching the given schema.
Output format (follow exactly, no exceptions):
- Return your answer as a JSON array field called "items".
- Each element of "items" is exactly one generated case, as a single plain-text
  string with no line breaks inside it.
- Generate exactly {case_count} item(s) in "items" - not more, not fewer, no matter
  how many cases the input template appears to suggest.
- Never merge multiple cases into a single array element, and never split one case
  across multiple array elements.
- Do not repeat the same or a near-duplicate item more than once in "items".
- Do not include numbering, bullet markers, headers, labels, or any commentary -
  each element must contain only the generated case content itself.
Action:
- Generate data similar in style and intent to what the user has given; do not
  answer the input as if it were a question directed at you.
- Follow the structure/template of the user's cases as closely as possible.
- Keep each item short: at most 2 sentences and no more than 70 words.
- Do not use personal information references like name, age, or address - a PII
  check runs downstream and will flag any such content.
- Do not deviate from the core business problem/domain the input data addresses.
- Understand the data, relate it to the industry it represents, and generate data
  accordingly, covering a similarly diverse and unbiased range of cases as the
  input without inventing unrelated scenarios.
"""

class Response(BaseModel):
    created_at: str
    items: list[str]

    @property
    def content(self) -> str:
        return "\n".join(self.items)

def run(user_prompt, selected_model, case_count):
    system_prompt = SYSTEM_PROMPT.format(case_count=case_count)

    response = ollama.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        format=Response.model_json_schema(),
        options={
            'think': False,
            'num_predict': -1
        }
    )

    resp = Response.model_validate_json(response.message.content)

    if len(resp.items) > case_count:
        resp.items = resp.items[:case_count]

    return resp

