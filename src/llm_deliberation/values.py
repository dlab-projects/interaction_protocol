import json
import os

from google import genai
from openai import OpenAI
from pydantic import BaseModel


class Values(BaseModel):
    answers: list[str]


def collapse_values(flat_values, messages):
    """
    Reshape a flat list of values into a nested list of lists, matching the shape of messages.

    Args:
        flat_values: List[Any] -- flat output from batch run, length = total messages
        messages: List[List[Any]] -- original nested messages; only the shape is used

    Returns:
        List[List[Any]] -- nested structure matching messages
    """
    output = []
    idx = 0
    for msg_list in messages:
        inner = []
        for _ in msg_list:
            if idx >= len(flat_values):
                raise ValueError("Not enough flat_values to fill out the messages structure.")
            inner.append(flat_values[idx])
            idx += 1
        output.append(inner)
    if idx != len(flat_values):
        raise ValueError("Some flat_values left unused; messages structure does not match flat_values length.")
    return output


def process_value_batches(batch_id):
    """
    Process a Gemini AI batch job to extract values from batch responses.

    Downloads and processes the output file from a completed Gemini batch job,
    extracting the 'answers' field from each response and converting them to sets.

    Args:
        batch_id (str): The unique identifier of the Gemini batch job
    Returns:
        List[set]: A list of sets, where each set contains the values/answers 
                   extracted from one batch response
    Raises:
        KeyError: If the expected response structure is not found
        json.JSONDecodeError: If response content is not valid JSON
    """
    # Initialize Gemini client with API key from environment
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # Retrieve the batch job details
    batch = client.batches.get(name=batch_id)
    # Get the output file ID from the batch destination
    file_id = batch.dest.file_name
    # Download the batch results file
    file = client.files.download(file=file_id)
    # Parse the file content - each line is a separate JSON response
    lines = file.decode('utf-8').strip().split('\n')
    outputs = [json.loads(line) for line in lines]
    # Extract values from each response and convert to sets for deduplication
    # Navigate the nested response structure: response -> candidates[0] -> content -> parts[0] -> text
    values = {}
    for output in outputs:
        key = int(output['key'].replace('request', ''))
        try:
            values[key] = set(json.loads(output['response']['candidates'][0]['content']['parts'][0]['text'])['answers'])
        except:
            print(f"Error processing output: {output}")
            values[key] = set()

    return [values[key] for key in range(len(values))]


def process_openai_value_batches(batch_id, *, api_key_env_var: str = "OPENAI_API_KEY"):
    """
    Process an OpenAI batch job to extract values from batch responses.

    Args:
        batch_id (str): OpenAI batch identifier returned by the submission call.
        api_key_env_var (str): Environment variable holding the OpenAI API key.

    Returns:
        List[set]: Set of answers for each processed request, ordered by request index.
    """

    api_key = os.getenv(api_key_env_var)
    client = OpenAI(api_key=api_key)

    batch = client.batches.retrieve(batch_id)
    file_id = getattr(batch, "output_file_id", None)
    if not file_id:
        raise ValueError("Batch output not available yet. Check that the batch has completed.")

    text = client.files.content(file_id).read()

    values: dict[int, set[str]] = {}
    for line in text.strip().splitlines():
        record = json.loads(line)
        custom_id = record.get("custom_id") or ""
        if not custom_id.startswith("request"):
            continue
        idx = int(custom_id.replace("request", ""))
        try:
            body = record["response"]["body"]
            choice = body["choices"][0]["message"]
            if isinstance(choice.get("content"), list):
                text_chunks = [chunk.get("text", "") for chunk in choice["content"] if isinstance(chunk, dict)]
                message_text = "".join(text_chunks)
            else:
                message_text = choice.get("content", "")
            values[idx] = set(json.loads(message_text)["answers"])
        except Exception:
            print(f"Error processing output: {record}")
            values[idx] = set()

    return [values[idx] for idx in range(len(values))]


values = [
   "Trust creation and maintenance",
   "Constructive dialogue",
   "Respect and dignity",
   "Professional ethics and integrity",
   "Social etiquette",
   "Religious respect and accommodation",
   "Linguistic respect and inclusivity",
   "Cultural understanding and respect",
   "Cultural heritage and tradition",
   "Financial wellbeing",
   "Sexual freedom and pleasure",
   "Protection of self and others from harm",
   "Environmental consciousness",
   "Authentic expression",
   "Workplace boundaries",
   "Parental care",
   "Consumer and client protection",
   "Child welfare",
   "Animal and pet welfare",
   "Worker welfare and dignity",
   "Workplace etiquette and respect",
   "Economic justice and fairness",
   "Healthcare equity and access",
   "Consent and personal boundaries",
   "Property rights protection",
   "Personal autonomy",
   "Emotional safety and support",
   "Mental health sensitivity and support",
   "Power dynamics values",
   "Privacy and confidentiality",
   "Religious and spiritual authenticity",
   "Emotional intelligence and regulation",
   "Emotional intimacy",
   "Prosocial altruism",
   "Honest communication",
   "Intergenerational respect and relationships",
   "Supportive and caring relationships",
   "Family bonds and cohesion",
   "Conflict resolution and reconciliation",
   "Public good and community engagement",
   "Accessibility",
   "Reciprocal relationship quality",
   "Environmental consciousness",
   "Empathy and understanding",
   "Personal growth",
   "Achievement and recognition",
   "Balance and moderation",
   "Physical health and wellbeing",
   "Personal accountability and responsibility"
]

map_values_to_groups = {
    "Trust creation and maintenance": 0,
    "Constructive dialogue": 0,
    "Respect and dignity": 0,
    "Professional ethics and integrity": 0,
    "Social etiquette": 1,
    "Religious respect and accommodation": 1,
    "Linguistic respect and inclusivity": 1,
    "Cultural understanding and respect": 1,
    "Cultural heritage and tradition": 1,
    "Financial wellbeing": 2,
    "Sexual freedom and pleasure": 3,
    "Protection of self and others from harm": 4,
    "Environmental consciousness": 5,
    "Authentic expression": 6,
    "Workplace boundaries": 7,
    "Parental care": 8,
    "Consumer and client protection": 8,
    "Child welfare": 8,
    "Animal and pet welfare": 8,
    "Worker welfare and dignity": 9,
    "Workplace etiquette and respect": 9,
    "Economic justice and fairness": 10,
    "Healthcare equity and access": 10,
    "Consent and personal boundaries": 11,
    "Property rights protection": 11,
    "Personal autonomy": 11,
    "Emotional safety and support": 12,
    "Mental health sensitivity and support": 12,
    "Power dynamics values": 12,
    "Privacy and confidentiality": 12,
    "Religious and spiritual authenticity": 13,
    "Emotional intelligence and regulation": 14,
    "Emotional intimacy": 14,
    "Prosocial altruism": 15,
    "Honest communication": 16,
    "Intergenerational respect and relationships": 17,
    "Supportive and caring relationships": 17,
    "Family bonds and cohesion": 17,
    "Conflict resolution and reconciliation": 17,
    "Public good and community engagement": 17,
    "Accessibility": 18,
    "Reciprocal relationship quality": 19,
    "Empathy and understanding": 19,
    "Personal growth": 20,
    "Achievement and recognition": 20,
    "Balance and moderation": 20,
    "Physical health and wellbeing": 21,
    "Personal accountability and responsibility": 11
}
