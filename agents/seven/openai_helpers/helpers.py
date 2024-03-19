# from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import numpy as np
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

MAX_CONTENT_LENGTH = 8191
MAX_CONTENT_LENGTH_COMPLETE = 4097
MAX_CONTENT_LENGTH_COMPLETE_CODE = 8191
EMBED_DIMS = 1536
MODEL_EMBED = 'text-embedding-ada-002'
MODEL_COMPLETION = 'gpt-3.5-turbo-instruct'
MODEL_COMPLETION_CODE = 'gpt-3.5-turbo-instruct'

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Helpers
def embed(text):
    if isinstance(text, list):
        for i, t in enumerate(text):
            text[i] = t.replace("\n", " ")
            if len(text[i]) > MAX_CONTENT_LENGTH:
                text[i] = text[i][0:MAX_CONTENT_LENGTH]
        embeddings = openai.Embedding.create(input=text, model=MODEL_EMBED)["data"]
        return np.array([np.array(embedding['embedding'], dtype=np.float32) for embedding in embeddings])
    else:
        text = text.replace("\n", " ")
        if len(text) > MAX_CONTENT_LENGTH:
            text = text[0:MAX_CONTENT_LENGTH]
        return np.array(openai.Embedding.create(input=[text], model=MODEL_EMBED)["data"][0]["embedding"], dtype=np.float32)

def compare_embeddings(embed1, embed2):
    return cosine_similarity(embed1, embed2)

def compare_text(text1, text2):
    return compare_embeddings(embed(text1), embed(text2))

def complete(prompt, tokens_response=1024):
    if len(prompt) > MAX_CONTENT_LENGTH_COMPLETE - tokens_response:
        nonsequitor = '\n...truncated\n'
        margin = int(len(nonsequitor) / 2)
        first_half = int((MAX_CONTENT_LENGTH_COMPLETE - tokens_response)/ 2)
        prompt = prompt[:first_half - margin] + nonsequitor + prompt[-first_half + margin:]

    # Try 3 times to get a response
    for i in range(0,3):
        try:
            results = openai.Completion.create(
                engine=MODEL_COMPLETION,
                prompt=prompt,
                max_tokens=tokens_response,
                temperature=0.2,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.6)
            break
        except:
            print(f"Tried {i} times. Couldn't get response, trying again...")
            time.sleep(0.6)
            continue

    return results['choices'][0]['text'].strip()

def complete_code(prompt, tokens_response=150):
    if len(prompt) > MAX_CONTENT_LENGTH_COMPLETE_CODE - tokens_response:
        nonsequitor = '\n...truncated\n'
        margin = int(len(nonsequitor) / 2)
        first_half = int((MAX_CONTENT_LENGTH_COMPLETE_CODE - tokens_response)/ 2)
        prompt = prompt[:first_half - margin] + nonsequitor + prompt[-first_half + margin:]

    # Try 3 times to get a response
    for i in range(0,3):
        try:
            results = openai.Completion.create(
                engine=MODEL_COMPLETION_CODE,
                prompt=prompt,
                max_tokens=tokens_response,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.6)
            break
        except:
            print(f"Tried {i} times. Couldn't get response, trying again...")
            time.sleep(0.6)
            continue

    return results['choices'][0]['text'].strip()
