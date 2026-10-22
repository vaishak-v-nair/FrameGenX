from dotenv import load_dotenv
load_dotenv()

from src.core.llm_router import LLMRouter
import openai
from openai import RateLimitError
import httpx

# Mock the OpenAI client temporarily
original_create = openai.resources.chat.completions.Completions.create

call_count = 0

def mocked_create(*args, **kwargs):
    global call_count
    call_count += 1
    if call_count == 1:
        # First call fails with rate limit
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        raise RateLimitError(
            message="Too many requests",
            response=response,
            body=None
        )
    return original_create(*args, **kwargs)

openai.resources.chat.completions.Completions.create = mocked_create

router = LLMRouter()
messages = [{"role": "user", "content": "Return the JSON: {\"success\": true}"}]

print("Testing generate_json with forced rate limit on first call...")
result = router.generate_json(messages)
print("Final result:", result)
