import openai
import os


client = openai.OpenAI(
    api_key=os.environ["IBM_LITELLM_API_KEY"],
    base_url=os.environ["IBM_LITELLM_URL"] # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="Azure/gpt-4o", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)