import os

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

if __name__ == "__main__":
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    with open("data/constitution.txt", "r") as file:
        content = file.read()

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize this text: {content}",
                    }
                ],
                model="gpt-3.5-turbo",
            )
            print(chat_completion.choices[0].message.content)
        except BadRequestError as e:
            print(f"Error calling OpenAI: {e}")

"""output:
(chat-pdf-py3.11) $ python limitations.py
Error calling OpenAI: Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 24886 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
"""
