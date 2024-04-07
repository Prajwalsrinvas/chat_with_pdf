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
