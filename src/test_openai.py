from openai import OpenAI
from config import OPENAI_API_KEY

def main() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Say: OpenAI connection works."
    )

    print(response.output_text)

if __name__ == "__main__":
    main()