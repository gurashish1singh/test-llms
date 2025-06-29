from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
from pydantic import (
    BaseModel,
    Field,
)

MODEL_NAME = "mistral"
PROMPT_COUNTRY_INFO = """
Provide information about {country}.
{format_instructions}
"""


class Country(BaseModel):
    name: str = Field(description="Name of the country.")
    capital: str = Field(description="Name of the capital of the country.")


def main():
    llm = ChatOllama(model="mistral")
    parser = PydanticOutputParser(pydantic_object=Country)

    country = "WrongCountry"
    message = HumanMessagePromptTemplate.from_template(template=PROMPT_COUNTRY_INFO)
    chat_prompt = ChatPromptTemplate(messages=[message])
    chat_prompt_with_values = chat_prompt.format_prompt(
        country=country, format_instructions=parser.get_format_instructions()
    )

    print("Invoking LLM")
    results = llm.invoke(chat_prompt_with_values.to_messages())
    print(results.content)


if __name__ == "__main__":
    main()
