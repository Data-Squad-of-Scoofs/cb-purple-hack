# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

templ1 = """Ты - умный помощник, созданный для помощи в решении вопросов, связанных с пониманием текста.  Ты обязан общаться по-русски.
Получив фрагмент текста, ты должен придумать пару из вопроса и ответа, которые гипотетически может задать клиент Центрального Банка РФ чат-боту. 
Придумывая пару вопрос/ответ, ты должен возвращать результат в следующем формате:
{{
    "question": "$ТВОЙ_ВОПРОС_ЗДЕСЬ",
    "answer": "$ТВОЙ_ОТВЕТ_ЗДЕСЬ"
}}
```.venv/lib/python3.10/site-packages/langchain/chains/qa_generation/prompt.py

Все, что находится между ```, должно быть корректным json. Если не уверен, что в данном тексте содержится информация, по которой клиент может гипотетически задать вопрос, пиши "-".
"""
templ2 = """Пожалуйста, придумай пару вопрос/ответ в указанном формате JSON для следующего текста:
----------------
{text}"""
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(templ1),
        HumanMessagePromptTemplate.from_template(templ2),
    ]
)
templ = """Ты - умный помощник, созданный для помощи в решении вопросов, связанных с пониманием текста.  Ты обязан общаться по-русски.
Получив фрагмент текста, ты должен придумать пару из вопроса и ответа, которые гипотетически может задать клиент Центрального Банка РФ чат-боту. 
Придумывая пару вопрос/ответ, ты должен возвращать результат в следующем формате:
{{
    "question": "$ТВОЙ_ВОПРОС_ЗДЕСЬ",
    "answer": "$ТВОЙ_ОТВЕТ_ЗДЕСЬ"
}}
```

Все, что находится между ```, должно быть корректным json. Если не уверен, что в данном тексте содержится информация, по которой клиент может гипотетически задать вопрос, пиши "-".

Пожалуйста, придумай пару вопрос/ответ в указанном формате JSON для следующего текста:
----------------
{text}"""
PROMPT = PromptTemplate.from_template(templ)

CUSTOM_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
