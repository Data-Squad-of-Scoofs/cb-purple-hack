from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate
import time
import json
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain

templ1 = """Ты - умный помощник, созданный для помощи в решении вопросов, связанных с пониманием текста. Ты обязан общаться по-русски.
Получив фрагмент текста, ты должен придумать пару из вопроса и ответа, а также указать контекст, к которому они относятся. Эти данные могут понадобиться клиенту Центрального Банка РФ.
Придумывая пару вопрос/ответ и контекст, ты должен возвращать результат в следующем формате:
{
    "question": "$ТВОЙ_ВОПРОС_ЗДЕСЬ",
    "answer": "$ТВОЙ_ОТВЕТ_ЗДЕСЬ",
    "context": "$КОНТЕКСТ_ЗДЕСЬ"
}
Все, что находится между ```, должно быть корректным json. Если не уверен в информации, пиши "-".
"""

templ2 = """Пожалуйста, придумай пару вопрос/ответ и укажи контекст в указанном формате JSON для следующего текста:
----------------
{text}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(templ1),
        HumanMessagePromptTemplate.from_template(templ2),
    ]
)

templ = templ1 + "\n" + templ2
PROMPT = PromptTemplate.from_template(templ)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)


class CustomQAGenerationChain(Chain):
    """Custom class for question-answer generation chains with manual context inclusion."""

    llm_chain: LLMChain
    text_splitter: TextSplitter = Field(default=RecursiveCharacterTextSplitter(chunk_overlap=500))
    input_key: str = "text"
    output_key: str = "qa_pairs"

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: Optional[BasePromptTemplate] = None, **kwargs: Any):
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, List]:
        docs = self.text_splitter.create_documents([inputs[self.input_key]])

        qa_pairs = []

        for i, doc in enumerate(docs):
            if i > 0:
                time.sleep(60)  # Delay between mini-batches

            result = self.llm_chain.generate([{"text": doc.page_content}], run_manager=run_manager)
            for res in result.generations:
                try:
                    qa_batch = json.loads(res[0].text)
                    if isinstance(qa_batch, dict) and "question" in qa_batch and "answer" in qa_batch:
                        qa_batch["context"] = doc.page_content  # Manually insert context
                        qa_pairs.append(qa_batch)
                    else:
                        print(f"\n[Invalid QA batch format: {qa_batch}]")
                except json.JSONDecodeError:
                    print(f"\n[Failed to decode JSON for batch starting with: {res[0].text[:30]}]")
                    continue

        return {self.output_key: qa_pairs}