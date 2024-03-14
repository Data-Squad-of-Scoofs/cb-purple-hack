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
from custom_prompt import CUSTOM_PROMPT_SELECTOR

class CustomQAGenerationChain(Chain):
    """Custom class for question-answer generation chains with delays between mini-batches."""

    llm_chain: LLMChain
    text_splitter: TextSplitter = Field(default=RecursiveCharacterTextSplitter(chunk_overlap=500))
    input_key: str = "text"
    output_key: str = "questions"
    k: Optional[int] = None

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: Optional[BasePromptTemplate] = None, **kwargs: Any):
        _prompt = prompt or CUSTOM_PROMPT_SELECTOR.get_prompt(llm)
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
                time.sleep(60)  # delay between mini-batches

            result = self.llm_chain.generate([{"text": doc.page_content}], run_manager=run_manager)
            for res in result.generations:
                try:
                    qa_batch = json.loads(res[0].text)
                    if isinstance(qa_batch, dict) and "question" in qa_batch and "answer" in qa_batch:
                        qa_batch["context"] = doc.page_content  # insert context
                        qa_pairs.append(qa_batch)
                    else:
                        print(f"\n[Invalid QA batch format: {qa_batch}]")
                except json.JSONDecodeError:
                    print(f"\n[Failed to decode JSON for batch starting with: {res[0].text[:30]}]")
                    continue

        return {self.output_key: qa_pairs}