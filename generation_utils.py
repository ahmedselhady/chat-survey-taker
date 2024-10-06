from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from langchain.schema import BaseOutputParser
from typing import List
import torch
import re


class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


class ResponseClassificationParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return re.split("Class:")[-1].strip()

    @property
    def _type(self) -> str:
        return "output_parser"
