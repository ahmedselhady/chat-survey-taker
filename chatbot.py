import re
import warnings
from typing import List

from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    pipeline,
)
from generation_utils import StopGenerationCriteria, ResponseClassificationParser
from configs import (
    MODEL_NAME,
    DEVICE,
    HF_TOKEN,
    NF4_CONFIGS,
    GEN_TEMPERATURE,
    GEN_SENTENCES,
    GEN_TOKEN_LIMIT,
    GEN_REPEAT_PENALTY,
    FILES_DIR,
)
from pprint import pprint
from prompt_utils import classify_response_prompt, followup_question_qa_prompt
from retreiver_utils import DBretriever
import mtranslate
from enum import Enum

warnings.filterwarnings("ignore", category=UserWarning)


def translate_user_response(response: str, source_lang="en"):
    try:
        translation = mtranslate.translate(
            response, to_language="en", from_language=source_lang
        )
        return translation
    except Exception as e:
        raise RuntimeError("Unable to translate the user's response")


def translate_back_system_response(response: str, target_lang="en"):
    try:
        translation = mtranslate.translate(
            response, to_language=target_lang, from_language="en"
        )
        return translation

    except Exception as e:
        raise RuntimeError("Unable to translate the system's response back")


class SurveyQuestionState(Enum):
    FOLLOWUP = 0
    ANSWER_RECEIVED = 1


class SurveyChatBot:

    def __init__(self, use_memory: bool = False) -> None:

        llm_pipeline = self._initialize_llm_pipeline()

        self.classification_chain = classify_response_prompt | llm_pipeline
        dbr = DBretriever(FILES_DIR)

        self.help_qa_chain = RetrievalQA.from_chain_type(
            llm=llm_pipeline,
            chain_type="stuff",
            retriever=dbr.db_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": followup_question_qa_prompt},
        )

    def _initialize_llm_pipeline(self):

        pprint(f"Creating a chatbot using model: {MODEL_NAME}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=NF4_CONFIGS if DEVICE != "cpu" else None,
            device_map=DEVICE,
            token=HF_TOKEN,
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        generation_config = model.generation_config
        self._update_generation_configs(generation_config, tokenizer)
        pprint(f"Using generation configs: {generation_config}")

        stop_tokens = [
            ["Survey", "question" ":"],
            ["User's", "response", ":"],
            ["Class", ":"],
        ]
        #default_stopping_criteria = StoppingCriteriaList(
        #    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
        #)

        pprint(f"Setting stopping tokens to {stop_tokens}")
        generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            #stopping_criteria=default_stopping_criteria,
            generation_config=generation_config,
        )

        llm_pipeline = HuggingFacePipeline(pipeline=generation_pipeline)
        return llm_pipeline

    def _update_generation_configs(self, generation_configs, tokenizer):

        generation_configs.temperature = GEN_TEMPERATURE
        generation_configs.num_return_sequences = (
            GEN_SENTENCES  # Limit to a single liners to avoid randomness
        )
        generation_configs.max_new_tokens = GEN_TOKEN_LIMIT
        generation_configs.use_cache = False
        generation_configs.repetition_penalty = (
            GEN_REPEAT_PENALTY  # High penalty to avoid the model repeating itself
        )
        generation_configs.pad_token_id = tokenizer.eos_token_id
        generation_configs.eos_token_id = tokenizer.eos_token_id

    def get_response_class(self, question: str, response: str): ...

    def answer_user_followup(self, follow_up_question: str): ...

    def invoke(self, survey_question: str, user_response: str, input_lang: str = "en"):

        if input_lang != "en":
            user_response = translate_user_response(user_response, input_lang)
        
        response_type = self.classification_chain.invoke(
            {"survey_question": survey_question, "user_response": user_response}
        )
        response_type = response_type.split("Class:")[-1].strip()
        if "follow" in response_type.lower():
            system_response = self.help_qa_chain({"query": user_response})
            return system_response, SurveyQuestionState.FOLLOWUP
        else:
            return user_response, SurveyQuestionState.ANSWER_RECEIVED