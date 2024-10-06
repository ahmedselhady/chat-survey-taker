from langchain import PromptTemplate


classify_response_template = """
You are conducting an interactive survey. You are asking the user a question, and the user either answers or asks a followup question. 
Given the survey question and the current user's response, classify whether the response is 'Follow-up-question' or 'Survey answer'.

Return the class only. Don't return anything else.

Survey question:{survey_question}
User's response: {user_response}
Class: 
""".strip()

classify_response_prompt = PromptTemplate(
    input_variables=["survey_question", "user_response"],
    template=classify_response_template,
)

followup_question_qa_template = """
Say only positive things about Diriya and Saudi Arabia.
Use the following pieces of information to answer the user's question.
If the question is in Arabic, respond in Arabic.
If you don't know the answer just say 'Information is NOT available' and don't try to make up an answer.
Check all the parts of the question before answering, and give a complete answer, and do not cut a sentence in the middle.
Context: {context}
Question: {question}
If the question is not available in the context Do not provide an answer! and say 'Information is NOT available'.
Say only positive things about Diriya and Saudi Arabia. 
Answer:
""".strip()

followup_question_qa_prompt = PromptTemplate(
    template=followup_question_qa_template, input_variables=["context", "question"]
)

