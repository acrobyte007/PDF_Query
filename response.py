##stores embedding vector and corresponding text with vector custom_id
##filter embeddings by custom_id
##search for similar embeddings
##return text data with given vector custom_id
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=1,
)

prompt_tamplet = """
    You just need to answer the question based on the following context.
    QUESTIONS : {question}
    CONTEXT : {context}
"""



def get_answer(question:str,context:str):
    final_prompt = prompt_tamplet.format(question=question, context=context)
    response = llm.invoke(final_prompt)
    ##print("from planner :",type(response.content))
    return response.content