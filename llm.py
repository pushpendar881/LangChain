from langchain_huggingface import HuggingFaceEndpoint
from secretkey import sec_key #import your API key 

import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st


st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="detail information about  {name}"
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# HuggingFaceEndpoint for LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7,
    huggingfacehub_api_token=sec_key  # Correct token argument
)

# Chains
chain = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

second_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="research and events happen in that {name}"
)

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world"
)

chain3 = LLMChain(
    llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory)

# Sequential chain combining the previous chains
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# Streamlit Input Handling
if input_text:
    result = parent_chain({'name': input_text})
    st.write(result)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
