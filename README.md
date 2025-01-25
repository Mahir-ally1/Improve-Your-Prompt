# Improve-Your-Prompt
Building a framework through RAGs implemented using LangChain and OpenAI models to provide guidance to improve a user inputted prompt based on prompt engineering literature. 

## Description 

This script was built using Retrieval Augmented Generation (RAG) using `LangChain`. It uses the  `text-embedding-3-small` model for calculating embeddings of the provided context, `Chroma` for the vector store and `gpt-4o-mini` as the Large Language Model (LLM). 

## Purpose 

Prompt engineering is a significant field and the method to interact with these foundational large language models. It is a complex but rewarding field and already has a lot of literature. Using some of the literature, the aim of this project is to build a solution to recommend improvements to an existing user-inputted prompt. 

## Setup

- Insert the OpenAI LLM Key in the `llm_secrets.py` file under the `llm_api_key` variable
- Setup: 
```sh
git clone https://github.com/Mahir-ally1/Improve-Your-Prompt.git
pip install -r requirements.txt
```
- Make any changes to the prompt in the [instructions](prompts/instruction_prompt.txt)
- Run `prompt_improver.py`


## Next Steps

Adding more context in the [guidelines](context_guidelines/) folder


