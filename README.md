## Lab1: Introduction to Chainlit

We will be using [Chainlit](https://docs.chainlit.io) as the frontend framework to develop our LLM Powered applications. Chainlit is an open-source Python package that makes it incredibly fast to build Chat GPT like applications with your own business logic and data.

### Exercise 1a:
Please add the proper decorator to this main function so Chainlit will call this function when it receives a message
```python
@cl.on_message
```

### Exercise 1b:
Please get the content of the chainlit Message and send it back as a reponse
```python
await cl.Message(content=f"Received: {message.content}").send()
```

## Lab2: Adding LLM to Chainlit App

Now we have a web interface working, we will now add an LLM to our Chainlit app to have our simplified version of ChatGPT. We will be using [LangChain](https://python.langchain.com/) as the framework for this course. It provides easy abstractions and a wide varieties of data connectors and interfaces for everything LLM app development.

## References:

- [Langchain's Prompt Template](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/#chatprompttemplate)
- [Langchain documentation](https://python.langchain.com/docs/modules/chains/foundational/llm_chain#legacy-llmchain)
- [Chainlit's documentation](https://docs.chainlit.io/get-started/pure-python)

### Exercise 1a:

Our Chainlit app should initialize the LLM chat via langchain at the start of a chat session.

First, we need to choose an LLM from OpenAI's list of models. Remember to set streaming=True for streaming tokens
```python
...
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp

...
    (repo_id, model_file_name) = (
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "mistral-7b-instruct-v0.1.Q4_0.gguf",
    )

    model_path = hf_hub_download(
        repo_id=repo_id, filename=model_file_name, repo_type="model"
    )

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0,
        max_tokens=512,
        # callback_manager=callback_manager,
        # n_gpu_layers=1,
        # n_batch=512,
        # n_ctx=4096,
        # stop=["[INST]"],
        verbose=False,
        streaming=True,
    )
```

### Exercise 1b:

Next, we will need to set the prompt templates for chat. Prompt templates is how we set prompts and then inject informations into the prompt.

Please create the prompt templates using ChatPromptTemplate. Use variable name "question" as the variable in the template.

Refer to the documentation listed in the README.md file for reference.

```python
...
from langchain.prompts import ChatPromptTemplate
...

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are Chainlit GPT, a helpful assistant."),
            ("human", "{question}"),
        ]
    )
```

### Exercise 1c:

Now we have model and prompt, let's build our Chain. A Chain is one or a series of LLM calls. We will use the default StrOutputParser to parse the LLM outputs.
```python
...
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
...

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
```

### Exercise 1d:

Everytime we receive a new user message, we will get the chain from user_session. We will run the chain with user's question and return LLM response to the user.

```python
response = await chain.arun(
    question=message.content, callbacks=[cl.LangchainCallbackHandler()]
)
```

## Lab3: Enabling Load PDF to Chainlit App

Building on top of the current simplified version of ChatGPT using Chainlit, we now going to add loading PDF capabilities into the application.

In this lab, we will utilize the build in PDF loading and parsing connectors inside Langchain, load the PDF, and chunk the PDFs into individual pieces with their associated metadata

## References

- [Langchain PDF Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
- [Langchain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/#text-splitters)
- [Chainlit Ask File Message](https://docs.chainlit.io/api-reference/ask/ask-for-file)

### Exercise 1a:

We have the input PDF file saved as a temporary file. The name of the file is 'tempfile.name'. Please use one of the PDF loaders in Langchain to load the file.

```python
...
from langchain_community.document_loaders import PDFPlumberLoader
...

        loader = PDFPlumberLoader(file.path)
        documents = loader.load()
```

### Exercise 1b:

We can now chunk the documents now it is loaded. Langchain provides a list of helpful text splitters. Please use one of the splitters to chunk the file.

```python
...
from langchain.text_splitter import RecursiveCharacterTextSplitter
...

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, # 3000 x 5 = 15000
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

```

### Exercise 1c:

At the start of our Chat with PDF app, we will first ask users to upload the PDF file they want to ask questions against.

Please use Chainlit's AskFileMessage and get the file from users.
```python
...
        files = await cl.AskFileMessage(
            content="Please upload the PDF file you want to ask questions against.",
            accept=["application/pdf"],
            max_size_mb=10,
        ).send()
```

## Lab4: Indexing Documents into Vector Database

In the previous lab, we enabled document loading and chunking them into smaller sub documents. Now, we will need to index them into our search engine vector database in order for us to build our Chat with PDF application using the RAG (Retrieval Augmented Generation) pattern.

In this lab, we will implement adding OpenAI's embedding model and index the documents we chunked in the previous section into a Vector Database. We will be using [Chroma](https://www.trychroma.com/) as the vector database of choice. Chroma is a lightweight embedding database that can live in memory, similar to SQLite.

## References
- [Langchain Embedding Models](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [ChromaDB Langchain Integration](https://docs.trychroma.com/integrations/langchain)

### Exercise 1:

Now we have defined our encoder model and initialized our search engine client, please create the search engine from documents

### Exercise 2:

Add OpenAI's embedding model as the encoder. The most standard one to use is text-embedding-ada-002

## Lab 5: Putting it All Together

In Lab 2, we created the basic scaffold of our Chat with PDF App. In Lab 3, we added PDF uploading and processing functionality. In Lab 4, we added the capability to indexing documents into a vector database. Now we have all the required pieces together, it's time for us to assemble our RAG (retrieval-augmented generation) system using Langchain.

## References

- [Langchain RetrivalQA](https://python.langchain.com/docs/use_cases/web_scraping#research-automation)

### Exercise 1:

Now we have search engine setup, our Chat with PDF application can do RAG architecture pattern. Please use the appropriate RetrievalQA Chain from Langchain.

Remember, we would want to set the model temperature to 0 to ensure model outputs do not vary across runs, and we would want to also return sources to our answers.
