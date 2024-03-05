__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from tempfile import NamedTemporaryFile
from typing import List

import chromadb
import chainlit as cl
from chromadb.config import Settings
from chainlit.types import AskFileResponse
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.schema import StrOutputParser, Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings


def process_file(*, file: AskFileResponse) -> List[Document]:
    """Processes one PDF file from a Chainlit AskFileResponse
    object by first loading the PDF document and then chunk it
    into sub documents. Only supports PDF files.

    Args:
        file (AskFileResponse): input file to be processed

    Raises:
        ValueError: when we fail to process PDF files. We
        consider PDF file processing failure when there's
        no text returned. For example, PDF's with only image
        contents, corrputed PDFs, etc.

    Returns:
        List[Document]: List of Document(s). Each individual
        document has two fields: page_content(string) and
        metadata(dict).
    """
    # We only support PDF as input.

    if file.type != "application/pdf":
        return TypeError("Only PDF files are supported")

    with NamedTemporaryFile() as tempfile:
        # tempfile.write(file.content)

        loader = PDFPlumberLoader(file.path)
        documents = loader.load()

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=100,  # 3000 x 5 = 15000
        #     chunk_overlap=20,
        # )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

    # We are adding source_id into the metadata here to denote
    # which source document it is.
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    if not docs:
        raise ValueError("PDF file parsing failed.")

    return docs


def create_search_engine(*, docs: List[Document], embeddings: Embeddings) -> VectorStore:
    """Takes a list of Langchain Documents and an embedding
    model API wrapper and build a search index using a
    VectorStore.

    Args:
        docs (List[Document]): List of Langchain Documents to
        be indexed into the search engine.
        embeddings (Embeddings): encoder model API used to
        calculate embedding

    Returns:
        VectorStore: Langchain VectorStore
    1"""
    # Initialize Chromadb client to enable resetting and
    # disable telemetry
    client = chromadb.EphemeralClient()
    client_settings = Settings(allow_reset=True, anonymized_telemetry=False)

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()

    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        embedding=embeddings,
        client_settings=client_settings,
    )
    return search_engine


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload the PDF file you want to ask questions against.",
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()

    file = files[0]

    # Send message to user to let them know we are processing the file
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    msg.content = f"`{file.name}` processed. Loading ..."
    await msg.update()

    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    try:
        search_engine = await cl.make_async(create_search_engine)(
            docs=docs, embeddings=embeddings
        )
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError

    msg.content = f"`{file.name}` loaded. You can now ask questions!"
    await msg.update()

    (repo_id, model_file_name) = (
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "mistral-7b-instruct-v0.1.Q4_0.gguf",
    )

    model_path = hf_hub_download(
        repo_id=repo_id, filename=model_file_name, repo_type="model"
    )

    # initialize LlamaCpp LLM model
    # n_gpu_layers, n_bach, and n_ctx are for GPU support.
    # When not set, CPU will be used
    # set 1 for Mac m2, and higher numbers based on your GPU support
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0,
        max_tokens=512,
        top_p=1,
        # callback_manager=callback_manager,
        # n_gpu_layers=1,
        # n_batch=512,
        n_ctx=4096,
        # stop=["[INST]"],
        verbose=False,
        streaming=True,
    )

    # Template you will use to structure your user input before
    # converting into a prompt. Here, my template first injects
    # the personality I wish to give to the LLM before in the
    # form of system_prompt pushing the actual prompt from the
    # user. Note that this chatbot doesn't have any memory of
    # the conversation. So we will inject the system prompt for
    # each message.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are Chainlit GPT, a helpful assistant."),
            ("human", "{question}"),
        ]
    )

    # chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=search_engine.as_retriever()
    )

    # Let's save the chain from user_session so we do not have
    # to rebuild every single time we receive a message
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Let's load the chain from user_session
    chain = cl.user_session.get("chain")  # type: LLMChain

    # response = await chain.arun(
    #     question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    # )
    # await cl.Message(content=reponse).send()

    response = await chain.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
    )
    answer = response["answer"]
    sources = response["sources"].strip()

    # Get all of the documents from user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Addding sources to the answer
    source_elements = []
    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue

            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"

        else:
            answer += "\nNo source found"

    await cl.Message(content=answer, elements=source_elements).send()
