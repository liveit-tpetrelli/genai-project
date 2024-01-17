import os
from configs.app_configs import AppConfigs

from langchain.vectorstores import Chroma
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings

from langchain.prompts import PromptTemplate

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    *gcloud_credentials.dir_path,
    gcloud_credentials.file_name
)

persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.persist_directory)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loader = NotionDirectoryLoader(os.path.join("db", "notion"))
    docs = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=2000,
    #     chunk_overlap=200
    # )
    # splits = text_splitter.split_documents(docs)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    splits = markdown_splitter.split_text(docs[0].page_content)
    print(splits[0])

    embeddings = VertexAIEmbeddings()

    if not os.listdir(persist_directory):
        vector_chromadb = Chroma.from_documents(
            documents=splits,
            persist_directory=persist_directory,
            embedding=embeddings)
    else:
        vector_chromadb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    print(vector_chromadb._collection.count())

    # query it
    query = "List 3 easy code challenges about Array."
    docs = vector_chromadb.similarity_search(query)
    print(docs[0].page_content)

    template = """
    Use the following pieces of context to answer \
    the question at the end. If you don't know the answer, \
    just say that you don't know, don't try to make up an answer. \
    Use three sentences maximum. Keep the answer as concise as possible. \
    Always say "thanks for asking!" at the end of the answer.
    
    ####
    {context}
    ####
    
    Question: {question}
    Helpful Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template, )

    # Run chain
    from langchain.chains import RetrievalQA

    llm = VertexAI(model_name="gemini-pro")
    question = "Can you give me 3 easy leet code challanges about arrays?"
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vector_chromadb.as_retriever(),
                                           return_source_documents=True,
                                           verbose=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": True})

    result = qa_chain({"query": question})
    print(result["result"])
