import os
from _ast import mod

from configs.app_configs import AppConfigs

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings

from langchain.chains import RetrievalQA, LLMChain

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

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

    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")

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
    # query = "List 3 easy code challenges about Array."
    # docs = vector_chromadb.similarity_search(query)
    # print(docs[0].page_content)

    # template = """
    # Use the following pieces of context to answer \
    # the question at the end. If you don't know the answer, \
    # just say that you don't know, don't try to make up an answer. \
    # Use three sentences maximum. Keep the answer as concise as possible. \
    # Always say "thanks for asking!" at the end of the answer.
    #
    # ####
    # {context}
    # ####
    #
    # Question: {question}
    # Helpful Answer:
    # """
    # QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template, )

    # Run chain

    # llm = VertexAI(model_name="gemini-pro")
    # question = "Can you give me 3 easy leet code challanges about arrays?"
    # qa_chain = RetrievalQA.from_chain_type(llm,
    #                                        retriever=vector_chromadb.as_retriever(),
    #                                        return_source_documents=True,
    #                                        verbose=True,
    #                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": True})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    llm = VertexAI(model_name="gemini-pro", temperature=0)
    # memory = ConversationSummaryMemory(
    #     llm=llm, memory_key="chat_history", return_messages=True
    # )

    # This controls how the standalone question is generated.
    # Should take `chat_history` and `question` as input variables.
    template = """Given the following chat history and a follow up question,\
    rephrase the follow up input question to be a standalone question.
    
    Chat History:####
    {chat_history}
    ####

    Follow Up Input: ####
    {question}
    ####
    
    Standalone question:"""

    prompt = PromptTemplate.from_template(template)

    qa_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    ####
    
    If the chat history exists, then USE IT to answer the question when the context provided isn't enough. Otherwise, just use the context and add to your final answer that you cannot find the chat history.
    
    
    {chat_history}
    
    ####
    
    Question: {question}
    Helpful Answer:"""
    qa_prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=qa_template, )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
        memory=memory,
        condense_question_prompt=prompt,
        # return_source_documents=True,
        # return_generated_question=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True
    )

    # chat_history = []

    question = "Can you give me 3 easy code challenges about arrays?"
    result = qa({"question": question})
    # chat_history.extend([(question, result["answer"])])
    # gen_question = result["generated_question"]
    # src_docs = result["source_documents"]
    answer = result['answer']

    print(qa)
    print(result)
    print(result['answer'])

    # question = "Can you repeat the first option that you have just given to me?"
    question = "What was the first option that you selected?"
    result = qa({"question": question})
    # chat_history.extend([(question, result["answer"])])
    print(result['answer'])

    question = "No, not that sorry. What was the second one?"
    result = qa({"question": question})
    # chat_history.extend([(question, result["answer"])])
    print(result['answer'])

    # result = qa_chain({"query": question})
    # print(result["result"])


