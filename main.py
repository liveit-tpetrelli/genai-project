import os
from typing import Any

import umap
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from numpy._typing import _64Bit
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.app_configs import AppConfigs

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain.text_splitter import (MarkdownHeaderTextSplitter,
                                     SentenceTransformersTokenTextSplitter,
                                     RecursiveCharacterTextSplitter)

from langchain_google_vertexai import (VertexAI,
                                       VertexAIEmbeddings)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    *gcloud_credentials.dir_path,
    gcloud_credentials.file_name
)


def project_embeddings(_embeddings: list[list[float]], _umap_transform: umap.UMAP) \
        -> np.ndarray[Any, np.dtype[np.floating[_64Bit]]]:
    umap_embeddings = np.empty((len(_embeddings), 2))
    for i, _emb in enumerate(tqdm(_embeddings)):
        umap_embeddings[i] = _umap_transform.transform([_emb])
    return umap_embeddings


def augment_query_generated(query: str, model: str = "gemini-pro") -> str:
    messages = [
        SystemMessage(content=(
            "You are a helpful expert assistant for newly recruited data and software engineers. ",
            "Your main role is to guide newly hired engineers throw a entry study program. "
            "Provide an example answer to the given question, that might be found in "
            "a document that contains useful resources that the employee have to study, "
            "such as a document with a list of topics or a document with some useful links.")
        ),
        HumanMessage(content=query)
    ]
    _llm = VertexAI(model_name=model)
    return _llm.invoke(messages)


if __name__ == '__main__':
    chroma_persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.persist_directory)
    notion_persist_directory = os.path.join(*app_configs.configs.NotionLocalDbConfigs.persist_directory)
    vertex_embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")

    if not os.listdir(chroma_persist_directory):
        loader = NotionDirectoryLoader(notion_persist_directory)
        docs = loader.load()

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
            strip_headers=False
        )
        md_header_splits = []
        for doc in docs:
            md_header_splits += (markdown_splitter.split_text(doc.page_content))

        text_splitter = RecursiveCharacterTextSplitter(  # separators=["\n\n", "\n", "(?<=\. )", " ", ""]
            chunk_size=600,
            chunk_overlap=30
        )
        text_splits = text_splitter.split_documents(md_header_splits)

        token_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=200,
            chunk_overlap=30
        )
        token_split_texts = token_splitter.split_documents(text_splits)

        vector_chromadb = Chroma.from_documents(
            documents=token_split_texts,
            persist_directory=chroma_persist_directory,
            embedding=vertex_embeddings_model)
    else:
        vector_chromadb = Chroma(
            persist_directory=chroma_persist_directory,
            embedding_function=vertex_embeddings_model
        )

    # # TEST 1
    # query1 = "List 3 easy code challenges about Array."
    # retrieved_documents1 = vector_chromadb.similarity_search(query=query1, k=5)
    #
    # for document in retrieved_documents1:
    #     print(document.page_content)
    #     print('\n')
    #
    retrieved_embeddings1 = vector_chromadb.get(include=['embeddings'])['embeddings']
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(retrieved_embeddings1)
    projected_dataset_embeddings = project_embeddings(retrieved_embeddings1, umap_transform)
    #
    # plt.figure()
    # plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('Projected Embeddings')
    # plt.axis('off')
    # plt.show()
    #
    # # TEST 2
    # query2 = "List 3 easy code challenges about Array."
    # retrieved_documents2 = vector_chromadb.similarity_search(query=query2, k=5)
    # docs2 = []
    # for doc in retrieved_documents2:
    #     docs2.append(doc.page_content)
    #
    # query2_embedding = vertex_embeddings_model.embed_query(query2)
    # retrieved_docs2_embeddings = vertex_embeddings_model.embed_documents(docs2)
    #
    # projected_query_embedding = project_embeddings([query2_embedding])
    # projected_retrieved_embeddings = project_embeddings(retrieved_docs2_embeddings)
    #
    # plt.figure()
    # plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
    # plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
    # plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title(f'{query2}')
    # plt.axis('off')
    # plt.show()

    # TEST 3 - Expansion with generated answers
    original_query = "Give me some materials about a topic i need to study."
    hypothetical_answer = augment_query_generated(original_query)
    joint_query = f"{original_query} {hypothetical_answer}"
    # print(joint_query)

    retrieved_documents3 = vector_chromadb.similarity_search(query=joint_query, k=5)

    docs3 = []
    for doc in retrieved_documents3:
        docs3.append(doc.page_content)

    retrieved_docs3_embeddings = vertex_embeddings_model.embed_documents(docs3)
    original_query_embedding = vertex_embeddings_model.embed_query(original_query)
    augmented_query_embedding = vertex_embeddings_model.embed_query(joint_query)

    projected_original_query_embedding = project_embeddings([original_query_embedding], umap_transform)
    projected_augmented_query_embedding = project_embeddings([augmented_query_embedding], umap_transform)
    projected_retrieved_embeddings = project_embeddings(retrieved_docs3_embeddings, umap_transform)

    plt.figure()
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
    plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
    plt.scatter(projected_original_query_embedding[:, 0], projected_original_query_embedding[:, 1], s=150, marker='X', color='r')
    plt.scatter(projected_augmented_query_embedding[:, 0], projected_augmented_query_embedding[:, 1], s=150, marker='X', color='orange')

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{original_query}')
    plt.axis('off')
    plt.show()



    # query it
    # query = "List 3 easy code challenges about Array."
    # docs = vector_chromadb.similarity_search(query)
    # print(docs[0].page_content)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    llm = VertexAI(model_name="gemini-pro", temperature=0.1)
    # memory = ConversationSummaryMemory(
    #     llm=llm, memory_key="chat_history", return_messages=True
    # )

    # This controls how the standalone question is generated.
    # Should take `chat_history` and `question` as input variables.
    condense_question_template = """Given the following chat history and a follow up question,\
    rephrase the follow up input question to be a standalone question.
    
    Chat History:####
    {chat_history}
    ####

    Follow Up Input: ####
    {question}
    ####
    
    Standalone question:"""

    condense_question_prompt = PromptTemplate.from_template(template=condense_question_template)

    combine_docs_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    
    
    If the chat history exists, and hence it is not empty, then USE IT to support your answer along with the context.
    Otherwise, just use the context and add to your final answer that there is not any chat history.
    
    {chat_history}
    
    Question: {question}
    Helpful Answer:"""

    combine_docs_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=combine_docs_template
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_chromadb.as_retriever(search_type="similarity", search_kwargs={"k": 1}),  # mmr
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        # return_source_documents=True,
        # return_generated_question=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
        verbose=True
    )

    # chat_history = []

    question = "Can you give me 3 easy code challenges about arrays?"
    result = qa({"question": question})
    # chat_history.extend([(question, result["answer"])])
    # gen_question = result["generated_question"]
    # src_docs = result["source_documents"]
    answer = result['answer']

    # print(qa)
    # print(result)
    print(result['answer'])

    question = "List only the titles."
    result = qa({"question": question})
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
