import os

from configs.app_configs import AppConfigs

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain.text_splitter import MarkdownHeaderTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import umap
import numpy as np
from tqdm import tqdm

app_configs = AppConfigs()
gcloud_credentials = app_configs.configs.GoogleApplicationCredentials.google_app_credentials_path
# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    *gcloud_credentials.dir_path,
    gcloud_credentials.file_name
)

persist_directory = os.path.join(*app_configs.configs.ChromaConfigs.persist_directory)
notion_directory = os.path.join(*app_configs.configs.NotionLocalDbConfigs.persist_directory)


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    loader = NotionDirectoryLoader(notion_directory)
    docs = loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    md_header_splits = []
    for doc in docs:
        md_header_splits += (markdown_splitter.split_text(doc.page_content))

    chunk_size, chunk_overlap = 600, 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    # Split
    text_splits = text_splitter.split_documents(md_header_splits)

    tokens_per_chunk = 200
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=tokens_per_chunk)
    token_split_texts = token_splitter.split_documents(text_splits)

    vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")

    if not os.listdir(persist_directory):
        vector_chromadb = Chroma.from_documents(
            documents=token_split_texts,
            persist_directory=persist_directory,
            embedding=vertex_embeddings)
    else:

        vector_chromadb = Chroma(
            persist_directory=persist_directory,
            embedding_function=vertex_embeddings
        )

    query = "List 3 easy code challenges about Array."
    retrieved_documents = vector_chromadb.similarity_search(query=query, k=5)

    for document in retrieved_documents:
        print(document.page_content)
        print('\n')

    embeddings = vector_chromadb._collection.get(include=['embeddings'])['embeddings']
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Projected Embeddings')
    plt.axis('off')
    plt.show()

    query = "List 3 easy code challenges about Array."
    # results = vector_chromadb._collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])
    # retrieved_documents = results['documents'][0]
    query_embedding = vertex_embeddings.embed_query(query)
    results = vector_chromadb.similarity_search(query=query, k=5)
    texts = []
    for res in results:
        texts.append(res.page_content)
    retrieved_embeddings = vertex_embeddings.embed_documents(texts)

    projected_query_embedding = project_embeddings([query_embedding], umap_transform)
    projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

    # Plot the projected query and retrieved documents in the embedding space
    plt.figure()
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
    plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
    plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{query}')
    plt.axis('off')
    plt.show()
    # print(vector_chromadb._collection.count())

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
