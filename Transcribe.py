import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain.callbacks import StreamlitCallbackHandler

st.title("PaLM Model Research Paper Query Assistant")

query = st.text_input("Enter your query:", placeholder="Ask me whatever ya can?")

@st.cache_resource
def initialize_models():
    hf_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    hf_token_api = st.secrets["HF_TOKEN"]
    
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/QwQ-32B-Preview",
        task="text-generation",
        max_new_tokens=4096,
        do_sample=False,
        repetition_penalty=1.02,
        huggingfacehub_api_token=hf_token_api,
    )
    
    ASTRA_DB_TOKEN = st.secrets["ASTRA_DB_TOKEN"]
    store = AstraDBGraphVectorStore(
        embedding=hf_embedding,
        token=ASTRA_DB_TOKEN,
        api_endpoint="https://c76a50e9-c364-419c-9961-ecf714d2f7b2-us-east-2.apps.astra.datastax.com",
        collection_name="Artificial_Intelligence"
    )
    
    return hf_embedding, llm, store

hf_embedding, llm, store = initialize_models()

callback_container = st.container()
st_callback = StreamlitCallbackHandler(parent_container=callback_container)

template = """
You are a highly knowledgeable assistant that provides detailed, structured, and organized answers in response to queries.
You are given a query by the user : {query}
Follow the below instructions:
### Instructions:
1. Provide the response in well-defined sections (Overview, Implementation Details, Data Overview, Ethical Considerations, Usage and Performance, etc.).
2. Include tabular format for any data that is better represented in a table (e.g., model sizes, datasets, or hardware details).
3. Do not display the query text in the response.
### Context:
{context}

### Structured Response:
"""

custom_rag_prompt = PromptTemplate.from_template(template)

if query:
    st.write("### Generating Response...")

    retriever = store.as_retriever(search_type="mmr_traversal", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    formatted_prompt = custom_rag_prompt.format(context=context,query = query)

    try:
        response = llm.invoke(formatted_prompt)
        st.markdown(response) 
    except Exception as e:
        st.error(f"An error occurred: {e}")
