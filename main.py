import streamlit as st
import base64



from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma



# -----------------------
# LLM
# -----------------------
llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=st.secrets["OPENAI_API_KEY"]
)
# -----------------------
# Vector Store
# -----------------------
embeddings = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"]
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Aushadha AI",
    page_icon="🌿",
    layout="wide"
)

# -----------------------
# Background Image
# -----------------------
with open("logo.png", "rb") as image:
    encoded = base64.b64encode(image.read()).decode()

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: 30%;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stChatInput"] {{
        position: fixed;
        bottom: 20px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Title
# -----------------------
st.title("🌿 Aushadha Nature Essential Store")

st.caption(
    "Ask about oils, dry fruits, seeds, laddus and their benefits"
)

# -----------------------
# Chat History
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------
# User Input
# -----------------------
prompt = st.chat_input(
    "Ask me anything about Aushadha products..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    docs = vectorstore.similarity_search(
        prompt,
        k=4
    )

    context = "\n".join(
        doc.page_content for doc in docs
    )

    response = llm.invoke(
        f"""
You are the AI assistant for Aushadha Nature Essential Store.

Answer ONLY from the context provided.

If the answer is not found in the context, say:
'I could not find that information in the store database.'

Context:
{context}

Question:
{prompt}
"""
    )

    answer = response.content

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )