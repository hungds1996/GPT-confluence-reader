import streamlit as st
import os, json, time
from dotenv import load_dotenv

from confluence_qa import ConfluenceQA

# ATATT3xFfGF0wO25NUzyM331LwOJaCqvm6kKvvuxtQS1BWRR_GWYssVmnNYTLxaHb-GaHRCO6yFlyUF3IHB31yteCm0ZOt80kzZj-zf4CS7WmgIA6NbLap-QU5gO582aN7kp5B3PdDMdv3Doq1e1xVllKBsUCTMiizE2G2n-9AdamfDf43cEwSk=E8F6DE34

try:
    from hyperplane.utils import is_jhub

    if is_jhub():
        openaiKeyFile = "/root/.secret/openai_key.json"
    else:
        openaiKeyFile = "/etc/hyperplane/secrets/openai_key.json"
    with open(openaiKeyFile) as f:
        os.environ["OPENAI_API_KEY"] = json.load(f)["openai_key"]
except Exception as e:
    print(e)
    load_dotenv()

st.set_page_config(
    page_title="Document reader Bot", layout="wide", initial_sidebar_state="auto"
)

if "config" not in st.session_state:
    st.session_state["config"] = {}
if "confluence_qa" not in st.session_state:
    st.session_state["confluence_qa"] = None


@st.cache_resource
def load_confluence(config):
    confluence_qa = ConfluenceQA(config=config)
    confluence_qa.init_embeddings()
    confluence_qa.init_models()
    confluence_qa.vector_db_confluence_docs()
    confluence_qa.retrieval_qa_chain()
    return confluence_qa


with st.sidebar.form(key="Form1"):
    st.markdown("ADD CONFIGS")
    confluence_url = st.text_input(
        "paste the confluence URL", "https://example.atlassian.net/wiki/"
    )
    username = st.text_input(
        label="Confluence username (email)",
        value="",
        type="password",
    )
    space_key = st.text_input(label="confluence space name", value="SIO")
    api_key = st.text_input(label="confluence api key", type="password")

    config_submit = st.form_submit_button(label="save config")

    if config_submit and confluence_url and space_key:
        st.session_state["config"] = {
            "persist_directory": None,
            "confluence_url": confluence_url,
            "username": username if username != "" else None,
            "api_key": api_key if api_key != "" else None,
            "space_key": space_key,
        }
        with st.spinner(text="Crawling confluence data..."):
            config = st.session_state["config"]
            st.session_state["config"] = config
            st.session_state["confluence_qa"] = load_confluence(
                st.session_state["config"]
            )

        st.write("Done crawling")

st.title("Q&A demo")
question = st.text_input(
    "Ask a question about the document", "Describe the first iteration?"
)

if st.button("Get answer", key="button2"):
    with st.spinner(text="Asking bot.."):
        confluence_qa = st.session_state.get("confluence_qa")
        if confluence_qa is not None:
            result = confluence_qa.answer_confluence(question)
            st.write(result)
        else:
            st.write("Load Confluence page first")
