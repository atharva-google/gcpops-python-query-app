import re
import json
import math
import calendar
import chromadb
import datetime
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.cloud import aiplatform
# from copy_to_clipboard import st_copy_to_clipboard
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

st.set_page_config(page_icon="🐬", page_title="S&O DataDiver")

# ---------------------------------------------------------------------------------------------------------
#                                             Current Date
# ---------------------------------------------------------------------------------------------------------

NOW = datetime.datetime.now()
YEAR = NOW.year
MONTH = NOW.month
QUARTER = math.ceil(NOW.month / 3)
DAYS_REMAINING = (datetime.date(YEAR, 12, 31) - datetime.date.today()).days

START_YEAR = 2023
END_YEAR = 2024

# ---------------------------------------------------------------------------------------------------------
#                                              GCP Setup
# ---------------------------------------------------------------------------------------------------------

LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"
MODEL_NAME = ""

DATASET_ID = f"{PROJECT_ID}.gcp_data"
REVENUE_TABLE_ID = f"{DATASET_ID}.revenue"
ACCOUNTS_TABLE_ID = f"{DATASET_ID}.accounts"

DATA_SCHEMA = f"""  Dataset: {DATASET_ID}
  Tables:
    * {ACCOUNTS_TABLE_ID}: [account_id, name, segment, type, nal_id, nal_name, nal_cluster, micro_region]
    * {REVENUE_TABLE_ID}: [account_id, date, net_revenue, gross_revenue]"""

bq_client = bigquery.Client(project=PROJECT_ID)
bq_job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100_000_000)

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------------------------------------
#                                             Initialize Chroma
# ---------------------------------------------------------------------------------------------------------

CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./embed_model"     # model: all-MiniLM-L6-v2

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        return EMBED_MODEL.encode(input).tolist()

@st.cache_data
def initialize_formulas():
    AVAILABLE_FORMULAS = {
        "drr": {"name": "Daily Run Rate", "text": f"- current: last N days rev / N (default N = 7)\n- past month: that month's rev / days in month"},
        "mrr": {"name": "Monthly Run Rate", "text": f"current month: last 90 days rev / 3\n- past month: last 3 months rev / 3"},
        "arr": {"name": "Annual Run Rate", "text": f"- method: rev(Period) * Number of periods in year\n- default: Current DRR * 365"},
        "inc arr": {"name": "Incremental Annual Run Rate", "text": f"- method: (Latest Period rev - Past Period rev) * Number of periods in year\n- explanation: rev growth between period * number of periods in year\n- default: (Current DRR - {YEAR-1} Dec DRR) * 365"},
        "revevnue gain loss": {"name": "Revenue Gain Loss", "text": f"- formula: rev(Period X) - rev(Period X-1)\n- explanation: rev growth between two consecutive periods\n- default: current last 7 days - last 7 days prior to current last 7 days"},
        "forecast": {"name": "Projected Revenue", "text": f"- method: (Year-to-Date rev) + rev(Period) * Number of Periods in year\n- explanation: estimated year end rev by adding the rev earned so far with a projected amount based on a given period.\n- default: ({YEAR} Year-to-Date rev) + (DRR * {DAYS_REMAINING})"},
        # "new billers": {"name": "Get accounts that started revenue for a time period", "text": f"- criteria: check total rev from the beginning of data to just before the given period is < 1 AND total revenue after or during the given period is > 1\n- default period: last 7 days\n- example to find new billers:\n  * in june 2024: (total rev from start to may 2024) < 1 AND (total rev in june 2024) > 1\n  * after june 2024: (total rev from start to june 2024) < 1 AND (total rev from june 2024 to current date) > 1"}
    }
    return AVAILABLE_FORMULAS

@st.cache_resource
def initialize_rag():
    embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    chroma_db = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())
    return embed_model, chroma_db

AVAILABLE_FORMULAS = initialize_formulas()
EMBED_MODEL, CHROMA_DB = initialize_rag()

# ---------------------------------------------------------------------------------------------------------
#                                                  Agents
# ---------------------------------------------------------------------------------------------------------

def QueryWriter(question, knowledge):
    prompt = f"""Write a simple BigQuery code to get data for the given question.
Use appropriate `Data Schema` to write the code. 
Use `Knowledge` as context. If it does have proper context, use your own thinking.
Do not include additional text in your response.

Question: {question}

Data Schema:
{DATA_SCHEMA}

Knowledge:
{knowledge}"""

    try:
        model_response = GenerativeModel("gemini-1.5-pro", 
                                         generation_config={"temperature": 0}
                                         ).generate_content(prompt).text
        model_response = (model_response.replace("```sql", "").replace("```", "").strip())
    except:
        model_response = None
        # print("Error getting response from Model\n")

    return model_response

def QueryValidator(generated_query):
    try:
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

        query_job = bq_client.query(generated_query,job_config=job_config)
        exec_result = ("This query will process {} bytes.".format(query_job.total_bytes_processed))

        return True, exec_result
    except Exception as e:
        return False, str(e)

def QueryDebugger(generated_query, error):
    prompt = f"""Act as a BigQuery Debugger.
Given a Query and the Error, write a new query to fix the error.
Use appropriate `Data Schema` to write the code.
Make sure the logic in the original query remains the same.
Do not include additional text in your response.

Data Schema:
{DATA_SCHEMA}

Query:
{generated_query}

Error:
{error}"""

    model_response = GenerativeModel("gemini-1.5-pro", generation_config={"temperature": 0}).generate_content(prompt).text
    model_response = (model_response.replace("```sql", "").replace("```", "").strip())
    # except:
    #     model_response = None
        # print("Error getting response from Model\n")

    return model_response

def QueryValidatorAndDebugger(generated_query, n_attempts=3):
    attempt_counter = n_attempts

    while attempt_counter > 0:
        validator_response = QueryValidator(generated_query)

        if validator_response[0]:
            return True, generated_query
        else:
            # print("Calling Debugger")
            generated_query = QueryDebugger(generated_query, validator_response[1])
            n_attempts -= 1

    return False, None

def DataVisualizer(columns):
    col_text = "[" + ", ".join(columns) + "]"

    prompt = f"""Analyze the given Data columns: {col_text}
Help me choose the right visualization to represent the data.
Chart can be either bar or Line Chart.
Return a JSON reponse in the following format: {{ "chart_type": "bar" or "line", "x_column": [column to plot on x-axis], "y_columns": [column(s) to plot on y-axis] }}
Do not include additional text in your response."""

    try:
        model_response = GenerativeModel("gemini-1.5-flash", 
                                         generation_config={"temperature": 0}
                                         ).generate_content(prompt).text
    except:
        model_response = None
        # print("Error getting response from Model\n")

    return model_response

def DataSummarizer(data):
    data = data.head(100)
    csv_data = ",".join(list(data.columns)) + "\n"

    data = data.astype(str)
    data = data.values.tolist()
    for row in data:
        csv_data += ",".join(row) + "\n"

    prompt = f"""Analyze the given csv data and generate a bullet point summary for it.
Do not just return data values. Give a detailed high level summary of the data.
Use vivid language and emojis in your response.

Data:
{csv_data}"""

    try:
        model_response = GenerativeModel("gemini-1.5-flash", 
                                         generation_config={"temperature": 1}
                                         ).generate_content(prompt).text
    except:
        model_response = None
        # print("Error getting response from Model\n")

    return model_response

# ---------------------------------------------------------------------------------------------------------
#                                             Streamlit UI
# ---------------------------------------------------------------------------------------------------------

N_RESULTS = 2
N_DEBUG_ATTEMPTS = 3

st.title(f"DataDiver 🐬")

if question := st.chat_input("Ask a question"):
    with st.chat_message("user"):
            st.markdown(question)

    data_tab, summary_tab, graph_tab = st.tabs(["🗃️ Data", "🔍 Summary", "📈 Graph"])

    relevant_formulas = CHROMA_DB.query(query_texts=[question], n_results=N_RESULTS)["documents"][0]
    rf_text = "\n".join([f"{AVAILABLE_FORMULAS[fid.split(':')[1]]['name']} ({fid.split(':')[1]}):" + "\n" + "\n".join(["  " + line for line in AVAILABLE_FORMULAS[fid.split(':')[1]]['text'].split("\n")]) for fid in relevant_formulas])

    query = QueryWriter(question, rf_text)
    

    if query:
        is_valid, valid_query = QueryValidatorAndDebugger(query, n_attempts=3)

        if is_valid:
            try:
                df = bq_client.query(valid_query).to_dataframe()

                with data_tab:
                    st.dataframe(df)

                visualizer_response = DataVisualizer(df.columns.tolist())
                visualizer_response = (visualizer_response.replace("```json", "").replace("```", "").strip())
                visualizer_response = json.loads(visualizer_response)

                with graph_tab:
                    if visualizer_response["chart_type"] == "bar":
                        st.bar_chart(df.head(10), x=visualizer_response["x_column"], y=visualizer_response["y_columns"])
                    elif visualizer_response["chart_type"] == "line":
                        st.line_chart(df.head(10), x=visualizer_response["x_column"], y=visualizer_response["y_columns"])
                    else:
                        st.warning("Unknown Chart Type!")

                summarizer_response = DataSummarizer(df)

                with summary_tab:
                    st.write(summarizer_response)

            except Exception as e:
                error = "Trouble geting data from BigQuery Client"
                with data_tab:
                    st.error(error)
                    st.exception(e)
                with summary_tab:
                    st.error(error)
                    st.exception(e)
                with graph_tab:
                    st.error(error)
                    st.exception(e)
        else:
            error = f"Model unable to write valid Query for the question: {question}"
            with data_tab:
                st.error(error)
            with summary_tab:
                st.error(error)
            with graph_tab:
                st.error(error)
    else:
        error = "Error getting Query from model"
        with data_tab:
            st.error(error)
        with summary_tab:
            st.error(error)
        with graph_tab:
            st.error(error)