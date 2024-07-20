import re
import math
import time
import chromadb
import datetime
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.cloud import aiplatform
from copy_to_clipboard import st_copy_to_clipboard
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction
from vertexai.preview.generative_models import GenerativeModel

# ---------------------------------------------------------------------------------------------------------
#                                       Initialization Global Variables
# ---------------------------------------------------------------------------------------------------------

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        return EMBED_MODEL.encode(input).tolist()

@st.cache_resource
def initialize():
    client = bigquery.Client(project=PROJECT_ID)

    df = client.query(f"SELECT * FROM {TABLE_ID}").to_dataframe()

    table_ref = client.get_table(TABLE_ID)
    data_schema = "\n".join([f"\t{str(field.name)}: {str(field.description)}" for field in table_ref.schema])

    formula_map = {
        "Promotion (Promo) Credits": "Gross Revenue - Net Revenue",
        "Daily Run Rate (DRR)": "Last 7 days revenue / 7 | Last 14 days revenue / 14 | [Any month revenue] / [Number of days in the month]",
        "Monthly Run Rate (MRR)": "[Last 90 days data] / 3 | [Last 3 data months from requested month] / 3 (eg: March 2024 MRR = ([january 2024 revenue] + [february 2024 revenue] + [march 2024 revenue]) / 3)",
        "Annual Run Rate (ARR)": "[PREVIOUS QUARTER] * 4",
        "Incremental Run Rate (IRR)": "[PREVIOUS QUARTER] * 4 - [last year Quarter 4] * 4",
        "Criteria to get accounts that started billing/revenue IN the last `N` days": "([Total revenue from STARTING DATE to ENDING DATE] - [last `N` days revenue from current date]) <= 0 AND [last `N` days revenue from current date] > 0",
        "Criteria to get accounts that started billing/revenue IN or DURING a `MONTH`": "([Total revenue from STARTING DATE to `MONTH`]) <= 0 AND [requested month revenue] > 0",
        "Criteria to get accounts that started billing/revenue AFTER a `MONTH`": "([Total Revenue from STARTING DATE to `MONTH`]) <= 0 AND [Total Revenue from requested month to ENDING DATE] > 0"
    }

    embed_model = SentenceTransformer("./all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_db = chroma_client.get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())

    return df, data_schema, formula_map, embed_model, chroma_db

@st.cache_data
def initialize_date():
    now = datetime.datetime.now()
    DAY = now.day
    MONTH = now.strftime("%B")
    YEAR = now.year
    QUARTER = f"Q{math.ceil(now.month / 3)}"
    return DAY, MONTH, YEAR, QUARTER
    
LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"
MODEL_NAME = "gemini-1.5-flash-001"

DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"

DAY, MONTH, YEAR, QUARTER = initialize_date()
df, DATA_SCHEMA, FORMULA_MAP, EMBED_MODEL, CHROMA_DB = initialize()

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------------------------------------
#                                           GCP Config Functions
# ---------------------------------------------------------------------------------------------------------

def generate_prompt(query_description, relevant_formulas):
    prompt = f"""Given a natural language question in English about a Pandas DataFrame df, write well-documented Python code that retrieves the relevant information and returns a new DataFrame named result

INSTRUCTIONS:
\tDo not modify the original DataFrame `df`, create a temporary DataFrame named `result` to hold any calculations or manipulations.
\tAlways return a DataFrame object even if the code results in no data, return an empty DataFrame named `result`.
\tInclude `reporting id` and `account name` columns by default, only omit them if absolutely unnecessary for the output.
\tDo not return irrelevant columns that would slow down code execution.
\tUse proper column names based on the dataframe schema.
\tUtilize DATAFRAME COLUMNS, KNOWLEDGE, and FORMULAS.

DATAFRAME COLUMNS:
{DATA_SCHEMA}

KNOWLEDGE:
\tCURRENT DATE: {DAY} {MONTH} {YEAR}
\tCURRENT QUARTER: {YEAR}-{QUARTER}
\tSTARTING DATE of the dataset is January 1 2023.
\tENDING DATE of the dataset is December 31, 2024.
\tQuarter Breakup:
\t\tQuarter 1 (Q1) = Jan - Mar
\t\tQuarter 2 (Q2) = Apr - Jun
\t\tQuarter 3 (Q3) = Jul - Sep
\t\tQuarter 4 (Q4) = Oct - Dec

FORMULAS:
{relevant_formulas}

Now, write Python code without any additional text for: {query_description}"""

    return prompt

def get_relevant_formulas(query, n_results):
    rfs = CHROMA_DB.query(query_texts=[question], n_results=2)["documents"][0]
    rf_text = "\n".join([f"\t{rf}: {FORMULA_MAP[rf]}" for rf in rfs])
    return rf_text

def get_model_response(prompt):
    model = GenerativeModel(model_name=MODEL_NAME)
    model_response = model.generate_content(prompt).text

    model_response = re.sub("```python", "", model_response)
    model_response = re.sub("```", "", model_response)
    model_response = model_response.strip()

    return model_response

def data_to_tsv(df):
    tsv_data = "\t".join(list(df.columns)) + "\n"
    df=df.astype(str)
    df = df.values.tolist()
    for row in df:
        tsv_data += "\t".join(row) + "\n"
    return tsv_data

# ---------------------------------------------------------------------------------------------------------
#                                             Streamlit UI
# ---------------------------------------------------------------------------------------------------------

st.title(f"GCP Query")

if question := st.chat_input("Ask a question"):
    with st.chat_message("user"):
        st.markdown(question)

    relevant_formulas = get_relevant_formulas(question, 2)

    with st.chat_message("assistant"):
        st.markdown(relevant_formulas)
    try:
        prompt = generate_prompt(question, relevant_formulas)
        model_code = get_model_response(prompt)

        try:
            st.code(str(model_code).strip(), language="python")

            result = None
            exec(model_code, globals())
            result_tsv = data_to_tsv(result)

            with st.chat_message("assistant"):
                if len(result) > 20:
                    st.dataframe(data=result, use_container_width=True, height=400)
                    st.info(f"Complete output has {len(result)} rows", icon="ℹ️")
                else:
                    st.dataframe(data=result, use_container_width=True)
                st_copy_to_clipboard(text="result_tsv", before_copy_label="📋 copy data", after_copy_label="✅ copied data")
        except Exception as e:
            st.error(f"Trouble executing query!\n\n{e}", icon="🔍")
    except Exception as e:
        st.error(f"Trouble getting query from model!\n\n{e}", icon="🤖")