import re
import math
import calendar 
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

@st.cache_data
def initialize_formulas():
    NOW = datetime.datetime.now()
    YEAR = NOW.year
    MONTH = NOW.month
    QUARTER = math.ceil(NOW.month / 3)
    DAYS_REMAINING = (datetime.date(YEAR, 12, 31) - datetime.date.today()).days

    START_DATE = "01 2023"
    END_DATE = "12 2024"

    formula_map = {
        "Promotion (Promo) Credits": "Gross revenue - Net revenue",

        "Daily Run Rate (DRR)": "Last `N` days revenue / `N` OR Any month revenue / Number of days in the month",
        "Monthly Run Rate (MRR)": "Last 90 days revenue / 3 OR Last 3 months revenue from `MONTH` / 3 (eg: March 2024 MRR = ([january 2024] + [february 2024] + [march 2024]) / 3)",

        "Annual Run Rate (ARR)": f"(total {YEAR} Q{QUARTER-1} revenue) * 4",
        "Incremental Run Rate (IRR)": f"(total {YEAR} Q{QUARTER-1} revenue) * 4 - (total {YEAR-1} Q4 revenue) * 4",

        "Forecast/Projected revenue": f"(total {YEAR} revenue) + (last `N` days revenue / `N`) * {DAYS_REMAINING}",

        "H1 growth": f"(total 01 {YEAR} to 06 {YEAR} revenue) - (total 01 {YEAR-1} to 06 {YEAR-1} revenue)",
        "H2 growth": f"(total 07 {YEAR} to 12 {YEAR} revenue) - (total 07 {YEAR-1} to 12 {YEAR-1} revenue)",
        "Year on Year (YoY) growth": f"(total {YEAR-1} revenue) - (total {YEAR} revenue)",
        "Month on Month (MoM) growth": f"({calendar.month_name[MONTH-2]} {YEAR} revenue) - ({calendar.month_name[MONTH-1]} {YEAR} revenue)",

        "New Billers": f"(total {YEAR-1} revenue) < 0 AND (total {YEAR} revenue) > 1",

        "billing/revenue IN the last `N` days": f"([total revenue from {START_DATE} to {END_DATE}] - [last `N` {YEAR} days revenue]) <= 0 AND [current last `N` days revenue] > 0",
        "billing/revenue IN or DURING a `MONTH`": f"([total revenue from {START_DATE} to `MONTH`]) <= 0 AND [`MONTH` revenue] > 0",
        "billing/revenue AFTER a `MONTH`": f"([total Revenue from {START_DATE} to `MONTH`]) <= 0 AND [Total Revenue from requested `MONTH` to {END_DATE}] > 0"
    }

    return formula_map

@st.cache_resource
def initialize_project():
    df = bigquery.Client(project=PROJECT_ID).query(f"SELECT * FROM {TABLE_ID}").to_dataframe()

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    chroma_db = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())

    return df, embed_model, chroma_db
   
LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"

EMBED_MODEL_NAME = "./all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-1.5-flash-001"

DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

N_FORMULAS = 4
CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"

FORMULA_MAP = initialize_formulas()
df, EMBED_MODEL, CHROMA_DB = initialize_project()

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------------------------------------
#                                               Functions
# ---------------------------------------------------------------------------------------------------------

def generate_prompt(query_description, relevant_formulas):
    prompt = f"""Given a question, write Python code that retrieves the relevant information from Pandas DataFrame `df` and returns a new DataFrame `result`.
Write a simple code without any pandas sort, groupby, aggregate functions.
Always create new columns to store arithmetic calculations.
Always return `INFO` and `calculated` columns in `result`, omit any if absolutely irrelevant.
Utilize DATAFRAME COLUMNS and FORMULAS to write code.

# start DATAFRAME COLUMNS #
INFO columns:
reporting_id
account_name
micro_region
nal_cluster
nal_id
nal_name
segment
account_type

Other columns:
net_[month]_[year]: Net Revenue of month (net_01_2023 to net_12_2024)
gross_[month]_[year]: Gross Revenue of month (gross_01_2023 to gross_12_2024)
net_l[N]_[year]: Net Revenue of last N days (N = 7, 14, 90)
gross_l[N]_[year]: Gross Revenue of last N days (N = 7, 14, 90)
# end DATAFRAME COLUMNS #

# start FORMULAS #
{relevant_formulas}
# end FORMULAS #

Write Python code without any additional text for: {query_description}"""

    return prompt

def get_relevant_formulas(question, n_results):
    rfs = CHROMA_DB.query(query_texts=[question], n_results=n_results)["documents"][0]
    rf_text = "\n".join([f"- {rf}: {FORMULA_MAP[rf]}" for rf in rfs])
    return rf_text

def get_model_response(prompt):
    model = GenerativeModel(model_name=GEMINI_MODEL_NAME)
    model_response = model.generate_content(prompt).text

    model_response = re.sub("```python", "", model_response)
    model_response = re.sub("```", "", model_response)
    model_response = model_response.strip()

    return model_response

def data_to_tsv(out_df):
    tsv_data = "\t".join(list(out_df.columns)) + "\n"
    out_df = out_df.astype(str)
    out_df = out_df.values.tolist()
    for row in out_df:
        tsv_data += "\t".join(row) + "\n"
    return tsv_data

# ---------------------------------------------------------------------------------------------------------
#                                             Streamlit UI
# ---------------------------------------------------------------------------------------------------------

st.title(f"GCP Query")
if question := st.chat_input("Ask a question"):
    with st.chat_message("user"):
        st.markdown(question)

    relevant_formulas = get_relevant_formulas(question, N_FORMULAS)
    try:
        prompt = generate_prompt(question, relevant_formulas)
        model_code = get_model_response(prompt)

        try:
            result = None
            exec(model_code, globals())
            
            st.dataframe(data=result, use_container_width=True, height=400)
            if len(result) > 20:
                st.info(f"Complete output has {len(result)} rows", icon="ℹ️")
            result_tsv = data_to_tsv(result)
            st_copy_to_clipboard(text="result_tsv", before_copy_label="📋 copy data", after_copy_label="✅ copied data")
            
            # columns = list(result.columns)
            # with st.expander("Chart Options"):
            #     x_options = st.multiselect("Data to plot on X axis", columns, max_selections=1)
            #     y_options = st.multiselect("Data to plot on Y axis", columns, max_selections=1)
        except Exception as e:
            st.exception(f"Trouble executing query!\n\n{e}")
    except Exception as e:
        st.error(f"Trouble getting query from model!\n\n{e}")