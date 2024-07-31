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

        "Daily Run Rate (DRR)": "Last `N` days revenue / `N` (`N`=7 & time period=current if not specified) OR Any month revenue / Number of days in the month",
        "Monthly Run Rate (MRR)": "Last 90 days revenue / 3 OR Last 3 months revenue from `MONTH` / 3 (eg: March 2024 MRR = ([january 2024] + [february 2024] + [march 2024]) / 3)",

        "Annual Run Rate (ARR)": f"(total {YEAR} Q{QUARTER-1} revenue) * 4",
        "Incremental Run Rate (Inc ARR)": f"(total {YEAR} Q{QUARTER-1} revenue) * 4 - (total {YEAR-1} Q4 revenue) * 4",

        "Year on Year (YoY)": f"(total {YEAR-1} revenue) - (total {YEAR} revenue)",
        "Month on Month (MoM)": f"({calendar.month_name[MONTH-2]} {YEAR} revenue) - ({calendar.month_name[MONTH-1]} {YEAR} revenue)",

        "Forecast/Projected revenue": f"(total {YEAR} revenue) + (last `N` days revenue / `N`) * {DAYS_REMAINING}",

        # "new billers IN or DURING a `MONTH`": f"([total revenue from {START_DATE} to `MONTH`]) <= 0 AND [`MONTH` revenue] > 0",
        "new billers IN `time_period`":  f"(total revenue from {START_DATE} to `time_period`) <= 0 AND (total revenue in `time_period`) >= 1",
        "new billers IN the last `N` days": f"([total revenue from {START_DATE} to {END_DATE}] - [{YEAR} last `N` days revenue]) <= 0 AND [{YEAR} last `N` days revenue] > 0",
    }

    return formula_map

@st.cache_resource
def initialize_rag():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    chroma_db = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())
    return embed_model, chroma_db

@st.cache_resource
def initialize_data():
    df = bigquery.Client(project=PROJECT_ID).query(f"SELECT * FROM {TABLE_ID}").to_dataframe()
    return df, df.columns.tolist()

LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"

EMBED_MODEL_NAME = "./embed_model"     #model: all-MiniLM-L6-v2
GEMINI_MODEL_NAME = "gemini-1.5-flash-001"

DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

N_FORMULAS = 4
CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"

df, DF_COLS = initialize_data()
FORMULA_MAP = initialize_formulas()
EMBED_MODEL, CHROMA_DB = initialize_rag()

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------------------------------------
#                                               Functions
# ---------------------------------------------------------------------------------------------------------

def generate_prompt(query_description, relevant_formulas):
    prompt = f"""Given a question, write Python code that retrieves information from Pandas DataFrame `df` and returns new DataFrame `result`.
Write a simple executable code, avoid using complex pandas functions.
Utilize COLUMNS & FORMULAS to write code.

# start COLUMNS #
reporting_id
account_name
micro_region
nal_cluster
nal_id
nal_name
segment
account_type
net_[month]_[year]: Net Revenue of month (net_01_2023 to net_12_2024)
gross_[month]_[year]: Gross Revenue of month (gross_01_2023 to gross_12_2024)
net_l[N]_[year]: Net Revenue of last N days (N=7,14,90 & year=2023,2024)
gross_l[N]_[year]: Gross Revenue of last N days (N=7,14,90 & year=2023,2024)
# end COLUMNS #

# start FORMULAS #
{relevant_formulas}
# end FORMULAS #

Write code without additional text for: {query_description}"""

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

            PLOT_COLS = []
            PRESENT_COLS = []
            for col in result.columns.tolist():
                if col not in DF_COLS:
                    PLOT_COLS.append(col)
                if col in ['account_name', 'micro_region', 'nal_cluster', 'nal_id', 'nal_name', 'segment', 'account_type']:
                    PRESENT_COLS.append(col)

            result = result[PRESENT_COLS + PLOT_COLS]
            st.dataframe(data=result, use_container_width=True, height=400)
            if len(result) > 20:
                st.info(f"Complete output has {len(result)} rows", icon="ℹ️")

            # result_tsv = data_to_tsv(result)
            # st_copy_to_clipboard(text="result_tsv", before_copy_label="📋 copy data", after_copy_label="✅ copied data")

            if PLOT_COLS and PRESENT_COLS:
                st.bar_chart(result.sort_values([PLOT_COLS[-1]], ascending=False).head(15), x=PRESENT_COLS[0], y=PLOT_COLS[-1])
            else:
                st.warning('The data cannot not be plotted', icon="⚠️")
        except Exception as e:
            st.exception(f"Trouble executing query!\n\n{e}")
    except Exception as e:
        st.error(f"Trouble getting query from model!\n\n{e}")