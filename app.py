import re
import math
import time
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

    START_DATE = "Jan 2023"
    END_DATE = "Dec 2024"

    formula_map = {
        "Promotion (Promo) Credits": "Gross revenue - Net revenue",

        "Daily Run Rate (DRR)": "Last 7 days revenue / 7 OR Last 14 days revenue / 14 OR Any month revenue / Number of days in the month",

        "Monthly Run Rate (MRR)": "Last 90 days revenue / 3 ",
        "Monthly Run Rate (MRR) of a requested `MONTH`": "Last 3 months revenue from `MONTH` / 3 (eg: March 2024 MRR = ([january 2024] + [february 2024] + [march 2024]) / 3)",

        "Annual Run Rate (ARR)": f"(total {YEAR} quarter {QUARTER-1} revenue) * 4",
        "Incremental Run Rate (IRR)": f"(total {YEAR} quarter {QUARTER-1} revenue) * 4 - (total {YEAR-1} quarter 4 revenue) * 4",
        
        "Projected revenue": f"(total {YEAR} revenue) + (last `N` days revenue / `N`) * {DAYS_REMAINING}",

        "H1 growth": f"(total Jan {YEAR} to Jun {YEAR}  revenue) - (total Jan {YEAR-1} to Jun {YEAR-1} revenue)",
        "H2 growth": f"(total Jul {YEAR} to Dec {YEAR}  revenue) - (total Jul {YEAR-1} to Dec {YEAR-1} revenue)",
        "Year on Year (YoY) growth": f"(total {YEAR-1} revenue) - (total {YEAR} revenue)",
        "Month on Month (MoM) growth": f"({calendar.month_name[MONTH-2]} {YEAR} revenue) - ({calendar.month_name[MONTH-1]} {YEAR} revenue)",

        "New Billers": f"(total {YEAR-1} revenue) < 0 AND (total {YEAR} revenue) > 1",

        "Started billing/revenue IN the last `N` days": f"([Total revenue from {START_DATE} to {END_DATE}] - [current last `N` days revenue]) <= 0 AND [current last `N` days revenue] > 0",
        "Started billing/revenue IN or DURING a `MONTH`": f"([Total revenue from {START_DATE} to `MONTH`]) <= 0 AND [`MONTH` revenue] > 0",
        "Started billing/revenue AFTER a `MONTH`": f"([Total Revenue from {START_DATE} to `MONTH`]) <= 0 AND [Total Revenue from requested `MONTH` to {END_DATE}] > 0"
    }

    return formula_map

@st.cache_resource
def initialize():
    client = bigquery.Client(project=PROJECT_ID)
    df = client.query(f"SELECT * FROM {TABLE_ID}").to_dataframe()

    # table_ref = client.get_table(TABLE_ID)
    # data_schema = "\n".join([f"\t- {str(field.name)}: {str(field.description)}" for field in table_ref.schema])

    embed_model = SentenceTransformer("./all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_db = chroma_client.get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())

    return df, embed_model, chroma_db
   
LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"
MODEL_NAME = "gemini-1.5-flash-001"

DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

N_FORMULAS = 4
CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"

FORMULA_MAP = initialize_formulas()
df, EMBED_MODEL, CHROMA_DB = initialize()

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------------------------------------
#                                               Functions
# ---------------------------------------------------------------------------------------------------------

def generate_prompt(query_description, relevant_formulas):
    prompt = f"""Given a natural language question in English, write Python code that retrieves the relevant information from Pandas DataFrame `df` and returns a new DataFrame named `result`.
Write a basic code, do not use any unnecessary pandas functions or methods.
Include `reporting id` and `account name` columns by default, only omit them if absolutely unnecessary for the output.
Do not include any unwanted columns in the `result`.
Use proper column names based on the dataframe schema.
Utilize DATAFRAME COLUMNS and FORMULAS to write code.

# start DATAFRAME COLUMNS #
reporting_id: Unique id mapped to each account
account_name: Account name
micro_region: Micro Region (alias: mr, MR)
nal_cluster: NAL Cluster
nal_id: Unique id mapped to each NAL Name
nal_name: NAL Name
sub_region: Sub Region (alias: region)
segment: Account segment
account_type: account type values: GREENFIELD A (alias GF A), GREENFIELD B (GF B), SPENDER
exit_2023_last_7_days_net_revenue: net revenue of last 7 days of December 2023
exit_2023_last_14_days_net_revenue: net revenue of last 14 days of December 2023
exit_2023_last_90_days_net_revenue: net revenue of last 90 days of year 2023 from October 2023 to December 2023
exit_2023_last_7_days_gross_revenue: gross revenue of last 7 days of December 2023
exit_2023_last_14_days_gross_revenue: gross revenue of last 14 days of December 2023
exit_2023_last_90_days_gross_revenue: gross revenue of last 90 days of year 2023 from October 2023 to December 2023
current_last_7_days_net_revenue: net revenue of last 7 days
current_last_14_days_net_revenue: net revenue of last 14 days
current_last_90_days_net_revenue: net revenue of last 90 days
current_last_7_days_gross_revenue: gross revenue of last 7 days
current_last_14_days_gross_revenue: gross revenue of last 14 days
current_last_90_days_gross_revenue: gross revenue of last 90 days
[MONTH]_[YEAR]_net_revenue: net revenue for `MONTH` in `YEAR`. Total 24 columns (e.g january_2023_net_revenue)
[MONTH]_[YEAR]_gross_revenue: gross revenue for `MONTH` in `YEAR`. Total 24 columns (e.g january_2023_net_revenue)
# end DATAFRAME COLUMNS #

# start FORMULAS #
{relevant_formulas}
# end FORMULAS #

Now, write Python code without any additional text for: {query_description}"""

    return prompt

def get_relevant_formulas(question, n_results):
    rfs = CHROMA_DB.query(query_texts=[question], n_results=n_results)["documents"][0]
    rf_text = "\n".join([f"- {rf}: {FORMULA_MAP[rf]}" for rf in rfs])
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

    relevant_formulas = get_relevant_formulas(question, N_FORMULAS)
    st.write(relevant_formulas)
    try:
        prompt = generate_prompt(question, relevant_formulas)
        model_code = get_model_response(prompt)

        try:
            st.code(str(model_code).strip(), language="python")

            result = None
            exec(model_code, globals())
            
            if len(result) > 20:
                st.dataframe(data=result, use_container_width=True, height=400)
                st.info(f"Complete output has {len(result)} rows", icon="ℹ️")
            else:
                st.dataframe(data=result, use_container_width=True)

            result_tsv = data_to_tsv(result)
            st_copy_to_clipboard(text="result_tsv", before_copy_label="📋 copy data", after_copy_label="✅ copied data")
        except Exception as e:
            st.error(f"Trouble executing query!\n\n{e}", icon="🔍")
    except Exception as e:
        st.error(f"Trouble getting query from model!\n\n{e}", icon="🤖")