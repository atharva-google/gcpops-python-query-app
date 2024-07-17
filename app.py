import re
import math
import time
import datetime
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.cloud import aiplatform
from copy_to_clipboard import st_copy_to_clipboard
from vertexai.preview.generative_models import GenerativeModel

# ---------------------------------------------------------------------------------------------------------
#                                       GCP Config and Helper Functions
# ---------------------------------------------------------------------------------------------------------

LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"
MODEL_NAME = "gemini-1.5-flash-001"

DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

@st.cache_resource
def initialize():
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.get_table(TABLE_ID)
    data_schema = "COLUMN NAMES and DESCRIPTION:"
    for field in table_ref.schema:
        data_schema += "\n    " + str(field.name) + ": " + str(field.description)
    df = client.query(f"SELECT * FROM {TABLE_ID}").to_dataframe()
    return df, data_schema

@st.cache_data
def initialize_date():
    now = datetime.datetime.now()
    DAY = now.day
    MONTH = now.strftime("%B")
    YEAR = now.year
    QUARTER = f"Q{math.ceil(now.month / 3)}"
    return DAY, MONTH, YEAR, QUARTER
    
df, DATA_SCHEMA = initialize()
DAY, MONTH, YEAR, QUARTER = initialize_date()
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def generate_prompt(query_description):
    prompt = f"""
Given a natural language question in English about a Pandas DataFrame df, write well-documented Python code that retrieves the relevant information and returns a new DataFrame named result

INSTRUCTIONS:
    - Do not modify the original DataFrame `df`, create a temporary DataFrame named `result` to hold any calculations or manipulations.
    - Always return a DataFrame object even if the code results in no data, return an empty DataFrame named `result`.
    - Include `reporting id` and `account name` columns by default, only omit them if absolutely unnecessary for the output.
    - Do not return irrelevant columns that would slow down code execution.
    - Use proper column names based on the dataframe schema.
    - Use any Python DataFrame functions or operators as necessary to retrieve the desired information.
    - Utilize DATAFRAME COLUMNS, KNOWLEDGE, and FORMULAS.

DATAFRAME COLUMNS:
{DATA_SCHEMA}

KNOWLEDGE:
    a. CURRENT DATE: {DAY} {MONTH} {YEAR}
    b. CURRENT QUARTER: {YEAR}-{QUARTER}
    b. STARTING DATE of the dataset is January 1 2023.
    c. ENDING DATE of the dataset is December 31, 2024.
    d. Quarter Breakup:
        Quarter 1 (Q1) = Jan 1 - Mar 31
        Quarter 2 (Q2) = Apr 1 - Jun 30
        Quarter 3 (Q3) = Jul 1 - Sep 30
        Quarter 4 (Q4) = Oct 1 - Dec 31

FORMULAS:
    a. Promotion (Promo) Credits = Gross Revenue - Net Revenue
    b. Daily Run Rate (DRR) =
        Method 1. Last 7 days revenue / 7
        Method 2. Last 14 days revenue / 14
        Method 3. [Any month revenue] / [Number of days in the month]
    c. Monthly Run Rate (MRR) = [Last 90 days data] / 3
    d. Any Past month's MRR = [Last 3 data months from requested month] / 3
        - Example: March 2024 MRR = ([january 2024 revenue] + [february 2024 revenue] + [march 2024 revenue]) / 3
    e. Last year Closing DRR = [Last year last `N` days revenue] / `N`  Note: `N` can be 7 or 14
    f. Last year Closing MRR = [Last year last 90 days revenue] / 3
    g. DRR or MRR Growth = [Current DRR or MRR] - [Past DRR or MRR]
    h. Criteria to get accounts that started billing:
        1. In the last `N` days = ([Total revenue from STARTING DATE to ENDING DATE] - [last `N` days revenue from current date]) <= 0 AND [last `N` days revenue from current date] > 0
        2. In or during a `MONTH` = ([Total revenue from STARTING DATE to `MONTH`]) <= 0 AND [requested month revenue] > 0
        3. After a `MONTH` = ([Total Revenue from STARTING DATE to `MONTH`]) <= 0 AND [Total Revenue from requested month to ENDING DATE] > 0
    i. Criteria to get accounts that started revenue: same as h.

Now, write Python code without any additional text for:
{query_description}
"""
    return prompt

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

    try:
        prompt = generate_prompt(question)
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