import re
import time
import pandas as pd
import streamlit as st
import bigframes.pandas as bpd
from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel

# ---------------------------------------------------------------------------------------------------------
#                                       GCP Config and Helper Functions
# ---------------------------------------------------------------------------------------------------------

LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"
MODEL_NAME = "gemini-1.5-flash-001"

DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

def get_dataset_schema(project_id, table_id):
    client = bigquery.Client(project=project_id)
    table_ref = client.get_table(TABLE_ID)

    data_schema = "COLUMN NAMES and DESCRIPTION:"
    for field in table_ref.schema:
        data_schema += "\n    " + str(field.name) + ": " + str(field.description)
    return data_schema

def generate_prompt(query_description, data_schema):
    prompt = f"""
Given a natural language question in English about a Pandas DataFrame df, write well-documented Python code that retrieves the relevant information and returns a new DataFrame named result. The code should:
Not modify the original DataFrame `df`, create a temporary DataFrame named `result` to hold any calculations or manipulations.
Always return a DataFrame even if the code results in no data, return an empty DataFrame named `result`.
Utilize DATAFRAME COLUMNS, KNOWLEDGE, and FORMULAS.

ADDITIONAL NOTES:
    - Do not return irrelevant columns that would slow down code execution.
    - Include `reporting id` and `account name` columns by default, only omit them if absolutely unnecessary for the answer.
    - Make sure to use proper column names based on the dataframe schema.
    - Feel free to use any Python DataFrame functions or operators that are necessary to retrieve the desired information.

DATAFRAME COLUMNS:
{data_schema}

KNOWLEDGE:
    a. DataFrame contains data from January 2023 onwards, so consider it as the STARTING POINT.
    b. DataFrame contains data till December 2024, so consider it as the ENDING POINT.
    c. Quarter Breakup:
        Quarter 1 (Q1) = Jan 1 - Mar 31
        Quarter 2 (Q2) = Apr 1 - Jun 30
        Quarter 3 (Q3) = Jul 1 - Sep 30
        Quarter 4 (Q4) = Oct 1 - Dec 31
    d. Criteria to get accounts that started billing:
        1. In the last `N` days = ([Total revenue from STARTING POINT] - [last `N` days revenue]) <= 0 AND [last `N` days revenue] > 0
        2. In or During a particular month = ([Total Revenue from STATING POINT to requested month]) <= 0 AND [requested month revenue] > 0
        3. After a particular month = ([Total Revenue from STATING POINT to requested month]) <= 0 AND [Total Revenue from requested month to ENDING POINT] > 0

FORMULAS:
    a. Promotion (Promo) Credits = Gross Revenue - Net Revenue
    b. Daily Run Rate (DRR) =
        Method 1. Last 7 days revenue / 7
        Method 2. Last 14 days revenue / 14
        Method 3. [Any month revenue] / [Number of days in the month]
    c. Monthly Run Rate (MRR) = [Last 90 days data] / 3
    d. Any Past month's Monthly Run Rate = [Last 3 data months from query MONTH] / 3
        - Example: March 2024 MRR = ([january 2024 revenue] + [february 2024 revenue] + [march 2024 revenue]) / 3
    e. Last year Closing DRR = [Last year last `N` days revenue] / `N`  Note: `N` can be 7 or 14
    f. Last year Closing MRR = [Last year last 90 days revenue] / 3
    g. DRR or MRR Growth = [Current DRR or MRR] - [Past DRR or MRR]

Now, write Python code without any additional text for: {query_description}
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

placeholder = st.empty()
with placeholder:
    with st.spinner(f"⏳ Setting up Project..."):
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        # bpd.options.bigquery.location = LOCATION
        # bpd.options.bigquery.project = PROJECT_ID

        # df = bpd.read_gbq(TABLE_ID).to_pandas()
        # bpd.close_session()

        DATA_SCHEMA = get_dataset_schema(PROJECT_ID, TABLE_ID)
placeholder.empty()

st.code(DATA_SCHEMA)
if question := st.chat_input("Ask a question"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    prompt = generate_prompt(question, DATA_SCHEMA)
    try:
        model_code = get_model_response(prompt)
        
        result = None
        try:
            exec(model_code, globals())
        except Exception as e:
            print(e)
        result_tsv = data_to_tsv(result)

        with st.chat_message("assistant"):
            st.code(str(model_code).strip(), language="python")
            
            if len(result) > 20:
                st.dataframe(data=result, use_container_width=True, height=400)
                st.info(f"Complete output has {len(result)} rows", icon="ℹ️")
            else:
                st.dataframe(data=result, use_container_width=True)

            st_copy_to_clipboard(text=result_tsv, before_copy_label="📋 copy data", after_copy_label="✅ copied data")
    except:
        st.error("Error getting query from model", icon="🤖")