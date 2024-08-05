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
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction
from vertexai.preview.generative_models import GenerativeModel

# ---------------------------------------------------------------------------------------------------------
#                                             Base Functions
# ---------------------------------------------------------------------------------------------------------

NOW = datetime.datetime.now()
YEAR = NOW.year
MONTH = NOW.month
QUARTER = math.ceil(NOW.month / 3)
DAYS_REMAINING = (datetime.date(YEAR, 12, 31) - datetime.date.today()).days

START_YEAR = 2023
END_YEAR = 2024

def promo(df, last_n_days=7, month=None, year=YEAR):
    if month:
        df[f"{month:02d}_{year}_promo_credits"] = df[f"gross_{month:02d}_{YEAR}"] - df[f"net_{month:02d}_{YEAR}"]
    else:
        df[f"l{last_n_days}_promo_credits"] = df[f"gross_l{last_n_days}_{YEAR}"] - df[f"gross_l{last_n_days}_{YEAR}"]
    return df

def drr(df, last_n_days=7, month=None, year=YEAR, rev_type="net"):
    if month and month != MONTH:
        num_days = calendar.monthrange(year, month)[1]
        df[f"{month:02d}_{year}_{rev_type}_DRR"] = df[f"{rev_type}_{month:02d}_{YEAR}"] / num_days
    else:
        df[f"current_l{last_n_days}d_{rev_type}_DRR"] = df[f"{rev_type}_l{last_n_days}_{YEAR}"] / last_n_days
    return df

def mrr(df, month=None, year=YEAR, rev_type="net"):
    if month and month != MONTH:
        df[f"{month:02d}_{year}_{rev_type}_MRR"] = 0

        month_count = 0
        curr_year = year
        curr_month = month
        for i in range(3):
            try:
                df[f"{month:02d}_{year}_{rev_type}_MRR"] += df[f"{rev_type}_{curr_month:02d}_{curr_year}"]
                curr_month -= 1
                if curr_month == 0:
                    curr_year -= 1
                    curr_month = 12
                month_count += 1
            except:
                break

        df[f"{month:02d}_{year}_{rev_type}_MRR"] /= month_count
    else:
        df[f"{MONTH:02d}_{year}_{rev_type}_MRR"] = df[f"{rev_type}_l90_{YEAR}"] / 90
    return df

def arr(df, quarter=QUARTER-1, year=YEAR, rev_type="net"):
    df[f"{year}_Q{quarter}_{rev_type}_ARR"] = 0
    for month in range(quarter*3-2, quarter*3+1):
        df[f"{year}_Q{quarter}_{rev_type}_ARR"] += df[f"{rev_type}_{month:02d}_{year}"]
    df[f"{year}_Q{quarter}_{rev_type}_ARR"] *= 4
    return df

def inc_arr(df, quarter=QUARTER-1, year=YEAR, rev_type="net"):
    df = arr(df, year=year-1, quarter=4, rev_type=rev_type)
    df = arr(df, quarter, rev_type=rev_type)
    df[f"{year}_Q{quarter}_{rev_type}_Inc_ARR"] = df[f"{year}_Q{quarter}_{rev_type}_ARR"] - df[f"{year-1}_Q4_{rev_type}_ARR"]
    df.drop([f"{year}_Q{quarter}_{rev_type}_ARR", f"{year-1}_Q4_{rev_type}_ARR"], axis=1, inplace=True)
    return df

def rev_from_to(df, start_month, start_year=YEAR, end_month=MONTH, end_year=YEAR, rev_type="net"):
    df[f"{start_month:02d}_{start_year}_to_{end_month:02d}_{end_year}_{rev_type}_rev"] = 0

    if start_year > end_year:
        pass # start > end
    elif start_year == end_year and start_month > end_month:
        pass # start > end
    else:
        for year in range(start_year, end_year+1):
            for month in range(1, 13):
                if year == start_year:
                    if month >= start_month:
                        df[f"{start_month:02d}_{start_year}_to_{end_month:02d}_{end_year}_{rev_type}_rev"] += df[f"{rev_type}_{month:02d}_{year}"]
                else:
                    df[f"{start_month:02d}_{start_year}_to_{end_month:02d}_{end_year}_{rev_type}_rev"] += df[f"{rev_type}_{month:02d}_{year}"]

                if year == end_year and month == end_month:
                    break
            if year == end_year:
                break
    return df

def qtd(df, quarter=QUARTER, year=YEAR, rev_type="net"):
    df = rev_from_to(df, start_month=quarter*3-2, start_year=year, end_month=quarter*3, end_year=year)
    df.rename(columns={f"{(quarter*3-2):2d}_{year}_to_{(quarter*3):02d}_{year}_{rev_type}_rev": f"{year}_Q{quarter*3}_{rev_type}_qtd"}, inplace=True)
    return df

def h1(df, year=YEAR, rev_type="net"):
    df = rev_from_to(df, start_month=1, start_year=year, end_month=6, end_year=year)
    df.rename(columns={f"01_{year}_to_06_{year}_{rev_type}_rev": f"{year}_{rev_type}_h1"}, inplace=True)
    return df

def h2(df, year=YEAR, rev_type="net"):
    df = rev_from_to(df, start_month=7, start_year=year, end_month=12, end_year=year)
    df.rename(columns={f"07_{year}_to_12_{year}_{rev_type}_rev": f"{year}_{rev_type}_h2"}, inplace=True)
    return df

def ytd(df, year, rev_type="net"):
    df = h1(df, year, rev_type=rev_type)
    df = h2(df, year, rev_type=rev_type)
    df[f"{year}_{rev_type}_ytd"] = df[f"{year}_{rev_type}_h1"] + df[f"{year}_{rev_type}_h2"]
    df.drop([f"{year}_{rev_type}_h1", f"{year}_{rev_type}_h2"], axis=1, inplace=True)
    return df

def yoy(df, year, rev_type="net"):
    df = ytd(df, year=year-1, rev_type=rev_type)
    df = ytd(df, year=year, rev_type=rev_type)
    df[f"{year}_{rev_type}_YoY"] = df[f"{year}_{rev_type}_ytd"] - df[f"{year-1}_{rev_type}_ytd"]
    df.drop([f"{year}_{rev_type}_ytd", f"{year-1}_{rev_type}_ytd"], axis=1, inplace=True)
    return df

def mom(df, month=MONTH, year=YEAR, rev_type="net"):
    if month == 1:
        prev_year = year - 1
        prev_month = 12
    else:
        prev_year = year
        prev_month = month - 1
    df[f"{month:02d}_{year}_{rev_type}_MoM"] = df[f"{rev_type}_{month:02d}_{year}"] - df[f"{rev_type}_{prev_month:02d}_{prev_year}"]
    return df

def fcst(df, month=MONTH, year=YEAR, last_n_days=7, rev_type="net"):
    days_remaining = (datetime.date(year, 12, 31) - datetime.date(year, month, 31)).days

    df = rev_from_to(df, start_month=1, start_year=year, end_month=month, end_year=year, rev_type=rev_type)
    df = drr(df, last_n_days=last_n_days, rev_type=rev_type)
    df[f"{month:02d}_{year}_{rev_type}_forecast"] = df[f"01_{year}_to_{month:02d}_{year}_{rev_type}_rev"] + (df[f"l{last_n_days}_{rev_type}_DRR"] * days_remaining)
    df.drop([f"01_{year}_to_{month:02d}_{year}_{rev_type}_rev", f"l{last_n_days}_{rev_type}_DRR"], axis=1, inplace=True)
    return df

def new_billers(df, year=YEAR, last_n_days=None, month=None, in_month=False, after_month=False):
    if month:
        df = rev_from_to(df, start_month=1, start_year=START_YEAR, end_month=month, end_year=year, rev_type="net")
        df.rename(columns={f"01_{START_YEAR}_to_{month:02d}_{year}_net_rev": f"rev_till_{month:02d}_{year}"}, inplace=True)

        df = rev_from_to(df, start_month=month, start_year=year, end_month=12, end_year=END_YEAR, rev_type="net")
        df.rename(columns={f"{month:02d}_{year}_to_{12}_{END_YEAR}_net_rev": f"rev_after_{month:02d}_{year}"}, inplace=True)

        if in_month and after_month:
            df[f"is_new_biller_in_after_{month:02d}_{year}"] = 0
            df[f"rev_till_{month:02d}_{year}"] -= df[f"net_{month:02d}_{year}"]
            df[f"rev_after_{month:02d}_{year}"] += df[f"net_{month:02d}_{year}"]
            df.loc[(df[f"rev_till_{month:02d}_{year}"] <= 0) & (df[f"rev_after_{month:02d}_{year}"] >= 1), f"is_new_biller_in_and_after_{month}_{year}"] = 1
        else:
            if in_month:
                df[f"is_new_biller_in_{month:02d}_{year}"] = 0
                df[f"rev_till_{month:02d}_{year}"] -= df[f"net_{month:02d}_{year}"]
                df.loc[(df[f"rev_till_{month:02d}_{year}"] <= 0) & (df[f"net_{month:02d}_{year}"] >= 1), f"is_new_biller_in_{month:02d}_{year}"] = 1
            elif after_month:
                df[f"is_new_biller_after_{month:02d}_{year}"] = 0
                df.loc[(df[f"rev_till_{month:02d}_{year}"] <= 0) & (df[f"rev_after_{month:02d}_{year}"] >= 1), f"is_new_biller_after_{month:02d}_{year}"] = 1
            else:
                pass
        df.drop([f"rev_till_{month:02d}_{year}", f"rev_after_{month:02d}_{year}"], axis=1, inplace=True)
    elif last_n_days:
        df[f"is_new_biller_in_l{last_n_days}d"] = 0

        df = rev_from_to(df, start_month=1, start_year=START_YEAR, end_month=12, end_year=END_YEAR, rev_type="net")
        df.rename(columns={f"01_{START_YEAR}_to_12_{END_YEAR}_net_rev": "rev_till_now"}, inplace=True)

        df["rev_till_now"] -= df[f"net_l{last_n_days}_{year}"]
        df.loc[(df["rev_till_now"] <= 0) & (df[f"net_l{last_n_days}_{year}"] >= 1), f"is_new_biller_in_l{last_n_days}d"] = 1

        df.drop(["rev_till_now", f"net_l{last_n_days}_{year}"], axis=1, inplace=True)
    else:
        df = rev_from_to(df, start_month=1, start_year=START_YEAR, end_month=12, end_year=YEAR-1, rev_type="net")
        df.rename(columns={f"01_{START_YEAR}_to_12_{YEAR-1}_net_rev": f"rev_before_{YEAR}"}, inplace=True)

        df = rev_from_to(df, start_month=1, start_year=YEAR, end_month=12, end_year=YEAR, rev_type="net")
        df.rename(columns={f"01_{YEAR}_to_12_{YEAR}_net_rev": f"rev_in_{YEAR}"}, inplace=True)

        df.loc[(df[f"rev_before_{YEAR}"] <= 0) & (df[f"rev_in_{YEAR}"] >= 1), "is_new_biller"] = 1
        df.drop([f"rev_before_{YEAR}", f"rev_in_{YEAR}"], axis=1, inplace=True)
    return df

# ---------------------------------------------------------------------------------------------------------
#                                              RAG setup
# ---------------------------------------------------------------------------------------------------------

N_RESULTS = 4
CHROMA_NAME = "functions"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./embed_model"     # model: all-MiniLM-L6-v2

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        return EMBED_MODEL.encode(input).tolist()
    
@st.cache_data
def initialize_functions():
    FUNCTION_MAP = {
        "promotion credits": {
            "name": "promo",
            "params": ["last_n_days", "month", "year"]
        },
        "daily run rate": {
            "name": "drr",
            "params": ["last_n_days", "month", "year", "rev_type"],
        },
        "monthly run rate": {
            "name": "mrr",
            "params": ["month", "year", "rev_type"]
        },
        "annual run rate": {
            "name": "arr",
            "params": ["quarter", "year", "rev_type"]
        },
        "incremental annual run rate": {
            "name": "inc_arr",
            "params": ["quarter", "year", "rev_type"],
        },
        "qurater to date": {
            "name": "qtd",
            "params": ["quarter", "year", "rev_type"]
        },
        "first half of year": {
            "name": "h1",
            "params": ["year", "rev_type"]
        },
        "second half of year": {
            "name": "h2",
            "params": ["year", "rev_type"],
        },
        "year to date": {
            "name": "ytd",
            "params": ["year", "rev_type"]
        },
        "year on year growth": {
            "name": "yoy",
            "params": ["year", "rev_type"],
        },
        "month on month growth": {
            "name": "mom",
            "params": ["month", "year", "rev_type"]
        },
        "forecast / projected revenue": {
            "name": "fcst",
            "params": ["month", "year", "last_n_days", "rev_type"]
        },
        "new billers or started billing/revenue": {
            "name": "new_billers",
            "params": ["year", "last_n_days", "month", "in_month", "after_month"]
        }
    }

    return FUNCTION_MAP

@st.cache_resource
def initialize_rag():
    embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    chroma_db = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())
    return embed_model, chroma_db

FUNCTION_MAP = initialize_functions()
EMBED_MODEL, CHROMA_DB = initialize_rag()
 
# ---------------------------------------------------------------------------------------------------------
#                                               GCP Setup
# ---------------------------------------------------------------------------------------------------------

LOCATION = "us-central1"
PROJECT_ID = "gcpops-427012"
MODEL_NAME = "gemini-1.5-flash"
DATASET_ID = f"{PROJECT_ID}.gcp_core"
TABLE_ID = f"{DATASET_ID}.revenue"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

@st.cache_data
def initialize_data():
    client = bigquery.Client(project=PROJECT_ID)
    df = client.query(f"SELECT * FROM {TABLE_ID}").to_dataframe()
    df_cols = df.columns.tolist()
    return df, df_cols

## UNCOMMENT
DF, DF_COLS = initialize_data()
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# ---------------------------------------------------------------------------------------------------------
#                                            Prompt Functions
# ---------------------------------------------------------------------------------------------------------

def generate_func_prompt(question, functions):
    func_names = [re.sub("\(.*?\)", "", func).strip() for func in functions]
    functions_text = "\n".join([f"{func}: {FUNCTION_MAP[func]}" for func in func_names])

    prompt = f"""Given a question, determine optimal functions and parameters.
If question requests a range, repeat the necessary function with updated params.
Return JSON array of function-parameter pairs and omit unnecessary params.
Do no include addiontal text in response.
Current year: {YEAR} & current month: {MONTH}

# start PARAM INFO #
year: 2023-2024 (default: current year)
month: 1-12 (default: current month)
last_n_days: 7, 14, 90 (default: 7)
rev_type: net, gross (default: net)
in_month: True, False (default: False)
after_month: True, False (default: False)
# end PARAM INFO #

# start FUNCTIONS #
{functions_text}
# end FUNCTIONS #

# start FORMAT #
[
    [func name, {{param values}}]
]
# end FORMAT #

Question: {question}
"""

    return prompt

def generate_filter_prompt(question, new_columns):
    new_columns = "\n".join([col + ": 1 or 0" if "new_biller" in col else col for col in new_columns])

    prompt = f"""Given a question, return JSON output with `filters`, `keep`, `groupby`, `sort`, `limit`.
Omit any fields from output if not necessary.
Current year: {YEAR} & current month: {MONTH}

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
net_l[N]_[year]: Net Revenue of last N days (N=7,14,90)
gross_l[N]_[year]: Gross Revenue of last N days (N=7,14,90)
{new_columns}
# end COLUMNS #

# start FORMAT #
{{
    "filters": [
        [column, operator (>, <, >=, <=, ==, !=), value]
    ],
    "keep": [columns],
    "groupby": {{
        cols: [columns], 
        func: (sum, mean, min, max, count)
    }},
    "sort": {{
        "cols": [columns], 
        "order": asc or desc
    }},
    "limit": number of rows
}}
# end FORMAT #

Question: {question}
"""

    return prompt

def generate_summary_prompt(question, dataset):
    prompt = f"""Given a question and a related CSV dataset, provide a JSON output containing `analysis` and `plot`.
Offer a 5-10 point deep analysis of each group's performance using vivid language and different emojis. Avoid simply stating facts.
Do not return any additional text.

Question: {question}

# start DATASET #
{dataset}
# end DATASET #

# start FORMAT #
{{
    "analysis": text,
    "plot": {{
        type: bar or line, 
        x: column, 
        y: [columns]
    }}
}}
# end FORMAT #
"""

    return prompt

# ---------------------------------------------------------------------------------------------------------
#                                            Helper Functions
# ---------------------------------------------------------------------------------------------------------

def get_model_response(prompt):
    model = GenerativeModel(model_name=MODEL_NAME)
    model_response = model.generate_content(prompt).text

    model_response = re.sub("```json", "", model_response)
    model_response = re.sub("```", "", model_response)
    model_response = model_response.strip()

    return model_response

def parse_json(text):
    json_obj = json.loads(text)
    return json_obj

def execute_model_functions(df, model_response):
    for func in model_response:
        func_name, params = func
        if func_name == "promo":
            df = promo(df, **params)
        elif func_name == "drr":
            df = drr(df, **params)
        elif func_name == "mrr":
            df = mrr(df, **params)
        elif func_name == "arr":
            df = arr(df, **params)
        elif func_name == "inc_arr":
            df = inc_arr(df, **params)
        elif func_name == "qtd":
            df = qtd(df, **params)
        elif func_name == "h1":
            df = h1(df, **params)
        elif func_name == "h2":
            df = h2(df, **params)
        elif func_name == "ytd":
            df = ytd(df, **params)
        elif func_name == "yoy":
            df = yoy(df, **params)
        elif func_name == "mom":
            df = mom(df, **params)
        elif func_name == "fcst":
            df = fcst(df, **params)
        elif func_name == "new_billers":
            df = new_billers(df, **params)

    return df

def apply_model_filters(df, model_filters, new_cols):
    try:
        filters = model_filters["filters"]

        if filters:
            for filter in filters:
                col = filter[0]
                val = filter[2]
                col_dtype = str(df.dtypes[col]).lower()

                if col_dtype == "object":
                    val = str(val)
                elif "int" in col_dtype:
                    val = int(val)
                elif "float" in col_dtype:
                    val = float(val)

                if filter[1] == "<":
                    df = df[df[col] < val]
                elif filter[1] == ">":
                    df = df[df[col] > val]
                elif filter[1] == "<=":
                    df = df[df[col] <= val]
                elif filter[1] == ">=":
                    df = df[df[col] >= val]
                elif filter[1] == "==":
                    if col_dtype == "object":
                        df = df[df[col].str.contains(val)]
                    else:
                        df = df[df[col] == val]
                elif filter[1] == "!=":
                    if col_dtype != "object":
                        df = df[~df[col].str.contains(val)]
                    else:
                        df = df[df[col] != val]
    except:
        pass

    try:
        keep_cols = model_filters["keep"]
        if keep_cols:
            df = df[keep_cols]
    except:
        pass

    try:
        groupby = model_filters["groupby"]
        if groupby:
            cols = groupby["cols"]
            func = groupby["func"]
            if func == "sum":
                df = df.groupby(cols).sum()
            elif func == "mean":
                df = df.groupby(cols).mean()
            elif func == "min":
                df = df.groupby(cols).min()
            elif func == "max":
                df = df.groupby(cols).max()
            elif func == "count":
                df = df.groupby(cols).count()
            df.reset_index(inplace=True)
    except:
        pass

    try:
        sort = model_filters["sort"]
        if sort:
            df = df.sort_values(sort[0], ascending=True)
    except:
         if new_cols:
            df = df.sort_values(new_cols, ascending=False)

    try:
        limit = int(model_filters["limit"])
        if limit:
            df = df.head(limit)
    except:
        pass

    return df

def data_to_csv(df):
    csv_data = ",".join(list(df.columns)) + "\n"
    df = df.astype(str)
    df = df.values.tolist()
    for row in df:
        csv_data += ",".join(row) + "\n"
    return csv_data

def get_new_cols(df_columns):
    new_cols = []
    for col in df_columns:
        if col not in DF_COLS:
            new_cols.append(col)
    return new_cols

# ---------------------------------------------------------------------------------------------------------
#                                             Streamlit UI
# ---------------------------------------------------------------------------------------------------------

st.title(f"Blitz 🚀")
if question := st.chat_input("Ask a question"):
    with st.chat_message("user"):
        st.markdown(question)

    relevant_formulas = CHROMA_DB.query(query_texts=[question], n_results=N_RESULTS)["documents"][0]
    prompt = generate_func_prompt(question, relevant_formulas)

    result = DF.copy()
    model_functions = get_model_response(prompt)
    try:
        model_functions = parse_json(model_functions)
    except Exception as e:
        st.error(model_functions)
        st.exception(e)
    result = execute_model_functions(result, model_functions)

    new_cols = get_new_cols(result.columns.tolist())
    prompt = generate_filter_prompt(question, new_cols)

    model_filters = get_model_response(prompt)
    try:
        model_filters = parse_json(model_filters)
    except Exception as e:
        st.error(model_filters)
        st.exception(e)
    result = apply_model_filters(result, model_filters, new_cols)

    result_csv = data_to_csv(result.head(10))
    prompt = generate_summary_prompt(question, result_csv)
    model_summary = get_model_response(prompt)
    try:
        model_summary = parse_json(model_summary)
    except Exception as e:
        st.error(model_summary)
        st.exception(e)

    with st.expander(label="Data", expanded=False):
        st.dataframe(data=result, use_container_width=True)

    try:
        plot = model_summary["plot"]
        with st.expander(label="Graph", expanded=True):
            if plot["type"] == "bar":
                st.bar_chart(result.head(10), x=plot["x"], y=plot["y"])
            elif plot["type"] == "line":
                st.line_chart(result.head(10), x=plot["x"], y=plot["y"])
            else:
                st.warning("No plot available!")
    except:
        st.warning("No plot available!")

    st.subheader("Summary")
    st.write(model_summary["summary"])