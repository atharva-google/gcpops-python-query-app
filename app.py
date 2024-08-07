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
from copy_to_clipboard import st_copy_to_clipboard
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
#                                             Base Functions
# ---------------------------------------------------------------------------------------------------------

def rev_month_range(df, start_month, start_year=YEAR, end_month=MONTH, end_year=YEAR, rev_type="net"):
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

# ------------------------------------------ Renaming Functions ------------------------------------------

def get_month_revenue(df, month=MONTH, year=YEAR, rev_type="net"):
    df[f"{month:02d}_{year}_{rev_type}_revenue"] = df[f"{rev_type}_{month:02d}_{YEAR}"]
    return df

def get_last_n_days_revenue(df, last_n_days=7, rev_type="net"):
    df[f"last_{last_n_days}_days_{rev_type}_revenue"] = df[f"net_l{last_n_days}_{YEAR}"]
    return df

# ----------------------------------------- Fundamental Functions -----------------------------------------

def current_drr(df, last_n_days=7, rev_type="net"):
    df[f"current_l{last_n_days}d_{rev_type}_DRR"] = df[f"{rev_type}_l{last_n_days}_{YEAR}"] / last_n_days
    return df

def past_drr(df, month=MONTH, year=YEAR, rev_type="net"):
    num_days = calendar.monthrange(year, month)[1]
    df[f"{month:02d}_{year}_{rev_type}_DRR"] = df[f"{rev_type}_{month:02d}_{YEAR}"] / num_days
    return df

def mrr(df, month=MONTH, year=YEAR, rev_type="net"):
    if month == MONTH and year == YEAR:
        df[f"{MONTH:02d}_{year}_{rev_type}_MRR"] = df[f"{rev_type}_l90_{YEAR}"] / 90
    else:
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
        
    return df

def arr(df, quarter=QUARTER-1, year=YEAR, rev_type="net"):
    if year == YEAR and quarter > QUARTER-1:
        quarter = QUARTER-1

    df[f"{year}_Q{quarter}_{rev_type}_ARR"] = 0
    for month in range(quarter*3-2, quarter*3+1):
        df[f"{year}_Q{quarter}_{rev_type}_ARR"] += df[f"{rev_type}_{month:02d}_{year}"]
    df[f"{year}_Q{quarter}_{rev_type}_ARR"] *= 4
    return df

def inc_arr(df, quarter=QUARTER-1, year=YEAR, rev_type="net"):
    if year == YEAR and quarter > QUARTER-1:
        quarter = QUARTER-1

    df = arr(df, year=year-1, quarter=4, rev_type=rev_type)
    df = arr(df, quarter, rev_type=rev_type)
    df[f"{year}_Q{quarter}_{rev_type}_Inc_ARR"] = df[f"{year}_Q{quarter}_{rev_type}_ARR"] - df[f"{year-1}_Q4_{rev_type}_ARR"]
    df.drop([f"{year}_Q{quarter}_{rev_type}_ARR", f"{year-1}_Q4_{rev_type}_ARR"], axis=1, inplace=True)
    return df

def fcst(df, month=MONTH, year=YEAR, last_n_days=7, rev_type="net"):
    days_remaining = (datetime.date(year, 12, 31) - datetime.date(year, month, 31)).days
    df = rev_month_range(df, start_month=1, start_year=year, end_month=month, end_year=year, rev_type=rev_type)

    if month == MONTH and year == YEAR:
        df = current_drr(df, last_n_days=last_n_days, rev_type=rev_type)
        df[f"{month:02d}_{year}_{rev_type}_forecast"] = df[f"01_{year}_to_{month:02d}_{year}_{rev_type}_rev"] + (df[f"current_l{last_n_days}d_{rev_type}_DRR"] * days_remaining)
        df.drop([f"01_{year}_to_{month:02d}_{year}_{rev_type}_rev", f"current_l{last_n_days}d_{rev_type}_DRR"], axis=1, inplace=True)
    else:
        df = past_drr(df, month=month, year=year, rev_type=rev_type)
        df[f"{month:02d}_{year}_{rev_type}_forecast"] = df[f"01_{year}_to_{month:02d}_{year}_{rev_type}_rev"] + (df[f"{month:02d}_{year}_{rev_type}_DRR"] * days_remaining)
        df.drop([f"01_{year}_to_{month:02d}_{year}_{rev_type}_rev", f"{month:02d}_{year}_{rev_type}_DRR"], axis=1, inplace=True)

    return df

def revenue_gain_loss(df, month=MONTH, year=YEAR, rev_type="net"):
    if month == MONTH and year == YEAR:
        df[f"current_{rev_type}_revenue_gain_loss"] = df[f"{rev_type}_l7_{YEAR}"]*2 - df[f"{rev_type}_l14_{YEAR}"]
    else:
        if month == 1:
            prev_month = 12
            prev_year = year - 1
        else:
            prev_month = month - 1
            prev_year = year

        df[f"{month:02d}_{year}_{rev_type}_revenue_gain_loss"] = df[f"{rev_type}_{month:02d}_{year}"]*2 - df[f"{rev_type}_{prev_month:02d}_{prev_year}"]
    return df

# ----------------------------------------- New Biller Functions -----------------------------------------

def new_billers_last_n_days(df, last_n_days=7):
    df[f"is_new_biller_in_l{last_n_days}d"] = 0

    df = rev_month_range(df, start_month=1, start_year=START_YEAR, end_month=12, end_year=YEAR, rev_type="net")
    df.rename(columns={f"01_{START_YEAR}_to_12_{YEAR}_net_rev": "rev_till_now"}, inplace=True)

    df["rev_till_now"] -= df[f"net_l{last_n_days}_{YEAR}"]
    df.loc[(df["rev_till_now"] <= 0) & (df[f"net_l{last_n_days}_{YEAR}"] >= 1), f"is_new_biller_in_l{last_n_days}d"] = 1

    df.drop(["rev_till_now", f"net_l{last_n_days}_{YEAR}"], axis=1, inplace=True)
    return df

def new_billers_month_range(df, start_month=MONTH, start_year=YEAR, end_month=MONTH, end_year=YEAR):
    df = rev_month_range(df, start_month=1, start_year=START_YEAR, end_month=start_month, end_year=start_year, rev_type="net")
    df.rename(columns={f"01_{START_YEAR}_to_{start_month:02d}_{start_year}_net_rev": f"rev_till_{start_month:02d}_{start_year}"}, inplace=True)

    df = rev_month_range(df, start_month=start_month, start_year=start_year, end_month=end_month, end_year=end_year, rev_type="net")
    df.rename(columns={f"{start_month:02d}_{start_year}_to_{end_month:02d}_{YEAR}_net_rev": f"rev_after_{start_month:02d}_{start_year}_to_{end_month:02d}_{end_year}_net_rev"}, inplace=True)

    df["is_new_biller"] = 0
    df.loc[(df[f"rev_till_{start_month:02d}_{start_year}"] <= 0) & (df[f"rev_after_{start_month:02d}_{start_year}_to_{end_month:02d}_{end_year}_net_rev"] >= 1), "is_new_biller"] = 1

    df.drop([f"rev_till_{start_month:02d}_{start_year}", f"rev_after_{start_month:02d}_{start_year}_to_{end_month:02d}_{end_year}_net_rev"], axis=1, inplace=True)
    return df

# ------------------------------------------- Growth Functions -------------------------------------------

def drr_growth(df, month=MONTH, year=YEAR, compare_month=MONTH-1, compare_year=YEAR, rev_type="net"):
    if compare_month == 0:
        compare_month = 12
        compare_year -= 1
    
    if month == MONTH and year == YEAR:
        df = current_drr(df, rev_type=rev_type)
        df = past_drr(df, month=compare_month, year=compare_year, rev_type=rev_type)
        df[f"current_vs_{compare_month:02d}_{compare_year}_{rev_type}_DRR_growth"] = df[f"current_l7d_{rev_type}_DRR"] - df[f"{compare_month:02d}_{compare_year}_{rev_type}_DRR"]
        df.drop([f"current_l7d_{rev_type}_DRR", f"{compare_month:02d}_{compare_year}_{rev_type}_DRR"], axis=1, inplace=True)
    else:
        df = past_drr(df, month=month, year=year, rev_type=rev_type)
        df = past_drr(df, month=compare_month, year=compare_year, rev_type=rev_type)
        df[f"{month:02d}_{year}_vs_{compare_month:02d}_{compare_year}_{rev_type}_drr_growth"] = df[f"{month:02d}_{year}_{rev_type}_DRR"] - df[f"{compare_month:02d}_{compare_year}_{rev_type}_DRR"]
        df.drop([f"{month:02d}_{year}_{rev_type}_DRR", f"{compare_month:02d}_{compare_year}_{rev_type}_DRR"], axis=1, inplace=True)
    return df

def mrr_growth(df, month=MONTH, year=YEAR, compare_month=MONTH-1, compare_year=YEAR, rev_type="net"):
    if compare_month == 0:
        compare_month = 12
        compare_year -= 1
    
    df = mrr(df, month=month, year=year, rev_type=rev_type)
    df = mrr(df, month=compare_month, year=compare_year, rev_type=rev_type)
    df[f"{month:02d}_{year}_vs_{compare_month:02d}_{compare_year}_{rev_type}_MRR_growth"] = df[f"{month:02d}_{year}_{rev_type}_MRR"] - df[f"{compare_month:02d}_{compare_year}_{rev_type}_MRR"]
    df.drop([f"{month:02d}_{year}_{rev_type}_MRR", f"{compare_month:02d}_{compare_year}_{rev_type}_MRR"], axis=1, inplace=True)
    return df

def arr_growth(df, quarter=QUARTER-1, year=YEAR, compare_quarter=QUARTER-2, compare_year=YEAR, rev_type="net"):
    if compare_quarter == 0:
        compare_quarter = 4
        compare_year -= 1
    
    df = arr(df, quarter=quarter, year=year, rev_type=rev_type)
    df = arr(df, quarter=compare_quarter, year=compare_year, rev_type=rev_type)
    df[f"{year}_Q{quarter}_vs_{compare_year}_Q{compare_quarter}_{rev_type}_ARR_growth"] = df[f"{year}_Q{quarter}_{rev_type}_ARR"] - df[f"{compare_year}_Q{compare_quarter}_{rev_type}_ARR"]
    df.drop([f"{year}_Q{quarter}_{rev_type}_ARR", f"{compare_year}_Q{compare_quarter}_{rev_type}_ARR"], axis=1, inplace=True)
    return df

def mom_growth(df, month=MONTH, year=YEAR, compare_month=MONTH-1, compare_year=YEAR, rev_type="net"):
    if compare_month == 0:
        compare_month = 12
        compare_year -= 1

    df[f"{month:02d}_{year}_vs_{compare_month:02d}_{compare_year}_{rev_type}_MoM_growth"] = df[f"{rev_type}_{month:02d}_{year}"] - df[f"{rev_type}_{compare_month:02d}_{compare_year}"]
    return df

def qoq_growth(df, quarter=QUARTER-1, year=YEAR, compare_quarter=QUARTER-2, compare_year=YEAR, rev_type="net"):
    def qtd(df, quarter=QUARTER, year=YEAR, rev_type="net"):
        df = rev_month_range(df, start_month=quarter*3-2, start_year=year, end_month=quarter*3, end_year=year)
        df.rename(columns={f"{(quarter*3-2):02d}_{year}_to_{(quarter*3):02d}_{year}_{rev_type}_rev": f"{year}_Q{quarter}_{rev_type}_qtd"}, inplace=True)
        return df

    if compare_quarter == 0:
        compare_quarter = 4
        compare_year -= 1
    
    df = qtd(df, quarter=quarter, year=year, rev_type=rev_type)
    df = qtd(df, quarter=compare_quarter, year=compare_year, rev_type=rev_type)

    df[f"{year}_Q{quarter}_vs_{compare_year}_Q{compare_quarter}_{rev_type}_QoQ_growth"] = df[f"{year}_Q{quarter}_{rev_type}_qtd"] - df[f"{compare_year}_Q{compare_quarter}_{rev_type}_qtd"]
    df.drop([f"{year}_Q{quarter}_{rev_type}_qtd", f"{compare_year}_Q{compare_quarter}_{rev_type}_qtd"], axis=1, inplace=True)
    return df

def h1_growth(df, year=YEAR, compare_year=YEAR-1, rev_type="net"):
    def h1(df, year=YEAR, rev_type="net"):
        df = rev_month_range(df, start_month=1, start_year=year, end_month=6, end_year=year)
        df.rename(columns={f"01_{year}_to_06_{year}_{rev_type}_rev": f"{year}_{rev_type}_h1"}, inplace=True)
        return df

    df = h1(df, year=year)
    df = h1(df, year=compare_year)
    df[f"{year}_vs_{compare_year}_{rev_type}_H1_growth"] = df[f"{year}_{rev_type}_h1"] - df[f"{compare_year}_{rev_type}_h1"]
    df.drop([f"{year}_{rev_type}_h1", f"{compare_year}_{rev_type}_h1"], axis=1, inplace=True)
    return df

def h2_growth(df, year=YEAR, compare_year=YEAR-1, rev_type="net"):
    def h2(df, year=YEAR, rev_type="net"):
        df = rev_month_range(df, start_month=7, start_year=year, end_month=12, end_year=year)
        df.rename(columns={f"07_{year}_to_12_{year}_{rev_type}_rev": f"{year}_{rev_type}_h2"}, inplace=True)
        return df

    df = h2(df, year=year)
    df = h2(df, year=compare_year)
    df[f"{year}_vs_{compare_year}_{rev_type}_H2_growth"] = df[f"{year}_{rev_type}_h2"] - df[f"{compare_year}_{rev_type}_h2"]
    df.drop([f"{year}_{rev_type}_h2", f"{compare_year}_{rev_type}_h2"], axis=1, inplace=True)
    return df

def yoy_growth(df, year=YEAR, compare_year=YEAR-1, rev_type="net"):
    def h1(df, year=YEAR, rev_type="net"):
        df = rev_month_range(df, start_month=1, start_year=year, end_month=6, end_year=year)
        df.rename(columns={f"01_{year}_to_06_{year}_{rev_type}_rev": f"{year}_{rev_type}_h1"}, inplace=True)
        return df
    def h2(df, year=YEAR, rev_type="net"):
        df = rev_month_range(df, start_month=7, start_year=year, end_month=12, end_year=year)
        df.rename(columns={f"07_{year}_to_12_{year}_{rev_type}_rev": f"{year}_{rev_type}_h2"}, inplace=True)
        return df
    def ytd(df, year, rev_type="net"):
        df = h1(df, year, rev_type=rev_type)
        df = h2(df, year, rev_type=rev_type)
        df[f"{year}_{rev_type}_ytd"] = df[f"{year}_{rev_type}_h1"] + df[f"{year}_{rev_type}_h2"]
        df.drop([f"{year}_{rev_type}_h1", f"{year}_{rev_type}_h2"], axis=1, inplace=True)
        return df

    df = ytd(df, year=year, rev_type=rev_type)
    df = ytd(df, year=compare_year, rev_type=rev_type)
    df[f"{year}_{rev_type}_YoY_growth"] = df[f"{year}_{rev_type}_ytd"] - df[f"{compare_year}_{rev_type}_ytd"]
    df.drop([f"{year}_{rev_type}_ytd", f"{compare_year}_{rev_type}_ytd"], axis=1, inplace=True)
    return df

# ---------------------------------------------------------------------------------------------------------
#                                            Prompt Functions
# ---------------------------------------------------------------------------------------------------------

def generate_func_prompt(question, functions):
    func_names = [re.sub("\(.*?\)", "", func).strip() for func in functions]
    functions_text = "\n".join([f"{key}: {FUNCTION_MAP[key]['name']}({', '.join(FUNCTION_MAP[key]['params'])})" for key in func_names])

    prompt = f"""Given a question, determine optimal functions and parameters and return JSON array of function-parameter pairs.
If question requests a range, repeat the necessary function with updated params.
Do no include addiontal text in response.
Current year: {YEAR} & current month: {MONTH}.

# start PARAM INFO #
month: 1-12
year: 2023-2024
last_n_days: 7, 14, 90 (default: 7)
rev_type: net, gross (default: net)
in_month: True, False (default: False)
after_month: True, False (default: False)
# end PARAM INFO #

# start FUNCTIONS #
last n days revenue: get_last_n_days_revenue(last_n_days, rev_type)
month revenue: get_month_revenue(month, year, rev_type)
revenue in month range: rev_month_range(start_month, start_year, end_month, end_year, rev_type)

{functions_text}
# end FUNCTIONS #

# start FORMAT #
[
    [func name, {{params}}]
]
# end FORMAT #

Question: {question}
"""

    return prompt

def generate_filter_prompt(question, new_columns):
    new_columns = "\n".join([col + ": 1 or 0" if "new_biller" in col else col for col in new_columns])

    prompt = f"""Given a question, determine optimal data processing steps.
Use `filters`, `keep`, `groupby`, `sort`, `limit` fields. Strictly omit unnecessary fields.
Do not include addiontal text in response
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
    "sort": {{"cols": [columns], "order": "asc" or "desc}}
    "limit": number of rows
}}
# end FORMAT #

Question: {question}
"""

    return prompt

def generate_summary_prompt(question, dataset):
    prompt = f"""Given a question and a related CSV dataset, provide a JSON output containing `analysis` and `plot`.
Give in depth analysis in bullet points for each group's performance using vivid language and different emojis. Avoid simply stating facts.
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

def get_model_response(prompt, config):
    model = GenerativeModel(model_name=MODEL_NAME)
    model_response = model.generate_content(prompt, generation_config=config).text

    model_response = re.sub("```json", "", model_response)
    model_response = re.sub("```", "", model_response)
    model_response = model_response.strip()

    return model_response

def parse_json(text):
    json_obj = json.loads(text)
    return json_obj

def data_to_sv(df, n_csv_rows=10):
    tsv_data = "\t".join(list(df.columns)) + "\n"
    csv_data = ",".join(list(df.columns)) + "\n"
    df = df.astype(str)
    df = df.values.tolist()
    for row in df:
        tsv_data += "\t".join(row) + "\n"
        if n_csv_rows >= 0:
            csv_data += ",".join(row) + "\n"
            n_csv_rows -= 1
    return csv_data, tsv_data

def get_new_cols(df_columns):
    new_cols = []
    for col in df_columns:
        if col not in DF_COLS:
            new_cols.append(col)
    return new_cols

def df_transpose(df, x, y):
    df = df[[x] + y]
    df.set_index(x, inplace=True)
    df = df.transpose()
    return df

# ---------------------------------------------------------------------------------------------------------
#                                         Execution Functions
# ---------------------------------------------------------------------------------------------------------

def execute_model_functions(df, model_response):
    for func in model_response:
        func_name, params = func
        df = eval(f"{func_name}(df, **params)")

    return df

def apply_model_filters(df, model_filters, new_cols):
    try:
        filters = model_filters["filters"]

        for filter in filters:
            col, opr, val = filter
            col_dtype = str(df.dtypes[col]).lower()

            if col_dtype == "object":
                val = str(val)

                df[col].fillna("", inplace=True)
                if opr == "==":
                    df = df[df[col].str.contains(val)]
                elif opr == "!=":
                    df = df[~df[col].str.contains(val)]
            else:
                if "int" in col_dtype:
                    val = int(val)
                elif "float" in col_dtype:
                    val = float(val)
                df = eval(f"df[df['{col}'] {str(opr)} {str(val)}]")
        print("Filters Applied")
    except:
        pass

    try:
        keep_cols = model_filters["keep"]
        if set(DF_COLS).intersection(set(keep_cols)):
            df = df[keep_cols]
        else:
            df = df[["account_name"] + keep_cols]
        print("Removed extra columns")
    except:
        pass

    
    try:
        groupby = model_filters["groupby"]
        if groupby:
            df = eval(f"df.groupby({groupby['cols']}).{groupby['func']}()")
            df.reset_index(inplace=True)
            if "index" in df.columns.tolist():
                df = df.drop("index", axis=1)
        print("Grouped Data")
    except:
        pass

    try:
        sort = model_filters["sort"]
        df = df.sort_values(sort["cols"], ascending=sort["order"]=="asc")
        print("Sorted Data")
    except:
        rem_cols = get_new_cols(df.columns.to_list())
        if rem_cols:
            df.sort_values(rem_cols, ascending=False)

    try:
        limit = int(model_filters["limit"])
        df = df.head(limit)
        print("Limitted Data")
    except:
        pass

    return df

# ---------------------------------------------------------------------------------------------------------
#                                              RAG setup
# ---------------------------------------------------------------------------------------------------------

CHROMA_NAME = "functions"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./embed_model"     # model: all-MiniLM-L6-v2

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        return EMBED_MODEL.encode(input).tolist()
    
@st.cache_data
def initialize_functions():
    FUNCTION_MAP = {
        # Fundamental Functions
        'current daily run rate': {'name': 'current_drr', 'params': ['last_n_days', 'rev_type']}, 
        'past daily run rate': {'name': 'past_drr', 'params': ['month', 'year', 'rev_type']}, 
        'monthly run rate': {'name': 'mrr', 'params': ['month', 'year', 'rev_type']}, 
        'annual run rate': {'name': 'arr', 'params': ['quarter', 'year', 'rev_type']}, 
        'incremental annual run rate': {'name': 'inc_arr', 'params': ['quarter', 'year', 'rev_type']}, 

        # Custom Functions
        'revenue gained lost': {'name': 'revenue_gain_loss', 'params': ['month', 'year', 'rev_type']},
        'forecast / projected revenue': {'name': 'fcst', 'params': ['month', 'year', 'last_n_days', 'rev_type']}, 

        # New Biller Functions
        'new billers or started billing/revenue in last n days': {'name': 'new_billers_last_n_days', 'params': ['last_n_days']}, 
        'new billers or started billing/revenue in month range': {'name': 'new_billers_month_range', 'params': ['start_month', 'start_year', 'end_month', 'end_year']}, 

        # Growth Functions
        'daily run rate growth/comparison': {'name': 'drr_growth', 'params': ['month', 'year', 'compare_month', 'compare_year', 'rev_type']},
        'monthly run rate growth/comparison': {'name': 'mrr_growth', 'params': ['month', 'year', 'compare_month', 'compare_year', 'rev_type']},
        'annual run rate growth/comparison': {'name': 'arr_growth', 'params': ['quarter', 'year', 'compare_quarter', 'compare_year', 'rev_type']},
        'month on month growth/comparison': {'name': 'mom_growth', 'params': ['month', 'year', 'compare_month', 'compare_year', 'rev_type']},
        'quarter on quarter growth/comparison': {'name': 'qoq_growth', 'params': ['quarter', 'year', 'compare_quarter', 'compare_year', 'rev_type']},
        'first half of year growth/comparison': {'name': 'h1_growth', 'params': ['year', 'compare_year', 'rev_type']},
        'second half of year growth/comparison': {'name': 'h2_growth', 'params': ['year', 'compare_year', 'rev_type']},
        'year on year growth/comparison': {'name': 'yoy_growth', 'params': ['year', 'compare_year', 'rev_type']},
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

CREATIVE_CONFIG = GenerationConfig(temperature=1, top_p=1, top_k=32)
NON_CREATIVE_CONFIG = GenerationConfig(temperature=0, top_p=0.2, top_k=8)

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
#                                             Streamlit UI
# ---------------------------------------------------------------------------------------------------------

N_RESULTS = 3
N_SUMMARY_ROWS = 10

st.title(f"DataDiver 🐬")

if question := st.chat_input("Ask a question"):
    with st.chat_message("user"):
            st.markdown(question)

    data_tab, analysis_tab, graph_tab = st.tabs(["🗃️ Data", "🔍 Analysis", "📈 Graph"])

    data_toast = st.toast("🌱 Starting Data Retrieval...")
    relevant_formulas = CHROMA_DB.query(query_texts=[question], n_results=N_RESULTS)["documents"][0]
    prompt = generate_func_prompt(question, relevant_formulas)
    model_functions = get_model_response(prompt, config=NON_CREATIVE_CONFIG)
    data_toast.toast("⚙️ Got Data Functions...")
    
    try:
        model_functions = parse_json(model_functions)

        result = DF.copy()
        result = execute_model_functions(result, model_functions)
        data_toast.toast("📐 Completed Function Execution...")
        
        new_cols = get_new_cols(result.columns.tolist())
        prompt = generate_filter_prompt(question, new_cols)
        model_filters = get_model_response(prompt, config=NON_CREATIVE_CONFIG)
        data_toast.toast("🧹 Got Data Filters...")

        try:
            model_filters = parse_json(model_filters)

            result = apply_model_filters(result, model_filters, new_cols)

            with data_tab:
                st.dataframe(data=result, use_container_width=True)
                data_toast.toast("🗃️ Data Displayed!")
                result_csv, result_tsv = data_to_sv(result, n_csv_rows=N_SUMMARY_ROWS)
                st_copy_to_clipboard(result_tsv, "📋 Copy Data", "✅ Data Copied")

            summary_toast = st.toast("🌱 Starting Summary Retrieval...")
            prompt = generate_summary_prompt(question, result_csv)
            model_summary = get_model_response(prompt, config=CREATIVE_CONFIG)
            try:
                model_summary = parse_json(model_summary)
                plot = model_summary["plot"]
                analysis = model_summary["analysis"]
                summary_toast = st.toast("🔍 Summary and Plot Displayed!")

                with analysis_tab:
                    st.write(analysis)

                with graph_tab:
                    if plot["type"] == "bar":
                        st.bar_chart(result.head(N_SUMMARY_ROWS), x=plot["x"], y=plot["y"])
                    elif plot["type"] == "line":
                        st.line_chart(df_transpose(result.head(N_SUMMARY_ROWS), x=plot["x"], y=plot["y"]))
                    else:
                        st.warning("No plot available!")
            except Exception as e:
                with analysis_tab:
                    st.warning(model_summary)
                    st.exception(e)
                with graph_tab:
                    st.warning(model_filters)
                    st.exception(e)
        except Exception as e:
            with data_tab:
                st.warning(model_filters)
                st.exception(e)
            with analysis_tab:
                st.warning(model_filters)
                st.exception(e)
            with graph_tab:
                st.warning(model_filters)
                st.exception(e)
    except Exception as e:
        with data_tab:
            st.warning(model_filters)
            st.exception(e)
        with analysis_tab:
            st.warning(model_filters)
            st.exception(e)
        with graph_tab:
            st.warning(model_filters)
            st.exception(e)