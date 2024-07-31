import shutil
import calendar
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction

YEAR, MONTH, QUARTER, START_DATE, END_DATE, DAYS_REMAINING = [1]*6

FORMULA_MAP = {
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

CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = SentenceTransformer("./embed_model")

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        return EMBED_MODEL.encode(input).tolist()
    
def create_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=CustomEmbeddingFunction())

    for i, document in enumerate(documents):
        db.add(documents=document, ids=str(i))
    return db, name

try:
    shutil.rmtree(CHROMA_PATH)
except:
    pass

create_chroma_db(list(FORMULA_MAP.keys()), CHROMA_PATH, CHROMA_NAME)