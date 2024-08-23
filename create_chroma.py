import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction

YEAR, DAYS_REMAINING = 0, 0

AVAILABLE_FORMULAS = {
    "drr": {"name": "Daily Run Rate", "text": f"- current: last N days rev / N (default N = 7)\n- past month: that month's rev / days in month"},
    "mrr": {"name": "Monthly Run Rate", "text": f"current month: last 90 days rev / 3\n- past month: last 3 months rev / 3"},
    "arr": {"name": "Annual Run Rate", "text": f"- method: rev(Period) * Number of periods in year\n- default: Current DRR * 365"},
    "inc arr": {"name": "Incremental Annual Run Rate", "text": f"- method: (Latest Period rev - Past Period rev) * Number of periods in year\n- explanation: rev growth between period * number of periods in year\n- default: (Current DRR - {YEAR-1} Dec DRR) * 365"},
    "revevnue gain loss": {"name": "Revenue Gain Loss", "text": f"- formula: rev(Period X) - rev(Period X-1)\n- explanation: rev growth between two consecutive periods\n- default: current last 7 days - last 7 days prior to current last 7 days"},
    "forecast": {"name": "Projected Revenue", "text": f"- method: (Year-to-Date rev) + rev(Period) * Number of Periods in year\n- explanation: estimated year end rev by adding the rev earned so far with a projected amount based on a given period.\n- default: ({YEAR} Year-to-Date rev) + (DRR * {DAYS_REMAINING})"},
    # "new billers": {"name": "Get accounts that started revenue for a time period", "text": f"- criteria: check total rev from the beginning of data to just before the given period is < 1 AND total revenue after or during the given period is > 1\n- default period: last 7 days\n- example to find new billers:\n  * in june 2024: (total rev from start to may 2024) < 1 AND (total rev in june 2024) > 1\n  * after june 2024: (total rev from start to june 2024) < 1 AND (total rev from june 2024 to current date) > 1"}
}

CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./embed_model"     # model: all-MiniLM-L6-v2
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_PATH)

chroma_docs = [f"{vals['name']}:{key}" for key, vals in AVAILABLE_FORMULAS.items()]

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
    chromadb.PersistentClient(path=CHROMA_PATH).delete_collection(name=CHROMA_NAME)
    print("Deleted older chroma collection")
except:
    pass

print("Creating Chroma DB")
create_chroma_db(chroma_docs, CHROMA_PATH, CHROMA_NAME)