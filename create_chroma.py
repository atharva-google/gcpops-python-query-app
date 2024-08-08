import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction

FUNCTION_MAP = {
    # Fundamental Functions
    'current daily run rate': {'name': 'current_drr', 'params': ['last_n_days', 'rev_type']}, 
    'past daily run rate': {'name': 'past_drr', 'params': ['month', 'year', 'rev_type']}, 
    'monthly run rate': {'name': 'mrr', 'params': ['month', 'year', 'rev_type']}, 
    'annual run rate for quarter': {'name': 'arr', 'params': ['quarter', 'year', 'rev_type']}, 
    'incremental annual run rate for quarter': {'name': 'inc_arr', 'params': ['quarter', 'year', 'rev_type']}, 

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

CHROMA_NAME = "functions"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./embed_model"     # model: all-MiniLM-L6-v2
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_PATH)

chroma_docs = [f"{key} ({vals['name']})" for key, vals in FUNCTION_MAP.items()]

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
except:
    pass

create_chroma_db(chroma_docs, CHROMA_PATH, CHROMA_NAME)