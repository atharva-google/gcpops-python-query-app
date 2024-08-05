import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction

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

N_RESULTS = 4
CHROMA_NAME = "functions"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./embed_model"     # model: all-MiniLM-L6-v2
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_PATH)

chroma_docs = [f"{key} ({vals['name']})" for key, vals in FUNCTION_MAP.items()]
chroma_docs

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