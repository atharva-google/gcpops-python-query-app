import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings, EmbeddingFunction

FORMULA_MAP = {
    "Promotion (Promo) Credits": "Gross Revenue - Net Revenue",
    "Daily Run Rate (DRR)": "Last 7 days revenue / 7 | Last 14 days revenue / 14 | [Any month revenue] / [Number of days in the month]",
    "Monthly Run Rate (MRR)": "[Last 90 days data] / 3 | [Last 3 data months from requested month] / 3 (eg: March 2024 MRR = ([january 2024 revenue] + [february 2024 revenue] + [march 2024 revenue]) / 3)",
    "Annual Run Rate (ARR)": "[PREVIOUS QUARTER] * 4",
    "Incremental Run Rate (IRR)": "[PREVIOUS QUARTER] * 4 - [last year Quarter 4] * 4",
    "Criteria to get accounts that started billing/revenue IN the last `N` days": "([Total revenue from STARTING DATE to ENDING DATE] - [last `N` days revenue from current date]) <= 0 AND [last `N` days revenue from current date] > 0",
    "Criteria to get accounts that started billing/revenue IN or DURING a `MONTH`": "([Total revenue from STARTING DATE to `MONTH`]) <= 0 AND [requested month revenue] > 0",
    "Criteria to get accounts that started billing/revenue AFTER a `MONTH`": "([Total Revenue from STARTING DATE to `MONTH`]) <= 0 AND [Total Revenue from requested month to ENDING DATE] > 0"
}

CHROMA_NAME = "formulas"
CHROMA_PATH = "./chroma_db"
MODEL = SentenceTransformer("./all-MiniLM-L6-v2")

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        return MODEL.encode(input).tolist()

def create_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=CustomEmbeddingFunction())

    for i, document in enumerate(documents):
        db.add(documents=document, ids=str(i))
    return db, name

# Uncomment to create `chroma_db`
# create_chroma_db(list(FORMULA_MAP.keys()), CHROMA_PATH, CHROMA_NAME)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
db = chroma_client.get_collection(name=CHROMA_NAME, embedding_function=CustomEmbeddingFunction())

n_results = 2
query="what is the gross drr and mrr in june 2023"

relevant_formulas = db.query(query_texts=[query], n_results=n_results)["documents"][0]
print(relevant_formulas)