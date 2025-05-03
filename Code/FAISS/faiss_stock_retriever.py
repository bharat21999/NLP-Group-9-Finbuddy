import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load data ===
df = pd.read_csv("/home/ubuntu/finbuddy/Data/stocks-list.csv")
companies = df["Company Name"].fillna("").tolist()
tickers = df["Symbol"].fillna("").tolist()

# === Create lookup ===
company_to_ticker = dict(zip(companies, tickers))

# === Manual aliases for hard matches ===
manual_map = {
    "google": "Alphabet Inc.",
    "jp morgan": "JPMorgan Chase & Co.",
    "amazon": "Amazon.com, Inc.",
    "meta": "Meta Platforms, Inc.",
    "facebook": "Meta Platforms, Inc.",
    "berkshire": "Berkshire Hathaway Inc.",
    "citi": "Citigroup Inc.",
}

# === Embed company names ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(companies, convert_to_numpy=True)

# === Setup FAISS index ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === Main retrieval function ===
def retrieve_ticker(user_query, top_k=1):
    query_clean = user_query.lower().strip()

    # Check for manual overrides
    if query_clean in manual_map:
        matched_company = manual_map[query_clean]
        return {
            "ticker": company_to_ticker[matched_company],
            "company": matched_company,
            "score": 0.0,
            "note": "🔧 Manually matched"
        }

    # Vector search fallback
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    matched_company = companies[I[0][0]]
    distance = float(D[0][0])
    note = "✅ Confident match" if distance < 0.8 else "⚠️ Possibly wrong match"

    return {
        "ticker": company_to_ticker[matched_company],
        "company": matched_company,
        "score": distance,
        "note": note
    }

# === CLI test ===
if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Company lookup (type 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            break
        result = retrieve_ticker(user_query)
        print(f" Match: {result['company']} → {result['ticker']} (distance={result['score']:.4f}) {result['note']}")
