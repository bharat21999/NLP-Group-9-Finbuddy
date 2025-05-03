import streamlit as st
from finbuddy import generate_answer, summarize_stock, model, tokenizer
from fetch_stock import fetch_stock_info
from faiss_stock_retriever import retrieve_ticker
import re

def escape_markdown(text):
    return re.sub(r'([_*`])', r'\\\1', text)


# === UI Setup ===
st.set_page_config(page_title="💬 FinBuddy Chatbot", layout="centered")
st.title("💬 FinBuddy: Your Personal Finance Teacher")
st.markdown("Ask me anything about finance or a stock (e.g., 'What is inflation?' or 'Tell me about Apple stock')")

# === User Input ===
user_input = st.text_input("💬 Ask a question:")

# === Chat Logic ===
if user_input:
    with st.spinner("🧠 Thinking..."):
        if "stock" in user_input.lower() or "price" in user_input.lower():
            try:
                result = retrieve_ticker(user_input)
                ticker = result['ticker']
                company = result['company']
                note = result.get('note', "")
                stock_response = fetch_stock_info(ticker)

                st.markdown(f"### 📈 {company} ({ticker})\n{note}")
                st.markdown("```text\n" + stock_response + "\n```")

                summary = summarize_stock(company, ticker, stock_response)
                st.markdown("### 🤖 FinBuddy's Explanation:")
                st.success(summary)

            except Exception as e:
                st.error(f"⚠️ Could not retrieve stock info: {e}")
        else:
            answer = generate_answer(user_input)
            escaped_answer = escape_markdown(answer)
            st.markdown("### 🤖 FinBuddy's Answer:")
            st.markdown(escaped_answer)

