# 📊 FinBuddy: Your Personal Finance Teacher and Stock Analyzer

This repository contains the full pipeline for **FinBuddy**, an NLP-powered finance education chatbot. It can explain financial terms with relatable analogies and provide real-time stock summaries using Retrieval-Augmented Generation (RAG).

---

### 🔐 Request Access to Fine-Tuned Model

The fine-tuned Mistral LoRA adapter used in this project is available via Google Drive.  
Access is restricted to prevent unauthorized usage.

If you'd like to use the model for academic or non-commercial purposes, please [**request access via Google Drive**](https://drive.google.com/drive/folders/1kcS5YJ3JpVNllqU9Z3ZU2byA4SwdEt24?usp=drive_link).

> 💡 **Once approved**, place the downloaded model files in the `Models/` directory before running `app.py`.


## 📁 Directory Structure

```bash
NLP-Group-9-Finbuddy/
├── Code/
│   ├── FAISS/
│   ├── FinBuddy_backend_frontend/
│   ├── Fine_tuning/
│   ├── Model_Evaluation/
│   ├── Scrapping_data/
│   ├── Data/  ← Contains all cleaned data required to run the app
│   ├── app.py  ← Frontend Streamlit app
│   ├── fetch_stock.py
│   ├── faiss_stock_retriever.py
│   ├── finbuddy.py
├── Final-Group-Project-Report/
├── Final-Group-Presentation/
├── Group-Proposal/
```

---

## 🚀 How to Run This Project

### 📌 Step 1: Scrape Financial Data (Optional if using pre-cleaned data)
If you'd like to generate your own dataset from scratch:

1. Navigate to `Code/Scrapping_data/`
2. Run the following scripts in order:

```bash
python scrapping_investopedia_num.py   # Scrape definitions from Investopedia
python scrapping_capital.py            # Scrape from Capital.com
```

3. Clean and merge scraped data:

```bash
python Cleaning_csv.py
python Merging_data.py
```

Alternatively, you can skip scraping — all **cleaned and merged data is already available in the `Data/` folder**.

---

### 📌 Step 2: Launch the Frontend

To start the Streamlit app and interact with FinBuddy:

```bash
cd Code/
streamlit run app.py
```

This will launch the chatbot in your default browser.

---

## ⚙️ Components Overview

### 🔍 Retrieval (RAG) System
- **faiss_stock_retriever.py**: Maps user input to company tickers using semantic similarity.
- **fetch_stock.py**: Uses `yfinance` to fetch live stock data.
- **finbuddy.py**: Contains core logic for generating and summarizing responses using the fine-tuned model.

### 🤖 Model
- **Fine_tuning/**: Includes scripts used to train the Mistral-7B model using QLoRA.
- **Model_Evaluation/**: Scripts for evaluating model performance (ROUGE, BERTScore).

---

## 📦 Requirements

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

Typical packages:
- `streamlit`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `yfinance`
- `pandas`, `numpy`, `scikit-learn`
- `selenium` (for scraping)

---

## 📄 Notes
- Ensure you have the appropriate **Chrome WebDriver** installed for Selenium.
- The app assumes access to the **fine-tuned model and tokenizer saved locally** after training.

---

