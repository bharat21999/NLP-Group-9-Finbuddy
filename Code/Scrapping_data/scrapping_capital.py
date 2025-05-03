from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os
import pandas as pd
options = Options()
options.add_argument("--headless=new")  # or remove this line if you want to see browser
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")


driver = webdriver.Chrome(options=options)

driver.get("https://capital.com/financial-dictionary")

# Wait until term links are visible
WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".alphabet-category-item .inner a"))
)

print(" Loaded glossary page")

# Get all links for financial terms
link_elements = driver.find_elements(By.CSS_SELECTOR, ".alphabet-category-item .inner a")

term_links = []
for elem in link_elements:
    href = elem.get_attribute("href")
    text = elem.text.strip()
    if href and text:
        term_links.append((text, href))

print(f" Found {len(term_links)} terms")

# Visit each link and extract the first paragraph
results = []

for name, link in term_links:
    try:
        driver.get(link)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article p"))
        )
        para = driver.find_element(By.CSS_SELECTOR, "article p").text.strip()
        if para:
            results.append({"term": name, "definition": para})
            print(f" Scraped: {name}")
    except Exception as e:
        print(f" Failed for {name} - {e}")
    time.sleep(1)

driver.quit()

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("/home/ubuntu/finbuddy/Data/capital_com_financial_dictionary.csv", index=False)
print(f"\n Done! Scraped and saved {len(df)} terms.")
