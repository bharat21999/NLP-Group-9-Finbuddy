from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup

BASE_URL = "https://www.investopedia.com"


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def get_num_terms(driver):
    url = "https://www.investopedia.com/terms-beginning-with-num-4769350"
    driver.get(url)
    time.sleep(4)
    links = []

    elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/terms/') and contains(@href, '.asp')]")
    for el in elements:
        href = el.get_attribute("href")
        text = el.text.strip()
        if href and text:
            links.append((href, text))
    return links


def get_definition(url):
    try:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, "html.parser")
        para = soup.find("p")
        return para.get_text(strip=True) if para else ""
    except:
        return ""


def main():
    driver = get_driver()
    print(" Scraping number-based financial terms...")

    term_links = get_num_terms(driver)
    print(f" Found {len(term_links)} number-based terms")

    all_data = []
    for url, term in term_links:
        definition = get_definition(url)
        if definition and len(definition) > 20:
            all_data.append({
                "term": term,
                "definition": definition,
                "url": url
            })
        time.sleep(0.3)

    driver.quit()

    df = pd.DataFrame(all_data)
    df.to_csv("/home/ubuntu/finbuddy/Data/investopedia_num_terms.csv", index=False)
    print(f" Saved {len(df)} number-terms to investopedia_num_terms.csv")


if __name__ == "__main__":
    main()
