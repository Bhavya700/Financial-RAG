import os
import json
import logging
from typing import List, Optional

# LangChain Document
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback Document class if langchain-core is not installed yet
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
        def __repr__(self):
            return f"Document(metadata={self.metadata}, page_content=...)"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_sec_documents(
    tickers: List[str],
    form_types: List[str],
    years: List[str],
    quarters: Optional[List[str]] = None,
    sections: Optional[List[str]] = None
) -> List[Document]:
    """
    Dynamically fetch SEC filings utilizing either sec_api or sec_edgar_downloader.
    Parses them to clean plain text and saves dynamically to data/{ticker}/{year}/{form_type}/.
    """
    api_key = os.getenv("SEC_API_KEY")
    if api_key:
        logger.info("SEC_API_KEY found. Utilizing sec_api for robust extraction.")
        return _fetch_with_sec_api(api_key, tickers, form_types, years, quarters, sections)
    else:
        logger.warning("SEC_API_KEY not found. Falling back to sec-edgar-downloader.")
        return _fetch_with_edgar_downloader(tickers, form_types, years, quarters, sections)


def _clean_html(html_str: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_str, "html.parser")
    # Replace common structural tags with line breaks
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p in soup.find_all(["p", "div", "h1", "h2", "h3", "li"]):
        p.insert_before("\n")
        p.insert_after("\n")
    return soup.get_text(separator="\n", strip=True)


def _save_document(ticker: str, year: str, form_type: str, quarter: str, section: str, text: str) -> str:
    # Build filepath
    save_dir = os.path.join(os.path.dirname(__file__), "data", ticker, year, form_type.replace("/", "-"))
    os.makedirs(save_dir, exist_ok=True)
    
    q_str = f"_{quarter}" if quarter else ""
    s_str = f"_Section_{section}" if section else ""
    filename = f"{ticker}_{year}{q_str}_{form_type}{s_str}.txt".replace("/", "-")
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
        
    return filepath


def _fetch_with_sec_api(
    api_key: str, 
    tickers: List[str], 
    form_types: List[str], 
    years: List[str], 
    quarters: Optional[List[str]],
    sections: Optional[List[str]]
) -> List[Document]:
    from sec_api import QueryApi, ExtractorApi
    
    query_api = QueryApi(api_key=api_key)
    extractor_api = ExtractorApi(api_key=api_key)
    documents = []
    
    for ticker in tickers:
        for form_type in form_types:
            for year in years:
                startDate = f"{year}-01-01"
                endDate = f"{year}-12-31"
                
                query_str = f"ticker:({ticker}) AND formType:(\"{form_type}\") AND periodOfReport:[{startDate} TO {endDate}]"
                query = {
                    "query": { "query_string": { "query": query_str } },
                    "from": "0", "size": "10",
                    "sort": [{ "periodOfReport": { "order": "desc" } }]
                }
                
                try:
                    response = query_api.get_filings(query)
                except Exception as e:
                    logger.error(f"Error querying {ticker} {form_type} {year}: {e}")
                    continue
                    
                filings = response.get('filings', [])
                if not filings:
                    logger.warning(f"No {form_type} found for {ticker} in {year}")
                    continue
                
                limit = 1 if form_type == "10-K" else min(len(filings), 4)
                
                for idx in range(limit):
                    filing = filings[idx]
                    filing_url = filing['linkToFilingDetails']
                    actual_quarter = "Q_Unknown"
                    if quarters and idx < len(quarters):
                        actual_quarter = quarters[idx]
                        
                    secs_to_extract = sections
                    if not secs_to_extract:
                        secs_to_extract = ["1", "1A", "7", "7A"] if form_type == "10-K" else ["part1item1", "part1item2"]
                        
                    for section in secs_to_extract:
                        try:
                            logger.info(f"Extracting Section {section} for {ticker} {form_type} {year} {actual_quarter}")
                            section_text_html = extractor_api.get_section(filing_url, section, "text")
                            
                            if not section_text_html or len(section_text_html.strip()) == 0:
                                logger.warning(f"Section {section} empty or not found. Skipping.")
                                continue
                                
                            clean_txt = _clean_html(section_text_html)
                            clean_txt = clean_txt[:200000] # Cap length to avoid huge vector spikes
                            
                            filepath = _save_document(ticker, year, form_type, actual_quarter, section, clean_txt)
                            
                            meta = {
                                "Ticker": ticker,
                                "Year": year,
                                "Quarter": actual_quarter,
                                "Form Type": form_type,
                                "Section": section,
                                "Filing URL": filing_url,
                                "Source": filepath
                            }
                            
                            documents.append(Document(page_content=clean_txt, metadata=meta))
                            
                        except Exception as e:
                            logger.error(f"Failed to extract {section} for {ticker}: {e}")
                            
    return documents


def _fetch_with_edgar_downloader(
    tickers: List[str], 
    form_types: List[str], 
    years: List[str], 
    quarters: Optional[List[str]],
    sections: Optional[List[str]]
) -> List[Document]:
    """Fallback method using sec-edgar-downloader"""
    from sec_edgar_downloader import Downloader
    
    # Needs valid company and email to comply with SEC rules
    dl = Downloader("FinancialRAG_LangGraph", "finrag@example.com", os.path.dirname(__file__))
    documents = []
    
    for ticker in tickers:
        for form_type in form_types:
            for year in years:
                dl_after = f"{year}-01-01"
                dl_before = f"{int(year)+1}-03-01" 
                amount = 1 if form_type == "10-K" else 4
                
                logger.info(f"Downloading {form_type} for {ticker} ({year}) using sec-edgar-downloader...")
                
                try:
                    dl.get(form_type, ticker, after=dl_after, before=dl_before, amount=amount)
                except Exception as e:
                    logger.error(f"Error downloading {ticker} {form_type}: {e}")
                    continue
                
                # Check sec-edgar-filings directory
                base_dir = os.path.join(os.path.dirname(__file__), "sec-edgar-filings", ticker, form_type)
                if not os.path.exists(base_dir):
                    logger.warning(f"No downloaded folder found at {base_dir}")
                    continue
                    
                accession_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                
                for idx, acc_dir in enumerate(accession_dirs):
                    html_path = os.path.join(base_dir, acc_dir, "primary-document.html")
                    txt_path = os.path.join(base_dir, acc_dir, "full-submission.txt")
                    
                    target_path = html_path if os.path.exists(html_path) else txt_path
                    if not os.path.exists(target_path):
                        continue
                        
                    with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_html = f.read()
                        
                    clean_txt = _clean_html(raw_html)
                    
                    actual_quarter = "Q_Unknown"
                    if quarters and idx < len(quarters):
                        actual_quarter = quarters[idx]
                        
                    secs_to_extract = sections if sections else ["Full"]
                    
                    for section in secs_to_extract:
                        extracted_text = clean_txt
                        if section != "Full":
                            # Attempt generic heuristic to slice text if possible
                            search_str = f"Item {section}."
                            start_idx = clean_txt.find(search_str)
                            if start_idx != -1:
                                extracted_text = clean_txt[start_idx : start_idx + 100000] # Arbitrary chunk
                            else:
                                logger.warning(f"Could not precisely isolate {section} using fallback. Returning partial document.")
                                extracted_text = clean_txt[:100000]
                                
                        filepath = _save_document(ticker, year, form_type, actual_quarter, section, extracted_text)
                        
                        meta = {
                            "Ticker": ticker,
                            "Year": year,
                            "Quarter": actual_quarter,
                            "Form Type": form_type,
                            "Section": section,
                            "Source": filepath
                        }
                        
                        documents.append(Document(page_content=extracted_text, metadata=meta))
                        
    return documents
