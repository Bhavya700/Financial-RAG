import os
import sys
import logging

from sec_extractor import fetch_sec_documents

logging.basicConfig(level=logging.INFO)

def run_tests():
    print("=====================================================")
    print(" Starting tests for SEC Extractor Module (Phase 1)   ")
    print("=====================================================")
    
    # If SEC_API_KEY is missing, it will use the fallback downloader.
    if not os.getenv("SEC_API_KEY"):
        print("\nNOTE: 'SEC_API_KEY' environment variable is NOT set.")
        print("      Extracting precise sections (Item 1A, Item 7) requires sec_api.")
        print("      Using fallback sec-edgar-downloader instead (will return full text / rough slice).\n")
    
    # Test 1: Apple 10-K from 2023 (Item 1A and Item 7)
    print("[Test 1] Fetching AAPL 10-K for 2023 (Sections: 1A, 7)")
    docs_aapl = fetch_sec_documents(
        tickers=["AAPL"],
        form_types=["10-K"],
        years=["2023"],
        sections=["1A", "7"]
    )
    
    print(f"\n-> Number of AAPL documents extracted: {len(docs_aapl)}")
    for d in docs_aapl:
        meta = d.metadata
        print(f" -> {meta.get('Ticker')} | {meta.get('Form Type')} | Section: {meta.get('Section')}")
        print(f"    Source: {meta.get('Source')}")
        print(f"    Content Preview: {d.page_content[:150].replace(chr(10), ' ')}...\n")
        
    print("=====================================================")
        
    # Test 2: Microsoft 10-Q from 2023 Q1 (Part 1, Item 1)
    print("\n[Test 2] Fetching MSFT 10-Q for 2023 (Quarter: Q1, Section: part1item1)")
    docs_msft = fetch_sec_documents(
        tickers=["MSFT"],
        form_types=["10-Q"],
        years=["2023"],
        quarters=["Q1"],
        sections=["part1item1"]
    )
    
    print(f"\n-> Number of MSFT documents extracted: {len(docs_msft)}")
    for d in docs_msft:
        meta = d.metadata
        print(f" -> {meta.get('Ticker')} | {meta.get('Form Type')} | Section: {meta.get('Section')} | Quarter: {meta.get('Quarter')}")
        print(f"    Source: {meta.get('Source')}")
        print(f"    Content Preview: {d.page_content[:150].replace(chr(10), ' ')}...\n")

    print("=====================================================")
    print(" Tests complete. Check the 'data/' directory for exported files.")
    print("=====================================================")

if __name__ == "__main__":
    run_tests()
