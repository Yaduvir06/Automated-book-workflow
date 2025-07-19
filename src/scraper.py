from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os
from typing import Tuple

def fetch_and_parse_chapter(url: str, output_dir: str = "data/screenshots") -> Tuple[str, str]:
    """
    Fetches content and a screenshot from a URL.
    Returns a tuple containing the clean text and the path to the screenshot.
    """
    os.makedirs(output_dir, exist_ok=True)
    screenshot_filename = url.split("/")[-1] + ".png"
    screenshot_path = os.path.join(output_dir, screenshot_filename)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.screenshot(path=screenshot_path, full_page=True)  # Fixed indentation
        content = page.content()
        browser.close()

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    main_content = soup.find('div', id='mw-content-text')

    if main_content:
        for nav in main_content.find_all(class_='noprint'):
            nav.decompose()
        paragraphs = main_content.find_all('p')
        clean_text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
    else:
        clean_text = "No content found."

    return clean_text, screenshot_path  # Fixed return values

def scrape_book(base_url: str, start_chapter: int, end_chapter: int, output_dir: str = "data/scraped_content"):
    """Scrapes a range of chapters and saves their text content to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for chapter in range(start_chapter, end_chapter + 1):
        chapter_url = f"{base_url}/Chapter_{chapter}"
        print(f"Fetching {chapter_url}...")
        
        chapter_text, _ = fetch_and_parse_chapter(chapter_url)  # Fixed unpacking
        
        file_path = os.path.join(output_dir, f"chapter_{chapter}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        print(f"Saved content to {file_path}")

if __name__ == "__main__":
    book_base_url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1"
    scrape_book(book_base_url, 1, 13)
