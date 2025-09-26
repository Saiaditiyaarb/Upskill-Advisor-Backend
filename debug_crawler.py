#!/usr/bin/env python3
"""
Debug script to inspect HTML structure from course websites.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import random


def debug_coursera(query: str = "python"):
    """Debug Coursera HTML structure."""
    print(f"\n🔍 Debugging Coursera for query: '{query}'")
    print("=" * 50)

    try:
        encoded_query = quote(query)
        url = f"https://www.coursera.org/search?query={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        print(f"✅ Successfully fetched Coursera page")
        print(f"📄 Page title: {soup.title.string if soup.title else 'No title'}")

        # Try to find any divs that might contain courses
        potential_selectors = [
            "div[data-testid]",
            "div[data-click-key]",
            "div[class*='card']",
            "div[class*='Card']",
            "div[class*='search']",
            "div[class*='result']",
            "article",
            "li"
        ]

        for selector in potential_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"\n🎯 Found {len(elements)} elements with selector: {selector}")

                # Show first few elements
                for i, elem in enumerate(elements[:3]):
                    elem_text = elem.get_text(strip=True)[:100]
                    elem_attrs = {k: v for k, v in elem.attrs.items() if k in ['class', 'data-testid', 'data-click-key', 'data-purpose']}
                    print(f"   {i+1}. Text: {elem_text}...")
                    print(f"      Attrs: {elem_attrs}")

                    # Look for links
                    links = elem.select("a[href]")
                    if links:
                        for link in links[:2]:
                            href = link.get('href')
                            if href and ('/learn/' in href or '/specializations/' in href):
                                print(f"      🔗 Course link: {href}")

                if len(elements) > 10:  # Likely found course cards
                    break

        # Save a sample of the HTML for manual inspection
        with open('coursera_debug.html', 'w', encoding='utf-8') as f:
            f.write(str(soup.prettify()))
        print(f"\n💾 Full HTML saved to: coursera_debug.html")

    except Exception as e:
        print(f"❌ Error debugging Coursera: {e}")


def debug_udemy(query: str = "python"):
    """Debug Udemy HTML structure."""
    print(f"\n🔍 Debugging Udemy for query: '{query}'")
    print("=" * 50)

    try:
        encoded_query = quote(query)
        url = f"https://www.udemy.com/courses/search/?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        session = requests.Session()
        session.headers.update(headers)

        time.sleep(random.uniform(0.5, 1.0))

        response = session.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        print(f"✅ Successfully fetched Udemy page")
        print(f"📄 Page title: {soup.title.string if soup.title else 'No title'}")

        # Look for course-related elements
        potential_selectors = [
            "div[data-purpose]",
            "div[data-testid]",
            "div[class*='course']",
            "div[class*='card']",
            "article",
            "li"
        ]

        for selector in potential_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"\n🎯 Found {len(elements)} elements with selector: {selector}")

                # Show first few elements
                for i, elem in enumerate(elements[:3]):
                    elem_text = elem.get_text(strip=True)[:100]
                    elem_attrs = {k: v for k, v in elem.attrs.items() if k in ['class', 'data-testid', 'data-purpose']}
                    print(f"   {i+1}. Text: {elem_text}...")
                    print(f"      Attrs: {elem_attrs}")

                    # Look for links
                    links = elem.select("a[href]")
                    if links:
                        for link in links[:2]:
                            href = link.get('href')
                            if href and '/course/' in href:
                                print(f"      🔗 Course link: {href}")

                if len(elements) > 5:  # Likely found something useful
                    break

        # Save a sample of the HTML for manual inspection
        with open('udemy_debug.html', 'w', encoding='utf-8') as f:
            f.write(str(soup.prettify()))
        print(f"\n💾 Full HTML saved to: udemy_debug.html")

    except Exception as e:
        print(f"❌ Error debugging Udemy: {e}")


def main():
    """Run debug analysis."""
    print("🔍 Course Website HTML Structure Debug")
    print("=" * 60)

    debug_coursera("python programming")
    debug_udemy("python programming")

    print("\n" + "=" * 60)
    print("🎉 Debug analysis completed!")
    print("📁 Check the generated HTML files for manual inspection:")
    print("   • coursera_debug.html")
    print("   • udemy_debug.html")


if __name__ == "__main__":
    main()