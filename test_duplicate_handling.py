#!/usr/bin/env python3
"""
Test script for duplicate handling functionality.
Tests duplicate detection, removal, and prevention.
"""

import sys
import os
import json
import asyncio
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.advisor_service import _remove_duplicates_from_json
from pathlib import Path


def analyze_duplicates(courses_file: str = "courses.json") -> Dict[str, Any]:
    """Analyze the courses.json file for duplicates."""
    print(f"🔍 Analyzing duplicates in {courses_file}")
    print("=" * 50)

    try:
        with open(courses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            courses = data
        elif isinstance(data, dict) and 'courses' in data:
            courses = data['courses']
        else:
            print("❌ Unexpected JSON structure")
            return {}

        total_courses = len(courses)
        print(f"📚 Total courses: {total_courses}")

        # Track duplicates
        seen_ids = {}
        seen_titles = {}
        seen_urls = {}

        id_duplicates = []
        title_duplicates = []
        url_duplicates = []

        for i, course in enumerate(courses):
            if not isinstance(course, dict):
                continue

            course_id = course.get('course_id', '')
            title = course.get('title', '').lower().strip()
            url = course.get('url', '').strip() if course.get('url') else None

            # Check for ID duplicates
            if course_id in seen_ids:
                id_duplicates.append({
                    'course_id': course_id,
                    'title': course.get('title', ''),
                    'original_index': seen_ids[course_id],
                    'duplicate_index': i
                })
            else:
                seen_ids[course_id] = i

            # Check for title duplicates
            if title in seen_titles:
                title_duplicates.append({
                    'title': course.get('title', ''),
                    'course_id': course_id,
                    'original_index': seen_titles[title],
                    'duplicate_index': i
                })
            else:
                seen_titles[title] = i

            # Check for URL duplicates
            if url and url in seen_urls:
                url_duplicates.append({
                    'url': url,
                    'title': course.get('title', ''),
                    'course_id': course_id,
                    'original_index': seen_urls[url],
                    'duplicate_index': i
                })
            elif url:
                seen_urls[url] = i

        # Report findings
        print(f"\n📊 Duplicate Analysis Results:")
        print(f"   • ID duplicates: {len(id_duplicates)}")
        print(f"   • Title duplicates: {len(title_duplicates)}")
        print(f"   • URL duplicates: {len(url_duplicates)}")

        if id_duplicates:
            print(f"\n🔴 ID Duplicates:")
            for dup in id_duplicates[:5]:  # Show first 5
                print(f"   • ID: {dup['course_id']} - '{dup['title']}'")

        if title_duplicates:
            print(f"\n🟡 Title Duplicates:")
            for dup in title_duplicates[:5]:  # Show first 5
                print(f"   • '{dup['title']}' (ID: {dup['course_id']})")

        if url_duplicates:
            print(f"\n🟠 URL Duplicates:")
            for dup in url_duplicates[:5]:  # Show first 5
                print(f"   • {dup['url']} - '{dup['title']}'")

        total_duplicates = len(id_duplicates) + len(title_duplicates) + len(url_duplicates)
        unique_courses = total_courses - total_duplicates

        return {
            'total_courses': total_courses,
            'id_duplicates': len(id_duplicates),
            'title_duplicates': len(title_duplicates),
            'url_duplicates': len(url_duplicates),
            'total_duplicates': total_duplicates,
            'unique_courses': unique_courses,
            'duplicate_details': {
                'id_duplicates': id_duplicates,
                'title_duplicates': title_duplicates,
                'url_duplicates': url_duplicates
            }
        }

    except Exception as e:
        print(f"❌ Error analyzing duplicates: {e}")
        return {}


async def test_deduplication(courses_file: str = "courses.json"):
    """Test the deduplication functionality."""
    print(f"\n🧪 Testing Deduplication Functionality")
    print("=" * 50)

    # Analyze before deduplication
    before_analysis = analyze_duplicates(courses_file)

    if not before_analysis:
        print("❌ Could not analyze file before deduplication")
        return

    if before_analysis['total_duplicates'] == 0:
        print("✅ No duplicates found - file is already clean!")
        return

    print(f"\n🔧 Running deduplication...")

    try:
        # Run deduplication
        success = await _remove_duplicates_from_json(courses_file)

        if success:
            print("✅ Deduplication completed successfully")

            # Analyze after deduplication
            after_analysis = analyze_duplicates(courses_file)

            if after_analysis:
                print(f"\n📈 Deduplication Results:")
                print(f"   • Before: {before_analysis['total_courses']} courses ({before_analysis['total_duplicates']} duplicates)")
                print(f"   • After: {after_analysis['total_courses']} courses ({after_analysis['total_duplicates']} duplicates)")
                print(f"   • Removed: {before_analysis['total_courses'] - after_analysis['total_courses']} duplicates")

                if after_analysis['total_duplicates'] == 0:
                    print("🎉 All duplicates successfully removed!")
                else:
                    print(f"⚠️  {after_analysis['total_duplicates']} duplicates still remain")

        else:
            print("❌ Deduplication failed")

    except Exception as e:
        print(f"❌ Error during deduplication test: {e}")


def test_course_id_generation():
    """Test the improved course ID generation."""
    print(f"\n🔧 Testing Course ID Generation")
    print("=" * 50)

    import hashlib

    test_courses = [
        {"title": "Python for Data Science", "url": "https://coursera.org/learn/python-data-science", "provider": "Coursera"},
        {"title": "Python for Data Science", "url": "https://coursera.org/learn/python-data-science", "provider": "Coursera"},  # Exact duplicate
        {"title": "Python for Data Science", "url": "https://udemy.com/python-data-science", "provider": "Udemy"},  # Same title, different URL
        {"title": "Machine Learning Basics", "url": "https://edx.org/ml-basics", "provider": "edX"},
    ]

    generated_ids = []

    for i, course in enumerate(test_courses):
        title = course['title']
        url = course['url']
        provider = course['provider'].lower()

        # Generate ID using the new method
        content_hash = hashlib.md5(f"{title.lower().strip()}|{url}".encode('utf-8')).hexdigest()[:8]
        course_id = f"online-{provider}-{content_hash}"

        generated_ids.append(course_id)

        print(f"   {i+1}. '{title}' ({provider})")
        print(f"      URL: {url}")
        print(f"      Generated ID: {course_id}")
        print()

    # Check for duplicates in generated IDs
    unique_ids = set(generated_ids)
    print(f"📊 ID Generation Results:")
    print(f"   • Total courses: {len(test_courses)}")
    print(f"   • Unique IDs generated: {len(unique_ids)}")
    print(f"   • Duplicates prevented: {len(test_courses) - len(unique_ids)}")

    if len(unique_ids) == len(test_courses) - 1:  # We expect 1 duplicate (courses 1 and 2)
        print("✅ Course ID generation working correctly!")
    else:
        print("⚠️  Unexpected ID generation behavior")


async def main():
    """Run all duplicate handling tests."""
    print("🚀 Starting Duplicate Handling Test Suite")
    print("=" * 60)

    # Test 1: Analyze current duplicates
    analyze_duplicates("courses.json")

    # Test 2: Test course ID generation
    test_course_id_generation()

    # Test 3: Test deduplication (only if duplicates exist)
    await test_deduplication("courses.json")

    print("\n" + "=" * 60)
    print("🎉 Duplicate Handling Test Suite Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())