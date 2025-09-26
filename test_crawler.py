#!/usr/bin/env python3
"""
Test script for the improved crawler service.
Tests real online course fetching with proper URLs.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.crawler_service import crawl_courses


def test_crawler_with_query(query: str, max_courses: int = 5) -> Dict[str, Any]:
    """Test crawler with a specific query and return results."""
    print(f"\n🔍 Testing crawler with query: '{query}'")
    print("=" * 50)

    start_time = time.time()

    try:
        courses = crawl_courses(query=query, max_courses=max_courses)
        end_time = time.time()

        duration = end_time - start_time

        print(f"⏱️  Crawling completed in {duration:.2f} seconds")
        print(f"📚 Found {len(courses)} courses")

        # Analyze results
        providers = {}
        courses_with_urls = 0
        courses_with_skills = 0

        for course in courses:
            provider = course.get('provider', 'Unknown')
            providers[provider] = providers.get(provider, 0) + 1

            if course.get('url'):
                courses_with_urls += 1
            if course.get('skills') and len(course.get('skills', [])) > 0:
                courses_with_skills += 1

        print(f"\n📊 Results Summary:")
        print(f"   • Courses with URLs: {courses_with_urls}/{len(courses)} ({courses_with_urls/len(courses)*100:.1f}%)")
        print(f"   • Courses with skills: {courses_with_skills}/{len(courses)} ({courses_with_skills/len(courses)*100:.1f}%)")
        print(f"   • Providers: {', '.join([f'{k}: {v}' for k, v in providers.items()])}")

        # Show sample courses
        print(f"\n📋 Sample Courses:")
        for i, course in enumerate(courses[:3]):
            print(f"   {i+1}. {course.get('title', 'No title')}")
            print(f"      Provider: {course.get('provider', 'Unknown')}")
            print(f"      URL: {course.get('url', 'No URL')}")
            print(f"      Skills: {', '.join(course.get('skills', []))}")
            print(f"      Difficulty: {course.get('difficulty', 'Unknown')}")
            print()

        return {
            'query': query,
            'duration': duration,
            'total_courses': len(courses),
            'courses_with_urls': courses_with_urls,
            'courses_with_skills': courses_with_skills,
            'providers': providers,
            'courses': courses
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"❌ Error during crawling: {e}")
        print(f"⏱️  Failed after {duration:.2f} seconds")

        return {
            'query': query,
            'duration': duration,
            'error': str(e),
            'total_courses': 0,
            'courses_with_urls': 0,
            'courses_with_skills': 0,
            'providers': {},
            'courses': []
        }


def main():
    """Run comprehensive crawler tests."""
    print("🚀 Starting Crawler Test Suite")
    print("=" * 60)

    # Test queries
    test_queries = [
        "python programming",
        "data science",
        "web development",
        "machine learning"
    ]

    all_results = []
    total_start_time = time.time()

    for query in test_queries:
        result = test_crawler_with_query(query, max_courses=6)
        all_results.append(result)

        # Small delay between tests
        time.sleep(1)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Overall summary
    print("\n" + "=" * 60)
    print("📈 OVERALL TEST RESULTS")
    print("=" * 60)

    total_courses = sum(r['total_courses'] for r in all_results)
    total_with_urls = sum(r['courses_with_urls'] for r in all_results)
    total_with_skills = sum(r['courses_with_skills'] for r in all_results)

    print(f"⏱️  Total test duration: {total_duration:.2f} seconds")
    print(f"📚 Total courses found: {total_courses}")
    print(f"🔗 Courses with URLs: {total_with_urls}/{total_courses} ({total_with_urls/total_courses*100:.1f}%)" if total_courses > 0 else "🔗 No courses found")
    print(f"🎯 Courses with skills: {total_with_skills}/{total_courses} ({total_with_skills/total_courses*100:.1f}%)" if total_courses > 0 else "🎯 No courses found")

    # Provider breakdown
    all_providers = {}
    for result in all_results:
        for provider, count in result['providers'].items():
            all_providers[provider] = all_providers.get(provider, 0) + count

    if all_providers:
        print(f"🏢 Provider breakdown: {', '.join([f'{k}: {v}' for k, v in all_providers.items()])}")

    # Error summary
    errors = [r for r in all_results if 'error' in r]
    if errors:
        print(f"❌ Queries with errors: {len(errors)}/{len(test_queries)}")
        for error_result in errors:
            print(f"   • {error_result['query']}: {error_result['error']}")
    else:
        print("✅ All queries completed successfully!")

    # Save detailed results
    try:
        with open('crawler_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: crawler_test_results.json")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    main()