#!/usr/bin/env python3
"""
Performance test script for the advisor endpoint.
Tests response times for offline vs online modes.
"""

import sys
import os
import asyncio
import time
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas.api import AdviseRequest, UserProfile, SkillDetail
from services.advisor_service import advise
from services.retriever import Retriever


async def test_advisor_performance(search_online: bool = False, iterations: int = 3) -> Dict[str, Any]:
    """Test advisor performance with multiple iterations."""
    mode = "ONLINE" if search_online else "OFFLINE"
    print(f"\n🚀 Testing {mode} Mode Performance")
    print("=" * 50)

    # Create a test profile
    profile = UserProfile(
        current_skills=[
            SkillDetail(name="python", expertise="Intermediate"),
            SkillDetail(name="sql", expertise="Beginner"),
            SkillDetail(name="statistics", expertise="Intermediate")
        ],
        goal_role="data scientist",
        years_experience=3
    )

    request = AdviseRequest(
        profile=profile,
        search_online=search_online
    )

    # Initialize retriever once
    retriever = Retriever()
    # Retriever uses lazy initialization, so we'll trigger it with a dummy search
    await retriever._ensure_bm25()

    response_times = []
    successful_requests = 0

    for i in range(iterations):
        print(f"   Test {i+1}/{iterations}...", end=" ")

        try:
            start_time = time.time()
            result = await advise(request, retriever, top_k=5)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)
            successful_requests += 1

            print(f"✅ {response_time:.3f}s")

            # Brief delay between requests
            if i < iterations - 1:
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)

        print(f"\n📊 {mode} Performance Results:")
        print(f"   • Successful requests: {successful_requests}/{iterations}")
        print(f"   • Average response time: {avg_time:.3f}s")
        print(f"   • Fastest response: {min_time:.3f}s")
        print(f"   • Slowest response: {max_time:.3f}s")

        # Check if offline mode meets the 2.5s requirement
        if not search_online:
            target_time = 2.5
            meets_requirement = avg_time <= target_time
            print(f"   • Target: ≤{target_time}s - {'✅ PASS' if meets_requirement else '❌ FAIL'}")

        return {
            'mode': mode,
            'search_online': search_online,
            'successful_requests': successful_requests,
            'total_requests': iterations,
            'response_times': response_times,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'meets_requirement': not search_online and avg_time <= 2.5
        }
    else:
        print(f"❌ No successful requests in {mode} mode")
        return {
            'mode': mode,
            'search_online': search_online,
            'successful_requests': 0,
            'total_requests': iterations,
            'response_times': [],
            'avg_time': 0,
            'min_time': 0,
            'max_time': 0,
            'meets_requirement': False
        }


async def test_component_performance():
    """Test individual component performance."""
    print(f"\n🔧 Component Performance Analysis")
    print("=" * 50)

    # Test retriever initialization
    print("   Retriever initialization...", end=" ")
    start_time = time.time()
    retriever = Retriever()
    await retriever._ensure_bm25()
    init_time = time.time() - start_time
    print(f"✅ {init_time:.3f}s")

    # Test course retrieval
    print("   Course retrieval...", end=" ")
    from services.retriever import RetrievalQuery
    query = RetrievalQuery(skills=["python", "data science"], target_skills=["machine learning"])

    start_time = time.time()
    courses = await retriever.hybrid_search(query, top_k=10)
    retrieval_time = time.time() - start_time
    print(f"✅ {retrieval_time:.3f}s ({len(courses)} courses)")

    print(f"\n📊 Component Timings:")
    print(f"   • Retriever init: {init_time:.3f}s")
    print(f"   • Course retrieval: {retrieval_time:.3f}s")
    print(f"   • Total base overhead: {init_time + retrieval_time:.3f}s")

    return {
        'init_time': init_time,
        'retrieval_time': retrieval_time,
        'total_overhead': init_time + retrieval_time
    }


async def main():
    """Run comprehensive performance tests."""
    print("🚀 Starting Advisor Performance Test Suite")
    print("=" * 60)

    # Test component performance first
    component_results = await test_component_performance()

    # Test offline performance (target: <2.5s)
    offline_results = await test_advisor_performance(search_online=False, iterations=5)

    # Test online performance for comparison
    online_results = await test_advisor_performance(search_online=True, iterations=3)

    # Summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE TEST SUMMARY")
    print("=" * 60)

    print(f"🔧 Component Performance:")
    print(f"   • Base overhead: {component_results['total_overhead']:.3f}s")

    print(f"\n⚡ Offline Mode (target: ≤2.5s):")
    if offline_results['successful_requests'] > 0:
        print(f"   • Average: {offline_results['avg_time']:.3f}s")
        print(f"   • Range: {offline_results['min_time']:.3f}s - {offline_results['max_time']:.3f}s")
        print(f"   • Status: {'✅ MEETS REQUIREMENT' if offline_results['meets_requirement'] else '❌ EXCEEDS TARGET'}")
    else:
        print("   • ❌ No successful offline requests")

    print(f"\n🌐 Online Mode (for comparison):")
    if online_results['successful_requests'] > 0:
        print(f"   • Average: {online_results['avg_time']:.3f}s")
        print(f"   • Range: {online_results['min_time']:.3f}s - {online_results['max_time']:.3f}s")
    else:
        print("   • ❌ No successful online requests")

    # Performance improvement suggestions
    if offline_results['successful_requests'] > 0 and not offline_results['meets_requirement']:
        print(f"\n💡 Performance Improvement Suggestions:")
        print(f"   • Current avg: {offline_results['avg_time']:.3f}s (target: ≤2.5s)")
        print(f"   • Overhead: {component_results['total_overhead']:.3f}s")
        print(f"   • Processing: {offline_results['avg_time'] - component_results['total_overhead']:.3f}s")

        if component_results['total_overhead'] > 1.0:
            print("   • Consider caching retriever initialization")
        if offline_results['avg_time'] - component_results['total_overhead'] > 1.0:
            print("   • Consider optimizing scoring algorithms")

    print("\n🎉 Performance test completed!")


if __name__ == "__main__":
    asyncio.run(main())