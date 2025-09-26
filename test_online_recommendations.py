#!/usr/bin/env python3
"""
Test script to verify that online courses are properly prioritized in recommendations.
"""

import sys
import os
import json
import asyncio
from typing import List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas.api import AdviseRequest, UserProfile, UserSkill
from schemas.course import Course
from services.advisor_service import advise
from services.retriever import Retriever


async def test_online_course_prioritization():
    """Test that online courses are prioritized in recommendations when online search is enabled."""
    print("🧪 Testing Online Course Prioritization")
    print("=" * 50)

    # Create a test profile for data scientist role
    profile = UserProfile(
        current_skills=[
            UserSkill(name="python", expertise="beginner"),
            UserSkill(name="statistics", expertise="intermediate")
        ],
        goal_role="data scientist",
        years_experience=2
    )

    # Create request with online search enabled
    request = AdviseRequest(
        profile=profile,
        search_online=True  # This should trigger online course fetching
    )

    try:
        # Initialize retriever
        retriever = Retriever()
        await retriever.initialize()

        print(f"📚 Loaded {len(retriever.courses)} courses from local database")

        # Get recommendations
        print("🔍 Getting recommendations with online search enabled...")
        result = await advise(request, retriever, top_k=5)

        print(f"\n📋 Recommended Courses ({len(result.recommended_courses)}):")
        online_count = 0
        local_count = 0

        for i, course in enumerate(result.recommended_courses):
            is_online = course.metadata and course.metadata.get('auto_added', False)
            course_type = "🌐 ONLINE" if is_online else "💾 LOCAL"

            if is_online:
                online_count += 1
            else:
                local_count += 1

            print(f"   {i+1}. {course_type} - {course.title}")
            print(f"      Provider: {course.provider}")
            print(f"      Skills: {', '.join(course.skills)}")
            print(f"      URL: {course.url}")
            print()

        print(f"📊 Summary:")
        print(f"   • Online courses: {online_count}")
        print(f"   • Local courses: {local_count}")
        print(f"   • Total: {len(result.recommended_courses)}")

        # Check if online courses are prioritized
        if online_count > 0:
            first_course_is_online = (result.recommended_courses[0].metadata and
                                    result.recommended_courses[0].metadata.get('auto_added', False))
            if first_course_is_online:
                print("✅ SUCCESS: Online courses are properly prioritized!")
            else:
                print("⚠️  WARNING: Online courses found but not prioritized in top position")
        else:
            print("ℹ️  INFO: No online courses found (may be due to rate limiting or network issues)")

        # Show learning plan
        if result.plan:
            print(f"\n📝 Learning Plan ({len(result.plan)} steps):")
            for i, step in enumerate(result.plan[:3]):  # Show first 3 steps
                print(f"   {i+1}. {step.get('skill', 'N/A')} - {step.get('resource', 'N/A')}")

        print(f"\n💡 Notes: {result.notes}")

        return {
            'success': True,
            'online_courses': online_count,
            'local_courses': local_count,
            'total_courses': len(result.recommended_courses),
            'online_prioritized': online_count > 0 and (result.recommended_courses[0].metadata and
                                                       result.recommended_courses[0].metadata.get('auto_added', False))
        }

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return {
            'success': False,
            'error': str(e),
            'online_courses': 0,
            'local_courses': 0,
            'total_courses': 0,
            'online_prioritized': False
        }


async def main():
    """Run the test."""
    print("🚀 Starting Online Course Prioritization Test")
    print("=" * 60)

    result = await test_online_course_prioritization()

    print("\n" + "=" * 60)
    print("📈 TEST RESULTS")
    print("=" * 60)

    if result['success']:
        print("✅ Test completed successfully!")
        print(f"📊 Results:")
        print(f"   • Online courses found: {result['online_courses']}")
        print(f"   • Local courses included: {result['local_courses']}")
        print(f"   • Total recommendations: {result['total_courses']}")
        print(f"   • Online courses prioritized: {'✅ Yes' if result['online_prioritized'] else '❌ No'}")
    else:
        print("❌ Test failed!")
        print(f"Error: {result['error']}")

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())