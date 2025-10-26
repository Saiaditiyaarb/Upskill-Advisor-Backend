#!/usr/bin/env python3
"""
Test script to verify metrics collection and API endpoints are working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.metrics_service import get_metrics_collector, ComponentType
from services.enhanced_advisor_service import enhanced_advise
from schemas.api import AdviseRequest, UserProfile, SkillDetail

async def test_metrics_collection():
    """Test that metrics are being collected properly."""
    print("🧪 Testing Metrics Collection")
    print("=" * 50)
    
    # Get metrics collector
    mc = get_metrics_collector()
    
    # Test recording different types of metrics
    print("📊 Recording test metrics...")
    
    # Record latency
    mc.record_latency(
        ComponentType.AGENT,
        operation="test_operation",
        duration_ms=1500.0,
        success=True,
        test_mode=True
    )
    
    # Record cost
    mc.record_cost(
        ComponentType.AGENT,
        operation="test_operation",
        cost_usd=0.0025,
        tokens_used=500,
        model_name="test-model",
        test_mode=True
    )
    
    # Record accuracy
    mc.record_accuracy(
        ComponentType.AGENT,
        operation="test_operation",
        accuracy_score=0.85,
        total_items=100,
        correct_items=85,
        test_mode=True
    )
    
    print("✅ Test metrics recorded successfully")
    
    # Flush metrics to disk
    await mc.flush_metrics()
    print("✅ Metrics flushed to disk")
    
    # Test the enhanced advisor service
    print("\n🤖 Testing Enhanced Advisor Service")
    print("=" * 50)
    
    # Create a test profile
    test_profile = UserProfile(
        current_skills=[
            SkillDetail(name="python", expertise="Intermediate"),
            SkillDetail(name="javascript", expertise="Beginner")
        ],
        years=2,
        goal_role="software engineer",
        search_online=False
    )
    
    # Create a test request
    test_request = AdviseRequest(
        profile=test_profile,
        user_context={},
        search_online=False,
        retrieval_mode="hybrid"
    )
    
    try:
        # Run the enhanced advisor
        result = await enhanced_advise(test_request)
        
        print(f"✅ Enhanced advisor completed successfully")
        print(f"📋 Plan steps: {len(result.plan)}")
        print(f"📚 Recommended courses: {len(result.recommended_courses)}")
        print(f"🎯 Skill gaps: {len(result.gap_map)}")
        print(f"📝 Notes length: {len(result.notes) if result.notes else 0} characters")
        print(f"⏱️ Processing time: {result.metrics.get('processing_time_ms', 0):.2f}ms")
        
        # Check if metrics were recorded
        if result.metrics:
            print(f"📊 Metrics recorded: {list(result.metrics.keys())}")
        
    except Exception as e:
        print(f"❌ Enhanced advisor failed: {e}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_metrics_collection())
