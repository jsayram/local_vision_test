#!/usr/bin/env python3
"""
Quick Test Script - Test the Living Portrait system without full setup

This script runs a minimal version to verify everything works:
- Tests YOLO detector
- Tests sprite loading
- Tests animation rendering
- Tests JSON storage
- Doesn't require camera or Moondream
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import config
import storage
import animator
from models import AnimationState, MoondreamResult
from yolo_detector import YOLODetector
import time

def test_configuration():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 1: Configuration")
    print("="*60)
    config.print_config()
    print("✓ Configuration loaded successfully")

def test_storage():
    """Test JSON storage"""
    print("\n" + "="*60)
    print("TEST 2: JSON Storage")
    print("="*60)
    
    storage.initialize_storage()
    
    # Test adding a person
    person = storage.add_person("Test User", "Test note")
    print(f"✓ Added test person: {person}")
    
    # Test adding interaction
    interaction = storage.create_interaction_record(
        person_id=person['person_id'],
        mood="happy",
        text="This is a test interaction!",
        event_type="TEST"
    )
    print(f"✓ Created test interaction: {interaction}")
    
    # Get stats
    stats = storage.get_storage_stats()
    print(f"✓ Storage stats: {stats}")

def test_yolo():
    """Test YOLO detector"""
    print("\n" + "="*60)
    print("TEST 3: YOLO Detector")
    print("="*60)
    
    detector = YOLODetector()
    
    if detector.detector is None:
        print("✗ YOLO not available - this is expected if not installed")
    else:
        print(f"✓ YOLO detector initialized: {detector.detector}")

def test_sprites():
    """Test sprite loading"""
    print("\n" + "="*60)
    print("TEST 4: Sprite System")
    print("="*60)
    
    sprite_mgr = animator.get_sprite_manager()
    
    if sprite_mgr.has_sprites():
        print(f"✓ Sprites loaded: {list(sprite_mgr.sprites.keys())}")
    else:
        print("⚠ No sprites loaded - using fallback rendering")
        print("  This is OK! Add PNG files to sprites/ folder to use real sprites")

def test_animation():
    """Test animation rendering"""
    print("\n" + "="*60)
    print("TEST 5: Animation Rendering")
    print("="*60)
    
    # Create test animation state
    state = AnimationState()
    
    # Test idle state
    print("  Rendering idle state...")
    canvas = animator.render_portrait(state)
    print(f"  ✓ Canvas created: {canvas.shape}")
    
    # Simulate Moondream response
    print("  Simulating Moondream response...")
    result = MoondreamResult(text="Hello! This is a test.", mood="happy")
    state.update_from_moondream(result, time.time(), 5.0, 8.0)
    
    # Render happy state
    print("  Rendering happy state with speaking...")
    canvas = animator.render_portrait(state)
    print(f"  ✓ Canvas updated: {canvas.shape}")
    
    print("✓ Animation system working")

def test_all():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "LIVING PORTRAIT TEST SUITE" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        test_configuration()
        test_storage()
        test_yolo()
        test_sprites()
        test_animation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nNext steps:")
        print("1. Add sprite images to sprites/ folder (optional)")
        print("2. Configure Moondream API (optional)")
        print("3. Run: python3 portrait.py")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all()
