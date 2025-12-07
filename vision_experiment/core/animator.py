"""
Animation System - Sprite-based rendering for the Living Portrait
Uses OpenCV for rendering with PNG sprite overlays
"""
import cv2
import numpy as np
import os
from typing import Dict, Optional
import time

from models.models import AnimationState
from core import config
from core import sprite_generator

# ============================================================================
# SPRITE LOADING
# ============================================================================

class SpriteManager:
    """Manages loading and caching of sprite images"""
    
    def __init__(self):
        self.sprites: Dict[str, Optional[np.ndarray]] = {}
        self.sprites_available = False
        # Try to generate sprites if they don't exist
        sprite_generator.ensure_sprites_exist()
        self.load_all_sprites()
    
    def load_all_sprites(self):
        """Load all sprite images from sprites directory"""
        if not os.path.exists(config.SPRITES_DIR):
            print(f"⚠ Sprites directory not found: {config.SPRITES_DIR}")
            print(f"  Creating directory and using fallback rendering...")
            os.makedirs(config.SPRITES_DIR, exist_ok=True)
            self.sprites_available = False
            return
        
        # Load each sprite
        loaded_count = 0
        for sprite_name, filename in config.SPRITE_FILES.items():
            sprite_path = os.path.join(config.SPRITES_DIR, filename)
            
            if os.path.exists(sprite_path):
                # Load with alpha channel (RGBA)
                sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
                
                if sprite is not None:
                    self.sprites[sprite_name] = sprite
                    loaded_count += 1
                    if config.DEBUG_MODE:
                        print(f"✓ Loaded sprite: {sprite_name} ({sprite.shape})")
                else:
                    print(f"✗ Failed to load sprite: {sprite_path}")
                    self.sprites[sprite_name] = None
            else:
                if config.DEBUG_MODE:
                    print(f"⚠ Sprite not found: {sprite_path}")
                self.sprites[sprite_name] = None
        
        self.sprites_available = loaded_count > 0
        
        if self.sprites_available:
            print(f"✓ Loaded {loaded_count}/{len(config.SPRITE_FILES)} sprites")
        else:
            print(f"⚠ No sprites loaded - using fallback rendering")
    
    def get_sprite(self, sprite_name: str) -> Optional[np.ndarray]:
        """Get a sprite by name"""
        return self.sprites.get(sprite_name)
    
    def has_sprites(self) -> bool:
        """Check if any sprites are loaded"""
        return self.sprites_available


# Global sprite manager instance
_sprite_manager = None

def get_sprite_manager() -> SpriteManager:
    """Get or create global sprite manager"""
    global _sprite_manager
    if _sprite_manager is None:
        _sprite_manager = SpriteManager()
    return _sprite_manager


# ============================================================================
# RENDERING UTILITIES
# ============================================================================

def overlay_sprite(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int, 
                  scale: float = 1.0) -> np.ndarray:
    """
    Overlay a sprite (with alpha channel) onto canvas at position (x, y)
    
    Args:
        canvas: Base image (BGR)
        sprite: Sprite image (BGRA with alpha channel)
        x: X position (top-left corner)
        y: Y position (top-left corner)
        scale: Scale factor for sprite (1.0 = original size)
    
    Returns:
        Canvas with sprite overlaid
    """
    # Scale sprite if needed
    if scale != 1.0:
        new_width = int(sprite.shape[1] * scale)
        new_height = int(sprite.shape[0] * scale)
        sprite = cv2.resize(sprite, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    sprite_h, sprite_w = sprite.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    
    # Calculate region of interest
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas_w, x + sprite_w)
    y2 = min(canvas_h, y + sprite_h)
    
    # Calculate sprite crop (if partially off-canvas)
    sprite_x1 = max(0, -x)
    sprite_y1 = max(0, -y)
    sprite_x2 = sprite_x1 + (x2 - x1)
    sprite_y2 = sprite_y1 + (y2 - y1)
    
    # Extract alpha channel if present
    if sprite.shape[2] == 4:  # BGRA
        sprite_rgb = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2, :3]
        alpha = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2, 3] / 255.0
        alpha = np.dstack([alpha] * 3)  # Make 3-channel
    else:  # BGR (no alpha)
        sprite_rgb = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
        alpha = np.ones((sprite_y2 - sprite_y1, sprite_x2 - sprite_x1, 3))
    
    # Blend sprite onto canvas
    canvas_roi = canvas[y1:y2, x1:x2]
    canvas[y1:y2, x1:x2] = (alpha * sprite_rgb + (1 - alpha) * canvas_roi).astype(np.uint8)
    
    return canvas


def draw_fallback_sprite(canvas: np.ndarray, mood: str, x: int, y: int, 
                        width: int, height: int) -> np.ndarray:
    """
    Draw a simple colored shape as fallback when sprites unavailable
    
    Args:
        canvas: Canvas to draw on
        mood: Current mood (determines color and shape)
        x, y: Top-left position
        width, height: Rectangle dimensions
    
    Returns:
        Canvas with shape drawn
    """
    # Map moods to display names (remove 'talking_' prefix)
    display_mood = mood.replace('talking_', '').replace('_', ' ').upper()
    color = config.FALLBACK_COLORS.get(mood, (128, 128, 128))
    
    # Draw filled circle instead of rectangle
    center_x = x + width // 2
    center_y = y + height // 2
    radius = min(width, height) // 3
    
    cv2.circle(canvas, (center_x, center_y), radius, color, -1)
    cv2.circle(canvas, (center_x, center_y), radius, (255, 255, 255), 3)
    
    # Draw mood text below circle
    text_size = cv2.getTextSize(display_mood, config.FONT_FACE, 0.7, 2)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + radius + 30
    cv2.putText(canvas, display_mood, (text_x, text_y), config.FONT_FACE, 
                0.7, (255, 255, 255), 2)
    
    return canvas


def draw_subtitle(canvas: np.ndarray, text: str, y_position: Optional[int] = None) -> np.ndarray:
    """
    Draw subtitle text at bottom of canvas
    
    Args:
        canvas: Canvas to draw on
        text: Subtitle text
        y_position: Y position (None = auto at bottom)
    
    Returns:
        Canvas with subtitle drawn
    """
    if not text:
        return canvas
    
    canvas_h, canvas_w = canvas.shape[:2]
    
    # Word wrap text if too long
    max_width = canvas_w - 40
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, config.FONT_FACE, 
                                    config.FONT_SCALE, config.FONT_THICKNESS)[0]
        
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate total height
    line_height = 30
    total_height = len(lines) * line_height + 20
    
    # Position at bottom if not specified
    if y_position is None:
        y_position = canvas_h - total_height - 10
    
    # Draw background rectangle
    cv2.rectangle(canvas, (10, y_position - 10), 
                 (canvas_w - 10, y_position + total_height - 10),
                 config.COLOR_SUBTITLE_BG, -1)
    cv2.rectangle(canvas, (10, y_position - 10), 
                 (canvas_w - 10, y_position + total_height - 10),
                 config.COLOR_TEXT, 2)
    
    # Draw text lines
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, config.FONT_FACE, 
                                    config.FONT_SCALE, config.FONT_THICKNESS)[0]
        text_x = (canvas_w - text_size[0]) // 2
        text_y = y_position + (i + 1) * line_height
        
        cv2.putText(canvas, line, (text_x, text_y), config.FONT_FACE,
                   config.FONT_SCALE, config.COLOR_TEXT, config.FONT_THICKNESS)
    
    return canvas


# ============================================================================
# MAIN RENDERING FUNCTION
# ============================================================================

def render_portrait(animation_state: AnimationState, canvas_size: tuple = None) -> np.ndarray:
    """
    Render the living portrait based on current animation state
    
    Args:
        animation_state: Current AnimationState
        canvas_size: (width, height) or None for default
    
    Returns:
        Rendered canvas image (BGR)
    """
    # Create canvas
    if canvas_size is None:
        canvas_size = (config.PORTRAIT_WIDTH, config.PORTRAIT_HEIGHT)
    
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # Dark gray background
    
    # Get sprite manager
    sprite_mgr = get_sprite_manager()
    
    # Determine which sprite to show
    if animation_state.speaking:
        # Talking animation - alternate between open and closed mouth
        sprite_name = 'talking_open' if animation_state.mouth_open else 'talking_closed'
    else:
        # Use mood sprite
        sprite_name = animation_state.mood
    
    # Render sprite or fallback
    if sprite_mgr.has_sprites():
        sprite = sprite_mgr.get_sprite(sprite_name)
        
        if sprite is not None:
            # Center sprite on canvas
            sprite_h, sprite_w = sprite.shape[:2]
            x = (canvas_size[0] - sprite_w) // 2
            y = (canvas_size[1] - sprite_h) // 2 - 40  # Offset up for subtitle space
            
            canvas = overlay_sprite(canvas, sprite, x, y)
        else:
            # Fallback for this specific sprite
            draw_fallback_sprite(canvas, sprite_name, 
                               canvas_size[0] // 4, canvas_size[1] // 4,
                               canvas_size[0] // 2, canvas_size[1] // 2)
    else:
        # Full fallback rendering
        draw_fallback_sprite(canvas, sprite_name,
                           canvas_size[0] // 4, canvas_size[1] // 4,
                           canvas_size[0] // 2, canvas_size[1] // 2)
    
    # Draw subtitle if present
    if animation_state.subtitle:
        canvas = draw_subtitle(canvas, animation_state.subtitle)
    
    # Draw debug info if enabled
    if config.DEBUG_MODE:
        debug_text = f"Mood: {animation_state.mood} | Speaking: {animation_state.speaking}"
        cv2.putText(canvas, debug_text, (10, 20), config.FONT_FACE,
                   0.5, config.COLOR_DEBUG_INFO, 1)
    
    return canvas


# ============================================================================
# ANIMATION LOOP HELPER
# ============================================================================

def update_and_render(animation_state: AnimationState, now: float) -> np.ndarray:
    """
    Update animation state and render portrait
    Convenience function combining update logic and rendering
    
    Args:
        animation_state: AnimationState to update and render
        now: Current timestamp
    
    Returns:
        Rendered portrait image
    """
    # Update speaking animation
    animation_state.update_speaking(now, config.TALKING_MOUTH_TOGGLE_INTERVAL)
    
    # Update subtitle visibility
    animation_state.update_subtitle(now)
    
    # Render
    return render_portrait(animation_state)
