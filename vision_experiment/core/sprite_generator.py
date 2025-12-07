"""
SVG-based Sprite Generator - Creates simple vector fallback sprites
Generates PNG sprites from Python when sprite files don't exist
"""
import cv2
import numpy as np
import os
from typing import Tuple

from core import config


def create_simple_face_sprite(
    size: Tuple[int, int] = (400, 400),
    face_color: Tuple[int, int, int] = (255, 200, 150),
    eye_state: str = "open",
    mouth_state: str = "neutral",
    background: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Create a simple face sprite using OpenCV drawing functions
    
    Args:
        size: (width, height) of sprite
        face_color: BGR color for face
        eye_state: "open", "closed", "happy"
        mouth_state: "neutral", "happy", "sad", "open", "closed", "talking"
        background: BGR background color (use 0,0,0 for transparent after conversion)
    
    Returns:
        BGRA numpy array (with alpha channel)
    """
    width, height = size
    
    # Create canvas with alpha channel
    sprite = np.zeros((height, width, 4), dtype=np.uint8)
    sprite[:, :, 3] = 0  # Fully transparent background
    
    # Face circle
    center_x, center_y = width // 2, height // 2
    face_radius = min(width, height) // 3
    
    # Draw face (filled circle with alpha)
    cv2.circle(sprite, (center_x, center_y), face_radius, 
               (*face_color, 255), -1)
    
    # Face outline
    cv2.circle(sprite, (center_x, center_y), face_radius,
               (100, 100, 100, 255), 3)
    
    # Eyes
    eye_y = center_y - face_radius // 3
    eye_offset_x = face_radius // 3
    left_eye_x = center_x - eye_offset_x
    right_eye_x = center_x + eye_offset_x
    eye_radius = face_radius // 8
    
    if eye_state == "closed":
        # Draw closed eyes as lines
        line_len = eye_radius * 2
        cv2.line(sprite, (left_eye_x - line_len//2, eye_y),
                (left_eye_x + line_len//2, eye_y), (50, 50, 50, 255), 3)
        cv2.line(sprite, (right_eye_x - line_len//2, eye_y),
                (right_eye_x + line_len//2, eye_y), (50, 50, 50, 255), 3)
    elif eye_state == "happy":
        # Draw happy eyes as arcs (^_^)
        for eye_x in [left_eye_x, right_eye_x]:
            cv2.ellipse(sprite, (eye_x, eye_y), (eye_radius, eye_radius//2),
                       0, 0, 180, (50, 50, 50, 255), 3)
    else:  # "open"
        # Draw open eyes as circles
        cv2.circle(sprite, (left_eye_x, eye_y), eye_radius,
                  (50, 50, 50, 255), -1)
        cv2.circle(sprite, (right_eye_x, eye_y), eye_radius,
                  (50, 50, 50, 255), -1)
        # Add pupils
        cv2.circle(sprite, (left_eye_x, eye_y), eye_radius//2,
                  (255, 255, 255, 255), -1)
        cv2.circle(sprite, (right_eye_x, eye_y), eye_radius//2,
                  (255, 255, 255, 255), -1)
    
    # Mouth
    mouth_y = center_y + face_radius // 3
    mouth_width = face_radius
    
    if mouth_state == "happy":
        # Smiling arc
        cv2.ellipse(sprite, (center_x, mouth_y), (mouth_width//2, mouth_width//3),
                   0, 0, 180, (50, 50, 50, 255), 3)
    elif mouth_state == "sad":
        # Frowning arc
        cv2.ellipse(sprite, (center_x, mouth_y + mouth_width//3), 
                   (mouth_width//2, mouth_width//3),
                   0, 180, 360, (50, 50, 50, 255), 3)
    elif mouth_state == "open":
        # Open mouth (oval)
        cv2.ellipse(sprite, (center_x, mouth_y), (mouth_width//4, mouth_width//3),
                   0, 0, 360, (50, 50, 50, 255), -1)
    elif mouth_state == "closed":
        # Small closed mouth
        cv2.ellipse(sprite, (center_x, mouth_y), (mouth_width//5, mouth_width//6),
                   0, 0, 360, (50, 50, 50, 255), -1)
    elif mouth_state == "talking":
        # Rounded rectangle for talking
        mouth_pts = np.array([
            [center_x - mouth_width//4, mouth_y - mouth_width//6],
            [center_x + mouth_width//4, mouth_y - mouth_width//6],
            [center_x + mouth_width//4, mouth_y + mouth_width//6],
            [center_x - mouth_width//4, mouth_y + mouth_width//6]
        ], np.int32)
        cv2.fillPoly(sprite, [mouth_pts], (50, 50, 50, 255))
    else:  # "neutral"
        # Straight line
        cv2.line(sprite, (center_x - mouth_width//4, mouth_y),
                (center_x + mouth_width//4, mouth_y), (50, 50, 50, 255), 3)
    
    return sprite


def generate_all_sprites(output_dir: str = None) -> dict:
    """
    Generate all required sprite images and save to files
    
    Args:
        output_dir: Directory to save sprites (default: config.SPRITES_DIR)
    
    Returns:
        Dictionary mapping sprite names to file paths
    """
    if output_dir is None:
        output_dir = config.SPRITES_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_sprites = {}
    
    # Define sprite configurations
    sprite_configs = {
        'idle': {
            'eye_state': 'open',
            'mouth_state': 'neutral',
            'face_color': (200, 180, 160)  # Neutral beige
        },
        'happy': {
            'eye_state': 'happy',
            'mouth_state': 'happy',
            'face_color': (180, 220, 255)  # Light happy blue
        },
        'curious': {
            'eye_state': 'open',
            'mouth_state': 'neutral',
            'face_color': (200, 255, 200)  # Light curious green
        },
        'thoughtful': {
            'eye_state': 'open',
            'mouth_state': 'neutral',
            'face_color': (255, 220, 180)  # Warm thinking orange
        },
        'concerned': {
            'eye_state': 'open',
            'mouth_state': 'sad',
            'face_color': (255, 200, 180)  # Light concerned orange
        },
        'sad': {
            'eye_state': 'closed',
            'mouth_state': 'sad',
            'face_color': (220, 200, 255)  # Soft sad purple
        },
        'talking_open': {
            'eye_state': 'open',
            'mouth_state': 'open',
            'face_color': (200, 180, 160)
        },
        'talking_closed': {
            'eye_state': 'open',
            'mouth_state': 'closed',
            'face_color': (200, 180, 160)
        }
    }
    
    # Generate each sprite
    for sprite_name, sprite_config in sprite_configs.items():
        sprite = create_simple_face_sprite(**sprite_config)
        
        filename = config.SPRITE_FILES.get(sprite_name, f"{sprite_name}.png")
        filepath = os.path.join(output_dir, filename)
        
        # Save sprite
        cv2.imwrite(filepath, sprite)
        generated_sprites[sprite_name] = filepath
        
        print(f"✓ Generated sprite: {sprite_name} -> {filepath}")
    
    return generated_sprites


def ensure_sprites_exist() -> bool:
    """
    Check if sprites exist, generate if missing
    
    Returns:
        True if sprites are available (existing or generated), False on error
    """
    os.makedirs(config.SPRITES_DIR, exist_ok=True)
    
    # Check if any sprites are missing
    missing_sprites = []
    for sprite_name, filename in config.SPRITE_FILES.items():
        filepath = os.path.join(config.SPRITES_DIR, filename)
        if not os.path.exists(filepath):
            missing_sprites.append(sprite_name)
    
    # Generate missing sprites
    if missing_sprites:
        print(f"⚠ Missing {len(missing_sprites)} sprites, generating...")
        try:
            generate_all_sprites()
            print(f"✓ Successfully generated all sprites")
            return True
        except Exception as e:
            print(f"✗ Error generating sprites: {e}")
            return False
    
    return True


if __name__ == "__main__":
    # Test sprite generation
    print("Generating test sprites...")
    generated = generate_all_sprites()
    print(f"\nGenerated {len(generated)} sprites:")
    for name, path in generated.items():
        print(f"  {name}: {path}")
