"""
Configurable Vision Prompts for Moondream
Allows customizing the level of detail based on model capability
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DetailLevel(Enum):
    """Vision analysis detail levels based on model capability"""
    BASIC = "basic"           # Raspberry Pi / lightweight models
    STANDARD = "standard"     # Default for desktop
    DETAILED = "detailed"     # Full feature extraction
    FULL = "full"             # Maximum detail (may be slow)


@dataclass
class VisionPromptConfig:
    """Configuration for vision analysis prompts"""
    
    # Detail level determines which features to analyze
    detail_level: DetailLevel = DetailLevel.STANDARD
    
    # Individual feature toggles (can override detail_level defaults)
    include_facial_expressions: bool = True
    include_body_language: bool = True
    include_room_details: bool = False
    include_objects: bool = True
    include_clothing: bool = True
    include_age_gender: bool = False  # Privacy-sensitive
    include_emotion: bool = True
    include_gaze_direction: bool = False
    include_hand_gestures: bool = False
    include_lighting: bool = False
    
    # Custom prompts to add
    custom_prompts: List[str] = field(default_factory=list)
    
    @classmethod
    def for_raspberry_pi(cls) -> 'VisionPromptConfig':
        """Lightweight config optimized for Raspberry Pi"""
        return cls(
            detail_level=DetailLevel.BASIC,
            include_facial_expressions=True,
            include_body_language=False,
            include_room_details=False,
            include_objects=False,
            include_clothing=False,
            include_age_gender=False,
            include_emotion=True,
            include_gaze_direction=False,
            include_hand_gestures=False,
            include_lighting=False
        )
    
    @classmethod
    def for_desktop(cls) -> 'VisionPromptConfig':
        """Standard config for desktop with good GPU"""
        return cls(
            detail_level=DetailLevel.STANDARD,
            include_facial_expressions=True,
            include_body_language=True,
            include_room_details=False,
            include_objects=True,
            include_clothing=True,
            include_age_gender=False,
            include_emotion=True,
            include_gaze_direction=False,
            include_hand_gestures=False,
            include_lighting=False
        )
    
    @classmethod
    def full_detail(cls) -> 'VisionPromptConfig':
        """Maximum detail for powerful systems"""
        return cls(
            detail_level=DetailLevel.FULL,
            include_facial_expressions=True,
            include_body_language=True,
            include_room_details=True,
            include_objects=True,
            include_clothing=True,
            include_age_gender=True,
            include_emotion=True,
            include_gaze_direction=True,
            include_hand_gestures=True,
            include_lighting=True
        )
    
    def get_feature_list(self) -> List[str]:
        """Get list of enabled features"""
        features = []
        if self.include_facial_expressions:
            features.append("facial_expressions")
        if self.include_body_language:
            features.append("body_language")
        if self.include_room_details:
            features.append("room_details")
        if self.include_objects:
            features.append("objects")
        if self.include_clothing:
            features.append("clothing")
        if self.include_age_gender:
            features.append("age_gender")
        if self.include_emotion:
            features.append("emotion")
        if self.include_gaze_direction:
            features.append("gaze_direction")
        if self.include_hand_gestures:
            features.append("hand_gestures")
        if self.include_lighting:
            features.append("lighting")
        return features


# =============================================================================
# VISION PROMPT TEMPLATES
# =============================================================================

VISION_PROMPTS = {
    # Facial expression analysis
    "facial_expressions": {
        "prompt": "Describe the person's facial expression. Are they smiling, frowning, neutral, surprised, confused, or showing other expressions?",
        "short_prompt": "facial expression",
    },
    
    # Body language analysis
    "body_language": {
        "prompt": "Describe the person's body language and posture. Are they leaning forward (engaged), leaning back (relaxed), crossed arms (defensive), open posture, fidgeting, or still?",
        "short_prompt": "body language and posture",
    },
    
    # Room/environment details
    "room_details": {
        "prompt": "Describe the room or environment behind the person. What kind of space is it? (office, living room, outdoors, etc.) What is the general atmosphere?",
        "short_prompt": "environment and background",
    },
    
    # Objects in view
    "objects": {
        "prompt": "List any notable objects visible in the image that might be relevant for conversation (books, decorations, devices, plants, artwork, etc.)",
        "short_prompt": "visible objects",
    },
    
    # Clothing description
    "clothing": {
        "prompt": "Briefly describe what the person is wearing (casual, formal, colors, any notable items like glasses, hat, jewelry)",
        "short_prompt": "clothing and accessories",
    },
    
    # Age and gender (privacy note)
    "age_gender": {
        "prompt": "Estimate the person's approximate age range and apparent gender",
        "short_prompt": "approximate age and gender",
    },
    
    # Emotion/mood
    "emotion": {
        "prompt": "What emotion or mood does the person appear to be in? (happy, sad, curious, bored, excited, calm, stressed, etc.)",
        "short_prompt": "apparent mood or emotion",
    },
    
    # Gaze direction
    "gaze_direction": {
        "prompt": "Where is the person looking? Are they looking at the camera, looking away, looking down at something, or distracted?",
        "short_prompt": "gaze direction",
    },
    
    # Hand gestures
    "hand_gestures": {
        "prompt": "Describe any hand gestures the person is making. Are their hands visible? Are they gesturing, pointing, holding something?",
        "short_prompt": "hand position and gestures",
    },
    
    # Lighting conditions
    "lighting": {
        "prompt": "Describe the lighting conditions. Is it bright, dim, natural light, artificial light, any interesting lighting effects?",
        "short_prompt": "lighting conditions",
    },
}


def build_vision_prompt(config: VisionPromptConfig) -> str:
    """
    Build a vision analysis prompt based on configuration
    
    Args:
        config: VisionPromptConfig specifying what to analyze
    
    Returns:
        Complete prompt string for Moondream
    """
    # Base instruction
    parts = [
        "Analyze this image and describe what you see. Focus on the following aspects:"
    ]
    
    # Add prompts for enabled features
    features = config.get_feature_list()
    for i, feature in enumerate(features, 1):
        if feature in VISION_PROMPTS:
            prompt_info = VISION_PROMPTS[feature]
            # Use short prompts for basic level, full prompts otherwise
            if config.detail_level == DetailLevel.BASIC:
                parts.append(f"{i}. {prompt_info['short_prompt']}")
            else:
                parts.append(f"{i}. {prompt_info['prompt']}")
    
    # Add custom prompts
    for custom in config.custom_prompts:
        parts.append(f"- {custom}")
    
    # Add format instruction based on detail level
    if config.detail_level == DetailLevel.BASIC:
        parts.append("\nProvide a brief, concise description (2-3 sentences max).")
    elif config.detail_level == DetailLevel.STANDARD:
        parts.append("\nProvide a clear description covering each point briefly.")
    elif config.detail_level == DetailLevel.DETAILED:
        parts.append("\nProvide detailed observations for each point.")
    else:  # FULL
        parts.append("\nProvide comprehensive, detailed observations for each point.")
    
    return "\n".join(parts)


def build_quick_vision_prompt() -> str:
    """Build a minimal prompt for quick scene description"""
    return "Briefly describe the person in the image and their apparent mood or state in 1-2 sentences."


def build_conversation_context_prompt(config: VisionPromptConfig) -> str:
    """
    Build a prompt that extracts context useful for natural conversation
    
    Args:
        config: VisionPromptConfig
    
    Returns:
        Prompt focused on conversational context
    """
    parts = [
        "Look at this image and extract information that would help carry on a natural conversation with this person."
    ]
    
    if config.include_facial_expressions or config.include_emotion:
        parts.append("- Their apparent mood or emotion")
    
    if config.include_body_language:
        parts.append("- Whether they seem engaged, distracted, or relaxed")
    
    if config.include_objects or config.include_room_details:
        parts.append("- Any interesting items or aspects of their environment worth mentioning")
    
    if config.include_clothing:
        parts.append("- Notable aspects of their appearance")
    
    parts.append("\nSummarize in 2-3 sentences, focusing on details useful for conversation.")
    
    return "\n".join(parts)


# =============================================================================
# GLOBAL CONFIG (can be modified at runtime)
# =============================================================================

# Default configuration - can be changed via config file or API
_current_config: Optional[VisionPromptConfig] = None


def get_vision_config() -> VisionPromptConfig:
    """Get the current vision prompt configuration"""
    global _current_config
    if _current_config is None:
        _current_config = VisionPromptConfig.for_desktop()
    return _current_config


def set_vision_config(config: VisionPromptConfig):
    """Set the vision prompt configuration"""
    global _current_config
    _current_config = config
    print(f"[Vision] Config updated: detail_level={config.detail_level.value}, features={config.get_feature_list()}")


def set_detail_level(level: str):
    """
    Set detail level by string name
    
    Args:
        level: "basic", "standard", "detailed", or "full"
    """
    level_map = {
        "basic": VisionPromptConfig.for_raspberry_pi,
        "standard": VisionPromptConfig.for_desktop,
        "detailed": VisionPromptConfig.for_desktop,  # Same as standard but can be customized
        "full": VisionPromptConfig.full_detail,
    }
    
    if level in level_map:
        set_vision_config(level_map[level]())
        print(f"[Vision] Detail level set to: {level}")
    else:
        print(f"[Vision] Unknown detail level: {level}. Use: basic, standard, detailed, full")
