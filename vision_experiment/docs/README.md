# Sprite Image Guidelines

## Required Sprites

Create 6 PNG images and place them in this directory:

1. **idle.png** - Neutral/default expression
2. **happy.png** - Happy/joyful expression
3. **curious.png** - Curious/questioning expression
4. **thoughtful.png** - Thoughtful/contemplative expression
5. **talking_open.png** - Mouth open (for talking animation)
6. **talking_closed.png** - Mouth closed (for talking animation)

## Image Specifications

- **Format**: PNG with transparency (RGBA)
- **Size**: 400x400 to 600x600 pixels recommended
- **Content**: Portrait/face of your character
- **Background**: Transparent (alpha channel)
- **Style**: Consistent across all sprites

## Quick Start Options

### Option 1: AI-Generated
Use AI image generators like:
- DALL-E, Midjourney, Stable Diffusion
- Prompt: "portrait of a magical painting character, [mood], transparent background, digital art"

### Option 2: Hand-Drawn
- Draw or commission artwork
- Export as PNG with transparency
- Tools: Procreate, Photoshop, GIMP, Krita

### Option 3: Use Emojis (Quick Test)
For testing, you can use large emojis:
- Download emoji PNGs from emoji databases
- idle.png → 😐
- happy.png → 😊
- curious.png → 🤔
- thoughtful.png → 🧐
- talking_open.png → 😮
- talking_closed.png → 🙂

### Option 4: Placeholder Mode
If no sprites are present, the system will automatically use:
- Colored rectangles as placeholders
- Each mood gets a different color
- Text label shows current mood
- **This works out of the box - no sprites required to test!**

## Testing Your Sprites

1. Add PNG files to this directory
2. Run: `python3 ../portrait.py`
3. System will report which sprites loaded successfully
4. Portrait should display your artwork!

## Tips

- Keep facial features clear and expressive
- Ensure good contrast for visibility
- Test different moods to verify they're distinguishable
- talking_open and talking_closed should have minimal differences (just mouth)
- Consider the canvas size (640x480) when designing

## Example Directory Structure

```
sprites/
├── README.md (this file)
├── idle.png (400x400, neutral face)
├── happy.png (400x400, smiling)
├── curious.png (400x400, raised eyebrow)
├── thoughtful.png (400x400, contemplative)
├── talking_open.png (400x400, mouth open)
└── talking_closed.png (400x400, mouth closed)
```

---

**Remember**: The system works fine without sprites! They're an enhancement, not a requirement.
