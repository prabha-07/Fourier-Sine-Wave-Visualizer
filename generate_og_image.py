#!/usr/bin/env python3
"""
Generate Open Graph image for Fourier's Utopia website
Creates a 1200x630px image optimized for social media previews
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Create image with dimensions for OG (LinkedIn, Facebook, Twitter)
width, height = 1200, 630
image = Image.new('RGB', (width, height), color='#020617')  # Dark background matching site
draw = ImageDraw.Draw(image)

# Try to use system fonts, fallback to default if not available
try:
    # Try to find a nice font
    title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 80)
    subtitle_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
    desc_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
except:
    try:
        title_font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 80)
        subtitle_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 36)
        desc_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 28)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()

# Draw gradient background effect (simplified)
for y in range(height):
    # Create a subtle gradient from dark blue to darker
    ratio = y / height
    r = int(2 + (15 - 2) * ratio)
    g = int(6 + (23 - 6) * ratio)
    b = int(23 + (42 - 23) * ratio)
    draw.line([(0, y), (width, y)], fill=(r, g, b))

# Draw sine wave pattern in background (subtle)
wave_color = (56, 189, 248, 60)  # Sky blue with transparency
for i in range(3):
    y_offset = 200 + i * 150
    points = []
    for x in range(0, width, 5):
        y = y_offset + 30 * (i + 1) * (1 if (x // 50) % 2 == 0 else -1)
        points.append((x, y))
    if len(points) > 1:
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=(56, 189, 248), width=2)

# Draw title with gradient effect (simulated)
title = "Fourier's Utopia"
subtitle = "Interactive DSP Toolkit"
description = "Upload audio • Decompose into sine waves • Watch the spectrum come alive"

# Calculate text positions (centered)
title_bbox = draw.textbbox((0, 0), title, font=title_font)
title_width = title_bbox[2] - title_bbox[0]
title_x = (width - title_width) // 2

subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
subtitle_x = (width - subtitle_width) // 2

desc_bbox = draw.textbbox((0, 0), description, font=desc_font)
desc_width = desc_bbox[2] - desc_bbox[0]
desc_x = (width - desc_width) // 2

# Draw text with gradient colors
y_start = 180

# Title - gradient from blue to purple
title_text = title
for i, char in enumerate(title_text):
    if char == ' ':
        continue
    # Gradient effect: blue to purple
    ratio = i / len(title_text) if len(title_text) > 0 else 0
    r = int(56 + (168 - 56) * ratio)
    g = int(189 + (85 - 189) * ratio)
    b = int(248 + (247 - 248) * ratio)
    
    char_bbox = draw.textbbox((0, 0), char, font=title_font)
    char_width = char_bbox[2] - char_bbox[0]
    char_x = title_x + sum([draw.textbbox((0, 0), title_text[:j], font=title_font)[2] - draw.textbbox((0, 0), title_text[:j], font=title_font)[0] for j in range(i)])
    
    draw.text((char_x, y_start), char, fill=(r, g, b), font=title_font)

# Simpler approach - draw title in gradient color
draw.text((title_x, y_start), title, fill=(56, 189, 248), font=title_font)  # Sky blue

# Subtitle
draw.text((subtitle_x, y_start + 100), subtitle, fill=(148, 163, 184), font=subtitle_font)

# Description
draw.text((desc_x, y_start + 160), description, fill=(148, 163, 184), font=desc_font)

# Add decorative elements - small sine waves at bottom
wave_y = height - 100
for x in range(0, width, 20):
    y1 = wave_y + int(15 * (1 if (x // 40) % 2 == 0 else -1))
    y2 = wave_y + int(15 * (1 if ((x + 20) // 40) % 2 == 0 else -1))
    draw.line([(x, y1), (x + 20, y2)], fill=(56, 189, 248, 100), width=2)

# Add badge text at bottom
badge_text = "Python Backend • Web Audio API • Flask Powered"
badge_bbox = draw.textbbox((0, 0), badge_text, font=desc_font)
badge_width = badge_bbox[2] - badge_bbox[0]
badge_x = (width - badge_width) // 2
draw.text((badge_x, height - 60), badge_text, fill=(100, 116, 139), font=desc_font)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), 'static', 'og-image.png')
image.save(output_path, 'PNG', optimize=True)
print(f"OG image created successfully at: {output_path}")
print(f"Image dimensions: {width}x{height}px")

