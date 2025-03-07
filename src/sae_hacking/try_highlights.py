#!/usr/bin/env python3
import random
import re


def highlight_tokens_with_intensity(text):
    """
    Highlights every token in text with green background of varying intensity.

    Args:
        text (str): The text to process

    Returns:
        str: HTML text with all tokens highlighted
    """
    # Tokenize the text (split by whitespace and punctuation)
    tokens = re.findall(r"\b\w+\b|\s+|[^\w\s]", text)

    html_parts = []

    for token in tokens:
        # Skip whitespace for highlighting but preserve it
        if token.isspace():
            html_parts.append(token)
            continue

        # Assign random intensity for each token (1-10)
        # You can replace this with your own intensity logic
        intensity = random.randint(1, 10)

        # Calculate green shade based on intensity
        red = max(235 - (intensity * 20), 100)
        green = 255
        blue = max(235 - (intensity * 20), 100)

        color = f"#{red:02x}{green:02x}{blue:02x}"

        # Create HTML for the highlighted token
        highlighted = f'<span style="background-color: {color};">{token}</span>'
        html_parts.append(highlighted)

    return "".join(html_parts)


# Example usage
def main():
    # Sample text
    text = (
        "Python is a versatile programming language that is powerful and easy to learn."
    )

    # Generate HTML with highlighted tokens
    html_output = highlight_tokens_with_intensity(text)

    # Create a complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Highlighted Text Example</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                margin: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Green Intensity Highlighting Example</h1>
        <p>{html_output}</p>
        <hr>
        <p>Each token is highlighted with a different intensity of green (1-10 scale).</p>
    </body>
    </html>
    """

    # Save to a file
    with open("highlighted_text.html", "w") as f:
        f.write(full_html)

    print("HTML file created with highlighted tokens!")
    print("\nPreview of highlighted text:")
    print(html_output)


if __name__ == "__main__":
    main()
