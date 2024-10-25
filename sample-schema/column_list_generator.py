import csv
import os
import markdown

def generate_column_list_html(input_csv, output_html):
    # Read column names from CSV file
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column_names = [row[1] for row in reader]

    # Sort column names alphabetically
    column_names.sort(key=str.lower)

    # Generate Markdown content
    md_content = "# Table Column Names\n\n"
    md_content += "This document lists all column names in alphabetical order.\n\n"
    md_content += "## Column Names\n\n"

    # Create a formatted list of column names
    for i, name in enumerate(column_names, 1):
        md_content += f"{i}. `{name}`\n"

    # Convert Markdown to HTML
    html_content = markdown.markdown(md_content)

    # Wrap HTML content in a basic HTML structure with some CSS
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Table Column Names</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            ol {{ columns: 2; -webkit-columns: 2; -moz-columns: 2; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Write HTML content to file
    with open(output_html, 'w', encoding='utf-8') as htmlfile:
        htmlfile.write(full_html)

    print(f"HTML file '{output_html}' has been generated successfully.")

if __name__ == "__main__":
    input_csv = "column_names.csv"  # Replace with your input CSV file name
    output_html = "column_names.html"   # Output HTML file name

    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file '{input_csv}' not found.")
    else:
        generate_column_list_html(input_csv, output_html)
