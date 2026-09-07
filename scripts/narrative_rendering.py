"""Safe, limited HTML rendering for untrusted narrative drafts."""

import re
from html import escape


def markdown_to_html(text):
    """Render limited formatting after escaping all provider-controlled HTML."""
    text = escape(text)
    # Convert bold **text** to <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert italic *text* to <em>text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    lines = text.split("\n")
    html_lines = []
    in_ordered_list = False
    in_unordered_list = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Check if numbered list item (e.g. 1. Velocity Anomaly)
        ol_match = re.match(r'^(\d+)\.\s+(.*)$', line_stripped)
        # Check if bullet list item (e.g. - or * Velocity Anomaly)
        ul_match = re.match(r'^[\*\-\+]\s+(.*)$', line_stripped)
        
        if ol_match:
            if in_unordered_list:
                html_lines.append('</ul>')
                in_unordered_list = False
            if not in_ordered_list:
                html_lines.append('<ol style="margin-top: 8px; margin-bottom: 8px; padding-left: 20px;">')
                in_ordered_list = True
            html_lines.append(f'<li style="margin-bottom: 8px; line-height: 1.6;">{ol_match.group(2)}</li>')
        elif ul_match:
            if in_ordered_list:
                html_lines.append('</ol>')
                in_ordered_list = False
            if not in_unordered_list:
                html_lines.append('<ul style="margin-top: 8px; margin-bottom: 8px; padding-left: 20px; list-style-type: disc;">')
                in_unordered_list = True
            html_lines.append(f'<li style="margin-bottom: 8px; line-height: 1.6;">{ul_match.group(2)}</li>')
        else:
            if in_ordered_list:
                html_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                html_lines.append('</ul>')
                in_unordered_list = False
                
            # Check for headers
            if line_stripped.startswith("###"):
                html_lines.append(f'<h5 style="margin-top: 16px; margin-bottom: 8px; color: #58a6ff; font-weight: 600;">{line_stripped[3:].strip()}</h5>')
            elif line_stripped.startswith("##"):
                html_lines.append(f'<h4 style="margin-top: 20px; margin-bottom: 10px; color: #58a6ff; font-weight: 600;">{line_stripped[2:].strip()}</h4>')
            elif line_stripped.startswith("#"):
                html_lines.append(f'<h3 style="margin-top: 24px; margin-bottom: 12px; color: #58a6ff; font-weight: 600;">{line_stripped[1:].strip()}</h3>')
            else:
                html_lines.append(f'<p style="margin-bottom: 12px; line-height: 1.6;">{line_stripped}</p>')
                
    if in_ordered_list:
        html_lines.append('</ol>')
    if in_unordered_list:
        html_lines.append('</ul>')
        
    return "\n".join(html_lines)

