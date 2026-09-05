"""
Convert presentation.pptx to reveal.js HTML.
Auto-extracts slide content, tables, and images.
"""
import os, re
from pptx import Presentation

PPTX_PATH = "/Users/jasmi/Desktop/AV-Deepfake1M/Try/presentation.pptx"
OUT_HTML  = "/Users/jasmi/Desktop/AV-Deepfake1M/Try/presentation.html"
PROJECT   = "/Users/jasmi/Desktop/AV-Deepfake1M/Try"

# Image mapping: (slide_index, image_ordinal) → relative path
IMAGE_MAP = {
    (4, 0): "figures/architecture.png",
    (6, 0): "figures/two_phase_training.png",
    (7, 0): "comparison_results/training_history.png",
    (7, 1): "figures/per_type_accuracy_bar_chart.png",
    (8, 0): "comparison_results/model_comparison.png",
    (8, 1): "figures/calibration_curves.png",
    (10, 0): "figures/web_analyze_fake.png",
    (10, 1): "figures/web_compare.png",
    (10, 2): "figures/web_history.png",
}

prs = Presentation(PPTX_PATH)
slides_data = []

for si, slide in enumerate(prs.slides):
    s = {'index': si, 'layout': slide.slide_layout.name, 'shapes': []}
    img_count = 0
    for shape in slide.shapes:
        stype = str(shape.shape_type)
        if 'PICTURE' in stype:
            img_path = IMAGE_MAP.get((si, img_count))
            if img_path:
                s['shapes'].append({'type': 'image', 'path': img_path})
            img_count += 1
            continue
        if shape.has_text_frame:
            paras = []
            for p in shape.text_frame.paragraphs:
                runs_data = []
                for r in p.runs:
                    ri = {'text': r.text}
                    if r.font.size: ri['size'] = int(r.font.size / 12700)
                    if r.font.bold is not None: ri['bold'] = bool(r.font.bold)
                    try: ri['color'] = '#' + str(r.font.color.rgb) if r.font.color and r.font.color.type else 'inherit'
                    except: ri['color'] = 'inherit'
                    runs_data.append(ri)
                if runs_data:
                    paras.append({'runs': runs_data, 'level': p.level or 0})
            if paras and any(any(r['text'].strip() for r in p['runs']) for p in paras):
                s['shapes'].append({'type': 'text', 'name': shape.name, 'paragraphs': paras})
        if shape.has_table:
            t = shape.table
            rows = [[t.cell(ri, ci).text for ci in range(len(t.columns))] for ri in range(len(t.rows))]
            s['shapes'].append({'type': 'table', 'rows': rows})
    slides_data.append(s)


def esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def render_paragraphs(paras):
    html = ''
    for p_data in paras:
        level = p_data.get('level', 0)
        indent = ' style="padding-left:%dpx;"' % (level * 20) if level else ''
        spans = f'<p{indent}>'
        for r in p_data['runs']:
            text = esc(r['text'])
            if not text: continue
            size = r.get('size', 18)
            bold = r.get('bold', False)
            color = r.get('color', 'inherit')
            style = f'font-size:{size}px; color:{color};'
            if bold: style += ' font-weight:bold;'
            spans += f'<span style="{style}">{text}</span>'
        spans += '</p>'
        html += spans
    return html

def render_table(rows):
    html = '<table>'
    for ri, row in enumerate(rows):
        tag = 'th' if ri == 0 else 'td'
        html += '<tr>' + ''.join(f'<{tag}>{esc(c)}</{tag}>' for c in row) + '</tr>'
    html += '</table>'
    return html


slide_blocks = []
for sd in slides_data:
    parts = []
    for sh in sd['shapes']:
        if sh['type'] == 'image':
            parts.append(f'<img src="{sh["path"]}" class="r-stretch" style="max-height:520px;">')
        elif sh['type'] == 'table':
            parts.append(render_table(sh['rows']))
        elif sh['type'] == 'text':
            parts.append(f'<div class="text-block">\n{render_paragraphs(sh["paragraphs"])}\n</div>')
    slide_blocks.append('      <section>\n' + '\n'.join(parts) + '\n      </section>')


html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Deepfake Detection Using Cross-Model Transformer Fusion</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.0/dist/reveal.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.0/dist/theme/white.css">
  <style>
    .reveal {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 28px; }}
    .reveal h1, .reveal h2, .reveal h3 {{ text-transform: none; color: #1A1A2E; }}
    .reveal h1 {{ font-size: 1.6em; }}
    .reveal h2 {{ font-size: 1.3em; }}
    .reveal h3 {{ font-size: 1.1em; color: #0F3460; }}
    .reveal table {{ font-size: 0.42em; margin: 5px auto; border-collapse: collapse; }}
    .reveal th {{ background: #0070C0; color: white; padding: 4px 8px; }}
    .reveal td {{ padding: 3px 8px; border-bottom: 1px solid #ddd; }}
    .reveal img {{ border: 0 !important; box-shadow: none !important; }}
    .reveal .slides section {{ padding: 15px 30px; text-align: left; }}
    .reveal .text-block p {{ margin: 2px 0; line-height: 1.15; }}
    .reveal .r-stretch {{ max-width: 98%; }}
    .slides section:first-child {{ text-align: center; }}
    .slides section:first-child .text-block {{ text-align: center; }}
  </style>
</head>
<body>
  <div class="reveal">
    <div class="slides">
{chr(10).join(slide_blocks)}
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/reveal.js@4.6.0/dist/reveal.js"></script>
  <script>
    Reveal.initialize({{
      hash: true, width: 1200, height: 700, margin: 0.04,
      transition: 'slide', center: false
    }});
  </script>
</body>
</html>'''

with open(OUT_HTML, 'w') as f:
    f.write(html)

print(f"Generated {OUT_HTML} — {len(slide_blocks)} slides ({len(html)} bytes)")