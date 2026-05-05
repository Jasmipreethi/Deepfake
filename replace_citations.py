import re

file_path = '/Users/jasmi/Desktop/AV-Deepfake1M/Try/draft.typ'

# Comprehensive mapping based on bibliography and user list
internal_replacements = {
    'Chesney and Citron, 2019': '@Chesney2019',
    'Milmo, 2024': '@Milmo2024',
    'Rossler et al., 2019': '@Rossler2019',
    'Dolhansky et al., 2020': '@Dolhansky2020',
    'Cai et al., 2024': '@Cai2024',
    'Yi et al., 2023': '@yi2023audiodeepfakedetectionsurvey',
    'He, 2021': '@He2021',
    'LeCun, 2015': '@Lecun2015',
    'Goodfellow et al., 2016': '@Goodfellow-et-al-2016',
    'Hornik, 1989': '@Hornik1989',
    'Westerlund, 2019': '@Westerlund2019',
    'Chesney and Cytowic, 2019': '@Chesney2019',
    'Li et al., 2018': '@Li2018',
    'Li et al., 2019': '@Li2019',
    'Zi et al., 2021': '@Zi2021',
    'Zhou et al., 2021': '@Zhou2021',
    'Narayan et al., 2023': '@Narayan2023',
    'Guo et al., 2021': '@Guo2021',
    'Shen et al., 2023': '@shen2023naturalspeech2latentdiffusion',
    'Cai et al., 2025': '@Cai2025',
    'Korshunov and Marcel, 2018': '@Korshunov2018',
    'Zhang, 2025': '@Zhang2025',
    'Stupp, 2019': '@Stupp2019',
    'Vaccari, 2020': '@Vaccari2020',
    'Allyn, 2022': '@Allyn2022',
    'Moore, 2025': '@Moore2025',
    'Bragg, 2025': '@Bragg2025',
    'ITV News, 2024': '@ITVNews2024',
    'Clark, 2025': '@Clark2025',
    'Gera, 2018': '@Gera2018',
    'Heusel et al., 2017': '@Heusel2017',
    'Cai, 2024': '@Cai2024',
}

# Add "NOT IN BIB" ones separately to handle parentheses correctly
not_in_bib = {
    'Lin et al., 2017': '/* NOT IN BIB */ (Lin et al., 2017)',
    'Tran et al., 2018': '/* NOT IN BIB */ (Tran et al., 2018)',
}

with open(file_path, 'r') as f:
    content = f.read()

# First, handle the known cases that might be inside parentheses OR already broken out
# Re-wrap broken ones or replace ones still in parentheses

# A broad strategy:
# 1. Replace all (A; B; C) groups by parsing them
# 2. Then replace any remaining standalone citation strings from the list

def fix_content(text):
    # Fix parenthetical groups first
    def repl_paren(match):
        inner = match.group(1)
        if not re.search(r'[0-9]{4}', inner):
            return match.group(0)
        
        parts = [p.strip() for p in re.split(r'[;]', inner)]
        new_parts = []
        all_automated = True
        for p in parts:
            if p in internal_replacements:
                new_parts.append(internal_replacements[p])
            elif p in not_in_bib:
                new_parts.append(not_in_bib[p])
                all_automated = False
            else:
                new_parts.append(p)
                all_automated = False
        
        if all_automated:
            return ' '.join(new_parts)
        else:
            return '(' + '; '.join(new_parts) + ')'

    text = re.sub(r'\(([^)]+)\)', repl_paren, text)

    # Now handle the broken ones (no parentheses)
    # Sort by length descending to avoid partial matches
    sorted_keys = sorted(internal_replacements.keys(), key=len, reverse=True)
    for key in sorted_keys:
        # Avoid replacing already replaced ones or ones in comments
        # We look for the literal string not preceded by @ or /*
        pattern = r'(?<![@/*])' + re.escape(key)
        text = re.sub(pattern, internal_replacements[key], text)
    
    # Handle not_in_bib standalones
    for key in not_in_bib:
        pattern = r'(?<![/*])' + re.escape(key)
        text = re.sub(pattern, not_in_bib[key], text)

    return text

new_content = fix_content(content)

# Final cleanup: if we have things like "/* NOT IN BIB */ (/* NOT IN BIB */ (Lin et al., 2017))"
# Or double @
# But our regexes try to avoid that.

with open(file_path, 'w') as f:
    f.write(new_content)

print("Replacement complete.")
