import re

file_path = '/Users/jasmi/Desktop/AV-Deepfake1M/Try/draft.typ'

with open(file_path, 'r') as f:
    content = f.read()

# Clean up multiple /* NOT IN BIB */ and nested parentheses for the specific authors
def cleanup(text):
    # Match any sequence of /* NOT IN BIB */ and parentheses around the target authors
    for author in ['Lin et al\., 2017', 'Tran et al\., 2018']:
        # This regex matches any combination of the comment and parentheses around the author string
        pattern = r'(?:\/\* NOT IN BIB \*\/|\s|\(|\))*' + author + r'(?:\/\* NOT IN BIB \*\/|\s|\(|\))*'
        # We want to replace it with: /* NOT IN BIB */ (Author, Year)
        # But wait, if it's in a sentence like "Lin et al. (2017)", we should be careful.
        # However, the prompt specifically said replace "(Lin et al., 2017)".
        
        # Let's find all (potentially nested) parenthetical occurrences
        def fix_occurrence(match):
            return ' /* NOT IN BIB */ (' + author.replace('\\', '') + ') '
        
        # We'll target the ones that definitely have the comma and year inside parentheses
        text = re.sub(r'\(.*?' + author + r'.*?\)', fix_occurrence, text)
        
        # Also clean up the ones that might have been broken out
        text = re.sub(r'\/\* NOT IN BIB \*\/.*?' + author + r'.*?([0-9]{4})', r'/* NOT IN BIB */ (\1)'.replace('\\1', author.replace('\\', '')), text)

    # Final pass to fix any double spaces or broken formatting
    text = re.sub(r'(\/\* NOT IN BIB \*\/ \([^)]+\))\s+\1', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text

# Actually, a simpler way to clean up the specific mess I made:
new_content = content
new_content = re.sub(r'(?:\/\* NOT IN BIB \*\/[\s()]*)+Lin et al\., 2017[\s()]*', ' /* NOT IN BIB */ (Lin et al., 2017) ', new_content)
new_content = re.sub(r'(?:\/\* NOT IN BIB \*\/[\s()]*)+Tran et al\., 2018[\s()]*', ' /* NOT IN BIB */ (Tran et al., 2018) ', new_content)

# Remove extra spaces before punctuation
new_content = re.sub(r'\s+([.,])', r'\1', new_content)

with open(file_path, 'w') as f:
    f.write(new_content)

print("Cleanup complete.")
