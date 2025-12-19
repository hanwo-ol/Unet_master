import os
import re
import glob

def scan_markdown_files(directory):
    files = glob.glob(os.path.join(directory, "*.md"))
    
    # Regex for math: $...$ (inline) or $$...$$ (block)
    # We look for non-ASCII characters ([\x80-\uffff]) inside these blocks.
    # Note: This is a heuristic. It handles $...$ but complex nested structures might need careful parsing.
    # This regex attempts to find $...$ content.
    
    issues = []
    
    print(f"Scanning {len(files)} files in {directory}...\n")
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.splitlines()
        
        # Simple line-by-line scanning for inline math with non-ascii
        # This covers the most common case: $한글$
        
        for i, line in enumerate(lines):
            # Find all inline math segments: $...$
            # Non-greedy match for content between $
            matches = re.finditer(r'\$([^\$]+)\$', line)
            
            for match in matches:
                math_content = match.group(1)
                # Check for non-ASCII
                if any(ord(c) > 127 for c in math_content):
                    # Exclude \text{...} patterns which are safe
                    # Check if the non-ascii part is NOT wrapped in \text{...}
                    # This is simple check: if \text is present, we assume it might be safe, 
                    # but refined check is hard with simple regex. 
                    # Let's just flag everything for manual review or smarter fix.
                    
                    if "\\text{" not in math_content:
                        issues.append({
                            "file": os.path.basename(file_path),
                            "line": i + 1,
                            "content": match.group(0),
                            "issue": "Non-ASCII chars in math"
                        })

    return issues

if __name__ == "__main__":
    posts_dir = os.path.join("content", "posts")
    found_issues = scan_markdown_files(posts_dir)
    
    if found_issues:
        print(f"Found {len(found_issues)} potential issues:\n")
        for issue in found_issues:
            print(f"[{issue['file']}:{issue['line']}] {issue['content']}")
    else:
        print("No issues found (heuristic scan).")
