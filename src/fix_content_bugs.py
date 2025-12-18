
import os
import glob
import frontmatter
import re

CONTENT_DIR = "content/posts"

def fix_post(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            post = frontmatter.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return

    content = post.content
    changed = False

    # 1. Fix Duplicate PDF Download Headers
    # Regex to find consecutive PDF Download headers (ignoring whitespace/newlines)
    # Using specific string "## PDF Download"
    if content.count("## PDF Download") > 1:
        print(f"Found duplicate PDF headers in {filepath}")
        # Replace 2 or more occurrences with a single one
        new_content = re.sub(r"(## PDF Download\s*){2,}", "## PDF Download\n", content)
        if new_content != content:
            post.content = new_content
            content = new_content
            changed = True
            print(" -> Fixed PDF headers")

    # 2. Fix Tags if they are Auto-Generated
    current_tags = post.get('tags', [])
    if "Auto-Generated" in current_tags or "Draft" in current_tags:
        print(f"Fixing tags for {filepath}")
        # Find "## 4. 태그 제안 (Tags Suggestion)" using flexible regex
        # Look for "## 4." up to "Suggestion)"
        tag_section_match = re.search(r"##\s*4\.\s*태그 제안.*Tags Suggestion.*\s+([\s\S]+)$", content, re.IGNORECASE)
        
        if tag_section_match:
            tags_block = tag_section_match.group(1)
            # Find numbered items: "1. Tag Name"
            extracted_tags = re.findall(r"\d+\.\s*([^\r\n]+)", tags_block)
            if extracted_tags:
                clean_tags = [t.strip() for t in extracted_tags]
                # Filter out empty strings
                clean_tags = [t for t in clean_tags if t]
                
                if clean_tags:
                    post['tags'] = clean_tags
                    changed = True
                    print(f" -> Extracted tags: {clean_tags}")
        else:
            print(" -> Tag section not found via regex")

    if changed:
        with open(filepath, 'wb') as f:
            frontmatter.dump(post, f)
        print(f"Saved changes to {filepath}")
    else:
        print(f"No changes for {filepath}")

def main():
    files = glob.glob(os.path.join(CONTENT_DIR, "*.md"))
    print(f"Found {len(files)} markdown files.")
    for f in files:
        fix_post(f)

if __name__ == "__main__":
    main()
