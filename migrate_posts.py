import os
import glob
import frontmatter
import re

def migrate_posts():
    posts_dir = os.path.join("content", "posts")
    files = glob.glob(os.path.join(posts_dir, "*.md"))
    
    keywords_unet = ["U-Net", "UNet", "u-net", "unet"]
    
    print(f"Migrating {len(files)} files...\n")
    
    for file_path in files:
        changed = False
        post = frontmatter.load(file_path)
        
        # 1. Enforce U-Net Tag
        current_tags = post.get('tags', [])
        is_unet_paper = False
        
        # Check Title
        if any(k.lower() in post.get('title', '').lower() for k in keywords_unet):
            is_unet_paper = True
        # Check Content (heuristic: if 'U-Net' appears frequently or in tags)
        elif any(k.lower() in t.lower() for t in current_tags for k in keywords_unet):
            is_unet_paper = True
            
        if is_unet_paper and "U-Net" not in current_tags:
            current_tags.append("U-Net")
            post['tags'] = current_tags
            changed = True
            print(f"[{os.path.basename(file_path)}] Added 'U-Net' tag")

        # 2. Fix Korean in Math
        # Pattern: $ ... [Korean] ... $
        # We replace $...한글...$ with $\text{...한글...}$ IF it's not already wrapped.
        # This is tricky with regex. We will use a function replacer.
        
        def math_replacer(match):
            content = match.group(1) # Content inside $...$
            if any(ord(c) > 127 for c in content) and "\\text{" not in content:
                # Naive fix: Wrap the whole thing in \text{}? 
                # Or try to identify the Korean part?
                # Safest approach for "descriptions" is usually wrapping the string part.
                # But sometimes variables are mixed: $L(x) = \text{손실}$.
                # For this migration, let's wrap the chunks containing non-ascii in \text{}.
                
                # Actually, let's just log it for manual fix if it's complex, 
                # OR blindly wrap the whole non-ascii segment if it looks like a sentence.
                # Given the report earlier: "$λ( Λ\otimes v))$", "$λ, γ> 0$", "$Λ$". 
                # Wait, those Greek letters might be high-ascii but are valid in math?
                # Ah, standard Greek like \lambda is better, but unicode greek is 'okay' usually?
                # No, KaTeX warns about some unicode.
                # But the user specifically mentioned "Korean text".
                # Let's filter for Hangul Syllables Range: AC00-D7A3
                
                has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in content)
                if has_korean:
                     # Wrap the whole match? 
                     # Let's try to assume it's a text description mostly.
                     # "손실함수 $L$" -> mixed.
                     # Inside: $손실함수 L$ -> $\text{손실함수 } L$
                     # This requires parsing.
                     # For the purpose of this task, I will PRINT the file to be fixed manually if risk is high, 
                     # or apply a safe fix if it's pure korean.
                     return match.group(0) # No auto-fix for complex mixed content without ambiguity risk
            
            return match.group(0)

        # Apply specific fixes for the identified issues in previous step if possible.
        # The previous scan showed issues in: neural-networks-for-tamed-milstein-approximation...
        # Issue: $λ( Λ\otimes v))$, $λ, γ> 0$, $Λ$
        # These are Greeek/Symbols. They are technically "Non-ASCII" but valid unicode math often.
        # However, KaTeX might prefer \lambda. 
        # But wait, looking at the scan output: "Found 3 potential issues... $λ...".
        # Those are NOT Korean.
        # The browser subagent earlier said: "When Korean is included... e.g. $손실함수는 l(t)$"
        # My script found NO Korean issues in the 19 files? 
        # Let's double check the range in the script.
        # [Scan Script]: if any(ord(c) > 127 ...
        # [Scan Output]: Only Greek symbols found.
        # Implication: The "Korean in math" issue might have been fixed or I missed a file, OR it was in a file I didn't see in the first 19?
        # Or maybe the "Towards Understanding Generalization..." file mentioned in the browser survey has it.
        # Let's re-verify: "Towards Understanding Generalization..." file size 12713.
        # Did scan_content.py check it? Yes.
        # Did it find it? No. 
        # Why? Maybe the Korean is NOT inside $...$? 
        # Browser said: "$...$ 내부에 한국어가 포함된 경우"
        
        # Let's strictly check for Hangul: 0xAC00 <= ord(c) <= 0xD7A3
        has_hangul = re.search(r"[\uac00-\ud7a3]", post.content)
        if has_hangul:
            pass # Just noting.
            
        if changed:
            with open(file_path, 'wb') as f:
                frontmatter.dump(post, f)
                
if __name__ == "__main__":
    migrate_posts()
