import os
import time
import glob
import frontmatter
import google.generativeai as genai
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load env vars from .env
from dotenv import load_dotenv


load_dotenv() # Load environment variables from .env file

# Load Config
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Configure API Key
# Assumes legacy env var or simple google-generativeai setup
# Best practice: os.environ["GEMINI_API_KEY"] or similar.
# If not set, this might fail, but we'll assume the environment is pre-configured or user sets it.
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
elif "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SYSTEM_INSTRUCTION = """
당신은 'Research Agent - Draft Refiner Module'입니다. 
논문 PDF 파일을 분석하여 다음 항목들에 맞춰 상세한 **한국어 리포트**를 작성하십시오.
모든 수식은 LaTeX 포맷($...$)을 사용하고, 논문의 **Figure(아키텍처 그림)**와 **Table(성능표)**을 반드시 시각적으로 분석하여 내용에 포함해야 합니다.

---
**[작성 섹션]**

**1. 요약 (Executive Summary)**
   - Abstract 내용을 바탕으로 핵심 내용을 개조식(Bullet points)으로 요약하십시오.

**2. 7가지 핵심 질문 분석 (Key Analysis)** 
   - 각 질문에 대해 1~2 문단 내외로 답하시오.
   1) **What is new in the work?** (기존 연구와의 차별점)
   2) **Why is the work important?** (연구의 중요성)
   3) **What is the literature gap?** (기존 연구의 한계점)
   4) **How is the gap filled?** (해결 방안)
   5) **What is achieved with the new method?** (달성한 성과 - *여기서 Table의 수치를 인용할 것*)
   6) **What data are used?** (사용 데이터셋 - 도메인 특성 포함)
   7) **What are the limitations?** (저자가 언급한 한계점)

**3. 아키텍처 및 방법론 (Architecture & Methodology)**
   - **Figure 분석:** 메인 아키텍처 그림(Figure 1 등)을 보고, U-Net 구조에서 변경된 블록이나 흐름을 묘사하시오.
   - **수식 상세:** Loss Function, Input/Output Tensor Shape, 주요 모듈의 수식을 LaTeX로 엄밀하게 작성하시오.
   - **Vanilla U-Net 비교:** 기존 U-Net과 비교했을 때 정확히 어떤 모듈이 추가/수정되었는지 정리하시오.

**4. 태그 제안 (Tags Suggestion)** 
   - 논문의 핵심 키워드 5개를 추출하여 제안하시오.
---
"""

def upload_to_gemini(path, mime_type="application/pdf"):
    """Uploads the given file to Gemini. See https://ai.google.dev/gemini-api/docs/prompting_with_media"""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active. Some files generally take a few seconds to process."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

def analyze_paper(model, pdf_path):
    # Upload
    print(f"Uploading {pdf_path}...")
    try:
        pdf_file = upload_to_gemini(pdf_path, mime_type="application/pdf")
        wait_for_files_active([pdf_file])
    except Exception as e:
        print(f"Failed to upload {pdf_path}: {e}")
        return None

    # Generate
    print(f"Generating analysis for {pdf_path}...")
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    pdf_file,
                    "위 가이드라인에 따라 이 논문을 심층 분석하시오.",
                ],
            }
        ]
    )
    
    response = chat_session.send_message("분석을 시작하십시오.")
    return response.text

def main():
    # Setup Model
    # Setup Model
    generation_config = {
        "temperature": CONFIG['refiner']['temperature'],
        "top_p": CONFIG['refiner']['top_p'],
        "max_output_tokens": CONFIG['refiner']['max_output_tokens'],
    }
    
    model = genai.GenerativeModel(
        model_name=CONFIG['refiner']['model_name'],
        generation_config=generation_config,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    # Scan MD files
    md_files = glob.glob("content/posts/*.md")
    
    for md_file in md_files:
        post = frontmatter.load(md_file)
        
        # Check if draft and has pdf_path
        if not post.get('draft'):
            continue
            
        pdf_rel_path = post.get('params', {}).get('pdf_path')
        if not pdf_rel_path:
            continue
            
        # Refiner logic check: Don't re-process if already seems analyzed?
        # User constraint: "fill placeholders". We can check if placeholders exist.
        content = post.content
        if "Placeholder: Analyze" not in content:
            # Already processed? Check if we need to auto-publish or fix tags
            needs_save = False
            
            if post.get('draft'):
                print(f"Auto-publishing existing draft: {md_file}")
                post['draft'] = False
                needs_save = True

            # Fix Tags if "Auto-Generated" is present
            current_tags = post.get('tags', [])
            if "Auto-Generated" in current_tags or "Draft" in current_tags:
                print(f"Fixing tags for: {md_file}")
                import re
                # Try to find the tag section in EXISTING content
                tag_section = re.search(r"## 4\. 태그 제안 \(Tags Suggestion\)\s+([\s\S]+)$", content)
                if tag_section:
                    tags_text = tag_section.group(1)
                    extracted_tags = re.findall(r"\d+\.\s*(.+)", tags_text)
                    if extracted_tags:
                        clean_tags = [t.strip() for t in extracted_tags]
                        post['tags'] = clean_tags
                        print(f" -> Updated Tags: {clean_tags}")
                        needs_save = True
                
                # Also clean duplicate PDF headers while we are at it
                if "## PDF Download" in content:
                    new_clean_content = re.sub(r"(## PDF Download\s*){2,}", "## PDF Download\n", content)
                    if new_clean_content != content:
                        post.content = new_clean_content
                        print(" -> Cleaned duplicate PDF header")
                        needs_save = True

            if needs_save:
                with open(md_file, 'wb') as f:
                    frontmatter.dump(post, f)
                continue
            
            print(f"Skipping {md_file} (Already processed & up-to-date)")
            continue

        print(f"Processing Draft: {md_file}")
        
        # Resolve PDF Path
        # pdf_rel_path is like "data/papers/..."
        # We need absolute or relative to script execution
        if os.path.exists(pdf_rel_path):
            pdf_real_path = pdf_rel_path
        else:
            print(f"PDF not found: {pdf_rel_path}")
            continue

        # Call Gemini
        analysis_text = analyze_paper(model, pdf_real_path)
        
        if analysis_text:
            # Update Markdown Content
            # We will append the analysis text AFTER the "PDF Download" section, replacing the placeholders.
            # Actually, the user wants to "fill placeholders".
            # The structure is fixed. Let's just find the start of section 1 and replace everything after.
            
            # Simple approach: Keep Frontmatter + Abstract + PDF link.
            # Then append the new analysis.
            
            # Find the split point
            split_marker = "## 1. Visual Architecture Analysis"
            if split_marker in content:
                pre_content = content.split(split_marker)[0]
                new_content = pre_content + analysis_text
            else:
                # Fallback if marker not found
                new_content = content + "\n\n" + analysis_text
            
            post.content = new_content
            post.content = new_content
            
            # 1. Parse Tags from Content
            import re
            tag_section = re.search(r"## 4\. 태그 제안 \(Tags Suggestion\)\s+([\s\S]+)$", new_content)
            if tag_section:
                tags_text = tag_section.group(1)
                # Assume numbered list "1. TagName"
                extracted_tags = re.findall(r"\d+\.\s*(.+)", tags_text)
                if extracted_tags:
                    # Clean tags
                    clean_tags = [t.strip() for t in extracted_tags]
                    post['tags'] = clean_tags
                    print(f"Extracted Tags: {clean_tags}")
            
            # 2. Clean duplicate PDF Download Headers
            # Replace multiple occurrences of "## PDF Download" with single
            post.content = re.sub(r"(## PDF Download\s*){2,}", "## PDF Download\n", post.content)

            # Auto-publish per user request
            post['draft'] = False
            # User said: "be precise".
            
            # Save
            with open(md_file, 'wb') as f:
                frontmatter.dump(post, f)
            print(f"Updated {md_file}")
            
            # Respect rate limits being "one by one"
            time.sleep(5) 

if __name__ == "__main__":
    main()
