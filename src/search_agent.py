import arxiv
import os
import re
import json
import yaml
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Load Config
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

class HistoryManager:
    def __init__(self, history_file):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"processed_arxiv_ids": []}

    def _save_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def is_processed(self, arxiv_id):
        return arxiv_id in self.history["processed_arxiv_ids"]

    def add(self, arxiv_id):
        if arxiv_id not in self.history["processed_arxiv_ids"]:
            self.history["processed_arxiv_ids"].append(arxiv_id)
            self._save_history()

def sanitize_filename(title):
    # Retrieve only alphanumeric characters, spaces, and hyphens, then replace spaces with hyphens
    # Convert to lowercase
    clean = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
    clean = re.sub(r'\s+', '-', clean).lower()
    return clean

def generate_markdown(result, pdf_path):
    # Prepare Frontmatter
    title = result.title.replace('"', '\\"')
    date_str = result.published.strftime('%Y-%m-%d')
    arxiv_id = result.get_short_id()
    pdf_rel_path = f"data/papers/{arxiv_id}.pdf" # Relative path for Hugo
    abs_text = result.summary.replace('\n', ' ')

    md_content = f"""---
title: "{title}"
date: {date_str}
categories: ["Literature Review", "U-Net"]
tags: ["Auto-Generated", "Draft"]
draft: true
params:
  arxiv_id: "{arxiv_id}"
  pdf_path: "{pdf_path.replace(os.sep, '/')}"
  arxiv_link: "{result.entry_id}"
---

## Abstract
{abs_text}

## PDF Download
## PDF Download
[Local PDF View]({pdf_path.replace(os.sep, '/')}) | [Arxiv Original]({result.entry_id})

## 1. Visual Architecture Analysis (To be filled)
> Placeholder: Analyze the main architecture diagram (Figure 1/2).

## 2. Performance & Tables (To be filled)
> Placeholder: Extract SOTA metrics from tables.

## 3. Critical Review
> Placeholder: Strengths and limitations based on visual & text analysis.

"""
    return md_content

def main():
    # Load Storage Path
    nas_path = os.getenv("NAS_PAPER_PATH")
    if nas_path and os.path.exists(nas_path):
        save_dir = nas_path
        print(f"Using NAS Storage: {save_dir}")
    else:
        save_dir = os.path.join(os.getcwd(), "data", "papers")
        if nas_path:
            print(f"Warning: NAS path '{nas_path}' not found. Falling back to {save_dir}")
        else:
            print(f"Using Local Storage: {save_dir}")

    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("content/posts", exist_ok=True)
    
    # Initialize History
    history_manager = HistoryManager(CONFIG['paths']['history_file'])
    os.makedirs(os.path.dirname(CONFIG['paths']['history_file']), exist_ok=True)

    # Search Query from Config
    query = CONFIG['search']['query']
    max_results = CONFIG['search']['max_results']
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    client = arxiv.Client()
    
    print(f"Searching for query: {query}")
    
    results = list(client.results(search))
    print(f"Found {len(results)} papers.")

    for result in results:
        arxiv_id = result.get_short_id()
        title = result.title
        
        if history_manager.is_processed(arxiv_id):
            print(f"Skipping [{arxiv_id}] (Already processed)")
            continue
            
        print(f"Processing: [{arxiv_id}] {title}")

        # Download PDF
        pdf_filename = f"{arxiv_id}.pdf"
        # download_pdf saves to dirpath/filename
        # Check if already exists to avoid re-downloading if run multiple times (optional, but good)
        # But instructions say "Action: search and download", so we will force or just let it download.
        # arxiv library skips if exists usually if configured, but let's just call download.
        
        try:
            pdf_path = result.download_pdf(dirpath=save_dir, filename=pdf_filename)
            print(f" - Downloaded PDF to {pdf_path}")
        except Exception as e:
            print(f" - Failed to download PDF: {e}")
            continue

        # Generate Hugo Markdown
        kebab_title = sanitize_filename(title)
        md_filename = f"{kebab_title}.md"
        md_path = os.path.join("content/posts", md_filename)

        md_content = generate_markdown(result, pdf_path)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        # Add to history after successful generation
        history_manager.add(arxiv_id)
        
        print(f" - Generated Draft: {md_path}")

if __name__ == "__main__":
    main()
