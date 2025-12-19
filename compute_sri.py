import hashlib
import base64
import requests

urls = [
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js",
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
]

for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        hash_obj = hashlib.sha384(response.content)
        base64_hash = base64.b64encode(hash_obj.digest()).decode('utf-8')
        print(f"URL: {url}")
        print(f"Integrity: sha384-{base64_hash}")
        print("-" * 20)
    else:
        print(f"Failed to fetch {url}")
