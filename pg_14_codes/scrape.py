import requests
from bs4 import BeautifulSoup
import json

# All PostgreSQL 14 runtime configuration documentation pages
doc_urls = [
    # Core performance tuning (most critical)
    "https://www.postgresql.org/docs/14/runtime-config-resource.html",      # Memory, disk, kernel resources
    "https://www.postgresql.org/docs/14/runtime-config-wal.html",           # Write-Ahead Logging
    "https://www.postgresql.org/docs/14/runtime-config-query.html",         # Query planner cost estimates
    "https://www.postgresql.org/docs/14/runtime-config-autovacuum.html",    # Autovacuum tuning
    
    # Connection & concurrency
    "https://www.postgresql.org/docs/14/runtime-config-connection.html",    # Connection settings
    "https://www.postgresql.org/docs/14/runtime-config-client.html",        # Client connection defaults
    
    # Monitoring & troubleshooting
    "https://www.postgresql.org/docs/14/runtime-config-logging.html",       # Logging configuration
    "https://www.postgresql.org/docs/14/runtime-config-statistics.html",    # Statistics collection
    
    # Advanced features
    "https://www.postgresql.org/docs/14/runtime-config-replication.html",   # Replication settings
    "https://www.postgresql.org/docs/14/runtime-config-locks.html",         # Lock management
]

def load_target_knobs():
    with open('../knowledge/postgres/all_knobs.txt', 'r') as f:
        return set(line.strip() for line in f if line.strip())

target_knobs = load_target_knobs()
knobs_dict = {}

for url in doc_urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all parameter sections
    for varlist in soup.find_all('div', class_='variablelist'):
        for item in varlist.find_all('dt'):
            varname_tag = item.find('code', class_='varname')
            if varname_tag:
                name = varname_tag.get_text(strip=True)
                dd = item.find_next_sibling('dd')
                if dd:
                    # Get all paragraphs in description
                    desc_parts = []
                    for p in dd.find_all('p'):
                        desc_parts.append(p.get_text(strip=True, separator=' '))
                    description = ' '.join(desc_parts)
                    knobs_dict[name] = description

# Filter to only include knobs from all_knobs.txt
filtered_knobs = [{"name": k, "description": v} for k, v in knobs_dict.items() if k in target_knobs]

print(f"Scraped {len(knobs_dict)} knobs, filtered to {len(filtered_knobs)} from all_knobs.txt")

with open('postgres_knob_from_docs.json', 'w') as f:
    json.dump(filtered_knobs, f, indent=2)