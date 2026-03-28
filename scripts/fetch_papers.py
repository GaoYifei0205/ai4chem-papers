#!/usr/bin/env python3
"""
AI for Chemistry Paper Fetcher
Fetches papers from arXiv, Semantic Scholar, HuggingFace Papers, and CrossRef.
Runs daily at 10:00 AM CST via GitHub Actions.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_FILE = ROOT_DIR / "data" / "papers.json"
INDEX_FILE = ROOT_DIR / "index.html"

CST = timezone(timedelta(hours=8))

CATEGORY_META = {
    "molecular_generation": {"label": "分子生成", "color": "#6366f1", "icon": "⚗️"},
    "drug_design":          {"label": "药物设计", "color": "#ec4899", "icon": "💊"},
    "reaction_prediction":  {"label": "反应预测", "color": "#f59e0b", "icon": "🔄"},
    "materials_science":    {"label": "材料科学", "color": "#10b981", "icon": "🔬"},
    "llm_chemistry":        {"label": "LLM化学",  "color": "#3b82f6", "icon": "🤖"},
    "ai_agent":             {"label": "AI Agent", "color": "#8b5cf6", "icon": "🦾"},
}

# arXiv queries: (search_query, [categories_to_assign])
ARXIV_QUERIES = [
    # Molecular generation
    ('all:"molecular generation" OR all:"de novo drug" OR all:"molecule generation" OR all:"SMILES generation"',
     ["molecular_generation"]),
    ('all:"GFlowNet" OR all:"diffusion model molecule" OR all:"score-based molecule"',
     ["molecular_generation"]),
    # Drug design
    ('all:"structure-based drug design" OR all:"SBDD" OR all:"binding affinity prediction" OR all:"ADMET prediction"',
     ["drug_design"]),
    ('all:"drug-target interaction" OR all:"drug-target affinity" OR all:"molecular docking" OR all:"lead optimization"',
     ["drug_design"]),
    # Reaction prediction
    ('all:"retrosynthesis" OR all:"reaction prediction" OR all:"reaction yield" OR all:"USPTO"',
     ["reaction_prediction"]),
    ('all:"reaction template" OR all:"forward synthesis" OR all:"reaction condition"',
     ["reaction_prediction"]),
    # Materials science
    ('all:"crystal structure prediction" OR all:"machine learning force field" OR all:"neural network potential"',
     ["materials_science"]),
    ('all:"materials discovery" OR all:"materials design machine learning" OR all:"battery machine learning"',
     ["materials_science"]),
    # LLM for Chemistry
    ('all:"chemical language model" OR all:"LLM chemistry" OR all:"molecular reasoning" OR all:"ChemBench"',
     ["llm_chemistry"]),
    ('all:"chemistry large language model" OR all:"molecular LLM" OR all:"chemical foundation model"',
     ["llm_chemistry"]),
    # AI Agent
    ('all:"self-driving lab" OR all:"autonomous synthesis" OR all:"robotic chemistry" OR all:"AI agent chemistry"',
     ["ai_agent"]),
]

# CrossRef journals for Nature/Science/中科院一区
CROSSREF_JOURNALS = [
    "Nature",
    "Nature Chemistry",
    "Nature Machine Intelligence",
    "Nature Computational Science",
    "Science",
    "Science Advances",
    "Journal of the American Chemical Society",
    "Angewandte Chemie",
    "ACS Central Science",
    "Chemical Science",
    "Journal of Chemical Information and Modeling",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ai4chem-papers/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def yesterday_str() -> str:
    return (datetime.now(CST) - timedelta(days=1)).strftime("%Y-%m-%d")


def today_str() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def load_existing() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"last_updated": "", "papers": []}


def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── arXiv Fetcher ─────────────────────────────────────────────────────────────

def fetch_arxiv(target_date: str) -> list[dict]:
    """Fetch papers from arXiv submitted on target_date."""
    papers = []
    seen_ids = set()
    base_url = "http://export.arxiv.org/api/query?"

    for query, categories in ARXIV_QUERIES:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 50,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = base_url + urllib.parse.urlencode(params)
        try:
            xml_text = fetch_url(url)
            time.sleep(3)  # arXiv rate limit
        except Exception as e:
            print(f"  arXiv fetch error: {e}")
            continue

        ns = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_text)

        for entry in root.findall("atom:entry", ns):
            published = entry.findtext("atom:published", "", ns)[:10]
            updated   = entry.findtext("atom:updated", "", ns)[:10]

            # Accept if submitted or updated on target_date
            if published != target_date and updated != target_date:
                continue

            raw_id = entry.findtext("atom:id", "", ns)
            arxiv_id = raw_id.split("/abs/")[-1].replace("v1","").replace("v2","").strip()
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            title = re.sub(r"\s+", " ", entry.findtext("atom:title", "", ns)).strip()
            summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", "", ns)).strip()
            # First 300 chars as summary
            summary_short = summary[:300] + ("…" if len(summary) > 300 else "")

            authors = [a.findtext("atom:name", "", ns)
                       for a in entry.findall("atom:author", ns)]
            # Try to extract affiliations from arxiv:affiliation
            institutions = list({
                aff.text.strip()
                for a in entry.findall("atom:author", ns)
                for aff in a.findall("arxiv:affiliation", ns)
                if aff.text
            })

            papers.append({
                "id": arxiv_id,
                "title": title,
                "authors": authors[:6],
                "institutions": institutions[:3],
                "date": published,
                "categories": categories,
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "abstract_summary": summary_short,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "hf_likes": 0,
                "citation_count": 0,
                "badges": [],
            })

    return papers


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def enrich_with_s2(papers: list[dict]) -> list[dict]:
    """Add citation counts from Semantic Scholar for papers with arXiv IDs."""
    s2_api = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount,influentialCitationCount"
    s2_key = os.environ.get("S2_API_KEY", "")
    headers = {"x-api-key": s2_key} if s2_key else {}

    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            continue
        url = s2_api.format(arxiv_id=arxiv_id)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "ai4chem-papers/1.0", **headers})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            paper["citation_count"] = data.get("citationCount", 0)
            if data.get("citationCount", 0) >= 5:
                paper["badges"].append("cited")
            time.sleep(1)
        except Exception:
            pass
    return papers


# ── HuggingFace Papers ────────────────────────────────────────────────────────

def fetch_hf_likes(papers: list[dict], target_date: str) -> list[dict]:
    """Try to match papers against HuggingFace daily papers and get like counts."""
    url = f"https://huggingface.co/papers?date={target_date}"
    try:
        html = fetch_url(url)
    except Exception:
        return papers

    # Extract arXiv IDs and like counts from HF page
    # Pattern: /papers/2603.XXXXX and upvote numbers
    hf_data = {}
    matches = re.findall(r'/papers/(\d{4}\.\d+)[^"]*"[^>]*>.*?(\d+)\s*(?:upvote|like)', html, re.S)
    for arxiv_id, likes in matches:
        hf_data[arxiv_id] = int(likes)

    # Also simpler pattern
    for m in re.finditer(r'/papers/([\d.]+)', html):
        aid = m.group(1)
        if aid not in hf_data:
            hf_data[aid] = 0

    for paper in papers:
        aid = paper.get("arxiv_id", "")
        if aid in hf_data:
            paper["hf_likes"] = hf_data[aid]
            if hf_data[aid] >= 10:
                if "hf_hot" not in paper["badges"]:
                    paper["badges"].append("hf_hot")
    return papers


# ── CrossRef (Nature/Science/中科院一区) ──────────────────────────────────────

def fetch_crossref(target_date: str) -> list[dict]:
    """Fetch papers from high-impact journals via CrossRef API."""
    papers = []
    seen_dois = set()
    keywords = [
        "molecular generation", "drug design", "retrosynthesis",
        "crystal structure prediction", "machine learning force field",
        "chemical language model", "reaction prediction",
        "molecular property prediction", "materials discovery",
    ]

    for journal in CROSSREF_JOURNALS:
        for kw in keywords[:3]:  # Limit queries per journal
            url = (
                f"https://api.crossref.org/works?"
                f"query={urllib.parse.quote(kw)}"
                f"&filter=container-title:{urllib.parse.quote(journal)}"
                f",from-pub-date:{target_date},until-pub-date:{target_date}"
                f"&rows=5&mailto=ai4chem@example.com"
            )
            try:
                raw = fetch_url(url)
                data = json.loads(raw)
                time.sleep(1)
            except Exception:
                continue

            for item in data.get("message", {}).get("items", []):
                doi = item.get("DOI", "")
                if not doi or doi in seen_dois:
                    continue
                seen_dois.add(doi)

                title_list = item.get("title", [""])
                title = title_list[0] if title_list else ""
                if not title:
                    continue

                authors_raw = item.get("author", [])
                authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                           for a in authors_raw[:6]]
                institutions = list({
                    aff.get("name", "")
                    for a in authors_raw
                    for aff in a.get("affiliation", [])
                    if aff.get("name")
                })[:3]

                pub_date_parts = item.get("published", {}).get("date-parts", [[]])
                pub_date = ""
                if pub_date_parts and pub_date_parts[0]:
                    parts = pub_date_parts[0]
                    pub_date = f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}" if len(parts) >= 3 else target_date

                journal_name = item.get("container-title", [""])[0] if item.get("container-title") else ""
                abstract = re.sub(r"<[^>]+>", "", item.get("abstract", ""))[:300]

                # Infer category from keywords in title/abstract
                cats = infer_categories(title + " " + abstract)

                papers.append({
                    "id": doi.replace("/", "_"),
                    "title": title,
                    "authors": authors,
                    "institutions": institutions,
                    "date": pub_date or target_date,
                    "categories": cats or ["llm_chemistry"],
                    "source": "journal",
                    "journal": journal_name,
                    "doi": doi,
                    "doi_url": f"https://doi.org/{doi}",
                    "abstract_summary": abstract,
                    "hf_likes": 0,
                    "citation_count": item.get("is-referenced-by-count", 0),
                    "badges": ["journal"],
                })

    return papers


def infer_categories(text: str) -> list[str]:
    text_l = text.lower()
    cats = []
    if any(w in text_l for w in ["molecular generation", "de novo", "smiles generation", "gflownet"]):
        cats.append("molecular_generation")
    if any(w in text_l for w in ["drug design", "binding affinity", "admet", "docking", "drug-target"]):
        cats.append("drug_design")
    if any(w in text_l for w in ["retrosynthesis", "reaction prediction", "reaction yield", "uspto"]):
        cats.append("reaction_prediction")
    if any(w in text_l for w in ["crystal structure", "force field", "materials", "battery"]):
        cats.append("materials_science")
    if any(w in text_l for w in ["language model", "llm", "gpt", "bert", "reasoning"]):
        cats.append("llm_chemistry")
    if any(w in text_l for w in ["agent", "autonomous", "self-driving", "robotic"]):
        cats.append("ai_agent")
    return cats


# ── HTML Generator ────────────────────────────────────────────────────────────

def generate_html(data: dict) -> str:
    papers = sorted(data["papers"], key=lambda p: p.get("date", ""), reverse=True)
    last_updated = data.get("last_updated", "")

    # Group by date
    by_date: dict[str, list] = {}
    for p in papers:
        d = p.get("date", "unknown")
        by_date.setdefault(d, []).append(p)

    # Build timeline HTML
    timeline_html = ""
    for date in sorted(by_date.keys(), reverse=True):
        day_papers = by_date[date]
        timeline_html += f'<div class="day-group" data-date="{escape(date)}">\n'
        timeline_html += f'  <div class="date-header">{escape(date)}</div>\n'
        for p in day_papers:
            timeline_html += render_paper_card(p)
        timeline_html += "</div>\n"

    category_checkboxes = ""
    for cat_id, meta in CATEGORY_META.items():
        category_checkboxes += f"""
        <label class="filter-item">
          <input type="checkbox" class="cat-filter" value="{cat_id}" checked>
          <span class="cat-dot" style="background:{meta['color']}"></span>
          {meta['icon']} {meta['label']}
        </label>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI for Chemistry Papers</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface2: #252840;
    --border: #2d3154;
    --text: #e2e8f0;
    --text-muted: #8892b0;
    --accent: #6366f1;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Pro Display', -apple-system, 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }}
  header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  header h1 {{ font-size: 1.3rem; font-weight: 700; color: #a5b4fc; }}
  .update-time {{ font-size: 0.8rem; color: var(--text-muted); }}
  .layout {{ display: flex; flex: 1; gap: 0; }}

  /* Sidebar */
  aside {{
    width: 220px;
    min-width: 220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 16px;
    position: sticky;
    top: 57px;
    height: calc(100vh - 57px);
    overflow-y: auto;
  }}
  aside h3 {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 10px;
    margin-top: 18px;
  }}
  aside h3:first-child {{ margin-top: 0; }}
  .filter-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 4px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.88rem;
    transition: background 0.15s;
  }}
  .filter-item:hover {{ background: var(--surface2); }}
  .filter-item input {{ accent-color: var(--accent); cursor: pointer; }}
  .cat-dot {{
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }}
  .stats {{
    margin-top: 24px;
    padding: 12px;
    background: var(--surface2);
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.8;
  }}

  /* Main timeline */
  main {{
    flex: 1;
    padding: 24px 32px;
    max-width: 900px;
  }}
  .day-group {{ margin-bottom: 32px; }}
  .date-header {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin-bottom: 12px;
    letter-spacing: 0.03em;
  }}

  /* Paper card */
  .paper-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.15s;
  }}
  .paper-card:hover {{
    border-color: var(--accent);
    transform: translateY(-1px);
  }}
  .paper-title {{
    font-size: 0.97rem;
    font-weight: 600;
    color: #c7d2fe;
    text-decoration: none;
    line-height: 1.4;
  }}
  .paper-title:hover {{ color: #a5b4fc; text-decoration: underline; }}
  .paper-meta {{
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }}
  .tag {{
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
    white-space: nowrap;
  }}
  .tag-source {{
    background: var(--surface2);
    color: var(--text-muted);
    border: 1px solid var(--border);
  }}
  .tag-institution {{
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
  }}
  .badge-hf {{ background: #fef3c7; color: #92400e; }}
  .badge-cited {{ background: #dcfce7; color: #166534; }}
  .badge-journal {{ background: #ede9fe; color: #4c1d95; }}
  .badge-code {{ background: #cffafe; color: #164e63; }}
  .badge-top {{ background: #fee2e2; color: #991b1b; }}
  .paper-authors {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 5px;
  }}
  .paper-abstract {{
    font-size: 0.83rem;
    color: #94a3b8;
    margin-top: 8px;
    line-height: 1.6;
    display: none;
  }}
  .paper-abstract.open {{ display: block; }}
  .paper-links {{
    margin-top: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .btn-link {{
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 6px;
    text-decoration: none;
    border: 1px solid var(--border);
    color: var(--text-muted);
    background: var(--surface2);
    transition: all 0.15s;
  }}
  .btn-link:hover {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }}
  .toggle-abstract {{
    font-size: 0.78rem;
    color: var(--accent);
    cursor: pointer;
    background: none;
    border: none;
    padding: 0;
    margin-top: 6px;
  }}
  .toggle-abstract:hover {{ text-decoration: underline; }}
  .hidden {{ display: none !important; }}

  /* Search */
  .search-wrap {{ margin-bottom: 20px; }}
  #search-input {{
    width: 100%;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s;
  }}
  #search-input:focus {{ border-color: var(--accent); }}

  @media (max-width: 768px) {{
    .layout {{ flex-direction: column; }}
    aside {{
      width: 100%;
      min-width: unset;
      position: static;
      height: auto;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
    }}
    main {{ padding: 16px; max-width: 100%; }}
  }}
</style>
</head>
<body>
<header>
  <h1>🧪 AI for Chemistry Papers</h1>
  <span class="update-time">最后更新: {escape(last_updated)}</span>
</header>
<div class="layout">
  <aside>
    <h3>研究方向</h3>
    {category_checkboxes}
    <h3>来源</h3>
    <label class="filter-item">
      <input type="checkbox" class="src-filter" value="arxiv" checked>
      📄 arXiv
    </label>
    <label class="filter-item">
      <input type="checkbox" class="src-filter" value="journal" checked>
      📰 期刊
    </label>
    <div class="stats" id="stats">
      加载中…
    </div>
  </aside>
  <main>
    <div class="search-wrap">
      <input type="text" id="search-input" placeholder="搜索标题、作者、机构…">
    </div>
    <div id="timeline">
{timeline_html}
    </div>
    <div id="no-results" class="hidden" style="color:var(--text-muted);text-align:center;padding:40px;">
      没有找到匹配的论文
    </div>
  </main>
</div>
<script>
(function() {{
  var cards = Array.from(document.querySelectorAll('.paper-card'));
  var dayGroups = Array.from(document.querySelectorAll('.day-group'));

  function getActiveCats() {{
    return Array.from(document.querySelectorAll('.cat-filter:checked')).map(e => e.value);
  }}
  function getActiveSrcs() {{
    return Array.from(document.querySelectorAll('.src-filter:checked')).map(e => e.value);
  }}
  function getSearchTerm() {{
    return document.getElementById('search-input').value.toLowerCase().trim();
  }}

  function applyFilters() {{
    var cats = getActiveCats();
    var srcs = getActiveSrcs();
    var term = getSearchTerm();
    var visible = 0;

    cards.forEach(function(card) {{
      var cardCats = (card.dataset.categories || '').split(',');
      var cardSrc = card.dataset.source || '';
      var cardText = card.dataset.searchtext || '';
      var catOk = cats.some(c => cardCats.includes(c));
      var srcOk = srcs.includes(cardSrc);
      var termOk = !term || cardText.includes(term);
      if (catOk && srcOk && termOk) {{
        card.classList.remove('hidden');
        visible++;
      }} else {{
        card.classList.add('hidden');
      }}
    }});

    // Hide empty day groups
    dayGroups.forEach(function(group) {{
      var anyVisible = group.querySelectorAll('.paper-card:not(.hidden)').length > 0;
      group.style.display = anyVisible ? '' : 'none';
    }});

    document.getElementById('no-results').classList.toggle('hidden', visible > 0);
    document.getElementById('stats').textContent = '显示 ' + visible + ' / ' + cards.length + ' 篇';
  }}

  document.querySelectorAll('.cat-filter, .src-filter').forEach(function(cb) {{
    cb.addEventListener('change', applyFilters);
  }});
  document.getElementById('search-input').addEventListener('input', applyFilters);

  // Abstract toggle
  document.querySelectorAll('.toggle-abstract').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var ab = this.parentElement.querySelector('.paper-abstract');
      if (ab) {{
        ab.classList.toggle('open');
        this.textContent = ab.classList.contains('open') ? '▲ 收起摘要' : '▼ 展开摘要';
      }}
    }});
  }});

  applyFilters();
}})();
</script>
</body>
</html>"""


def render_paper_card(p: dict) -> str:
    cats = p.get("categories", [])
    source = p.get("source", "arxiv")
    title = escape(p.get("title", "Untitled"))
    arxiv_url = p.get("arxiv_url", p.get("doi_url", "#"))
    pdf_url = p.get("pdf_url", "")
    doi_url = p.get("doi_url", "")
    code_url = p.get("code_url", "")
    abstract = escape(p.get("abstract_summary", ""))
    authors = escape(", ".join(p.get("authors", [])[:4]))
    if len(p.get("authors", [])) > 4:
        authors += " et al."
    institutions = p.get("institutions", [])
    badges = p.get("badges", [])
    hf_likes = p.get("hf_likes", 0)
    citations = p.get("citation_count", 0)
    journal = escape(p.get("journal", ""))

    # Category tags
    cat_tags = ""
    for c in cats:
        meta = CATEGORY_META.get(c, {"label": c, "color": "#6b7280", "icon": ""})
        cat_tags += f'<span class="tag" style="background:{meta["color"]}22;color:{meta["color"]};border:1px solid {meta["color"]}55">{meta["icon"]} {escape(meta["label"])}</span>'

    # Source tag
    src_label = journal if journal else ("arXiv" if source == "arxiv" else source)
    source_tag = f'<span class="tag tag-source">{escape(src_label)}</span>'

    # Institution tags
    inst_tags = ""
    for inst in institutions[:2]:
        inst_tags += f'<span class="tag tag-institution">🏛 {escape(inst)}</span>'

    # Badge tags
    badge_html = ""
    if "hf_hot" in badges or hf_likes >= 10:
        badge_html += f'<span class="tag badge-hf">⭐ HF {hf_likes}</span>'
    if "cited" in badges or citations >= 5:
        badge_html += f'<span class="tag badge-cited">📈 {citations} 引用</span>'
    if "journal" in badges:
        badge_html += '<span class="tag badge-journal">📰 期刊</span>'
    if "code" in badges:
        badge_html += '<span class="tag badge-code">💻 代码</span>'
    if "top_institution" in badges:
        badge_html += '<span class="tag badge-top">🏆 顶级机构</span>'

    # Links
    links = f'<a class="btn-link" href="{escape(arxiv_url)}" target="_blank" rel="noopener">arXiv</a>'
    if pdf_url:
        links += f' <a class="btn-link" href="{escape(pdf_url)}" target="_blank" rel="noopener">PDF</a>'
    if doi_url and doi_url != arxiv_url:
        links += f' <a class="btn-link" href="{escape(doi_url)}" target="_blank" rel="noopener">DOI</a>'
    if code_url:
        links += f' <a class="btn-link" href="{escape(code_url)}" target="_blank" rel="noopener">Code</a>'

    cats_str = ",".join(cats)
    search_text = (title + " " + authors + " " + " ".join(institutions)).lower()

    return f"""  <div class="paper-card" data-categories="{escape(cats_str)}" data-source="{escape(source)}" data-searchtext="{escape(search_text)}">
    <a class="paper-title" href="{escape(arxiv_url)}" target="_blank" rel="noopener">{title}</a>
    <div class="paper-meta">
      {source_tag}
      {cat_tags}
      {inst_tags}
      {badge_html}
    </div>
    <div class="paper-authors">{authors}</div>
    <div class="paper-abstract">{abstract}</div>
    <button class="toggle-abstract">▼ 展开摘要</button>
    <div class="paper-links">{links}</div>
  </div>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    target_date = os.environ.get("TARGET_DATE", yesterday_str())
    print(f"Fetching papers for date: {target_date}")

    existing = load_existing()
    existing_ids = {p["id"] for p in existing["papers"]}

    new_papers: list[dict] = []

    print("Fetching from arXiv…")
    arxiv_papers = fetch_arxiv(target_date)
    print(f"  Found {len(arxiv_papers)} arXiv papers")
    new_papers.extend(arxiv_papers)

    print("Fetching from CrossRef (journals)…")
    journal_papers = fetch_crossref(target_date)
    print(f"  Found {len(journal_papers)} journal papers")
    new_papers.extend(journal_papers)

    # Deduplicate
    unique_new = [p for p in new_papers if p["id"] not in existing_ids]
    print(f"Adding {len(unique_new)} new unique papers")

    print("Enriching with Semantic Scholar citation counts…")
    unique_new = enrich_with_s2(unique_new)

    print("Fetching HuggingFace likes…")
    unique_new = fetch_hf_likes(unique_new, target_date)

    existing["papers"].extend(unique_new)
    existing["last_updated"] = datetime.now(CST).strftime("%Y-%m-%d %H:%M CST")

    save_data(existing)
    print(f"Saved {len(existing['papers'])} total papers to {DATA_FILE}")

    print("Generating index.html…")
    html = generate_html(existing)
    with open(INDEX_FILE, "w") as f:
        f.write(html)
    print(f"Written to {INDEX_FILE}")


if __name__ == "__main__":
    main()
