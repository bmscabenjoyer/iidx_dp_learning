"""
Scrape zasa.sakura.ne.jp/dp/run.php for DP difficulty ratings.

Outputs ~/projects/iidx_data/zasa_ratings.csv with columns:
  title, level, diftype, zasa_rating, zasa_url, manifest_title, match_score

Only targets lv10 and lv11 charts (lv12 is already covered by ereter).
Matches to the existing labeled_manifest.csv by fuzzy title + exact level/diftype.
"""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

# Borrow the curated ereter↔textage title aliases so zasa mismatches get the
# same treatment (zasa shares the same discrepancies as ereter vs textage).
sys.path.insert(0, str(Path('/home/jysuh/projects/ereterextractor')))
from title_aliases import TITLE_ALIASES, TITLE_ALIASES_LEVELED  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
URL            = 'https://zasa.sakura.ne.jp/dp/run.php'
MANIFEST_PATH  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
OUTPUT_PATH    = Path('/home/jysuh/projects/iidx_data/zasa_ratings.csv')
TARGET_LEVELS  = {10, 11}
FUZZY_THRESH   = 80   # minimum rapidfuzz score to accept a title match

SPAN_CLASS_TO_DIFTYPE = {
    'N': '[DP NORMAL]',
    'H': '[DP HYPER]',
    'A': '[DP ANOTHER]',
    'L': '[DP LEGGENDARIA]',
}

STAR_RE   = re.compile(r'☆(\d+)')
RATING_RE = re.compile(r'\(([\d.]+)\)')

# Zasa-specific overrides not covered by the ereter alias dict.
# ƒ (U+0192) doesn't NFKC-collapse to f; ə (U+0259) ≠ ә (U+04D9); ♡ without
# surrounding spaces doesn't round-trip through the ereter alias keys.
# Map raw zasa title → correct textage title (applied before normalize()).
ZASA_OVERRIDES: dict[str, str] = {
    'Sweet Sweet♡Magic':   'Sweet Sweet ♥ Magic',
    'Double♡♡Loving Heart': 'Double ♥♥ Loving Heart',
    'Punch Love♡仮面':      'Punch Love ♥ 仮面',
    'ƒƒƒƒƒ':               'fffff',
    'uən':                 'uәn',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(s) -> str:
    """NFKC + strip + lowercase + cosmetic canonicalization for fuzzy comparison."""
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('♡', '♥')
    # strip spaces around brackets and wave dashes
    s = re.sub(r'\s*([（）()～〜])\s*', r'\1', s)
    # normalize spaces around hyphens used as subtitle delimiters
    s = re.sub(r'\s+-\s*|\s*-\s+', '-', s)
    # normalize typographic quotes
    s = re.sub(r'["“”]', '"', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def parse_cell(td) -> dict | None:
    """Extract (star_level, rating, diftype, url) from a rank <td>, or None."""
    a = td.find('a', class_='music')
    if not a:
        return None
    span = a.find('span')
    if not span:
        return None
    text = span.get_text(strip=True)
    star_m   = STAR_RE.search(text)
    rating_m = RATING_RE.search(text)
    if not star_m or not rating_m:
        return None
    span_cls = span.get('class', [''])[0]
    diftype  = SPAN_CLASS_TO_DIFTYPE.get(span_cls)
    if diftype is None:
        return None
    return {
        'level':       int(star_m.group(1)),
        'zasa_rating': float(rating_m.group(1)),
        'diftype':     diftype,
        'zasa_url':    'https://zasa.sakura.ne.jp/dp/' + a['href'],
    }


# ── Scrape ────────────────────────────────────────────────────────────────────
print(f'Fetching {URL} ...', flush=True)
resp = requests.get(URL, timeout=20)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, 'lxml')

table = soup.find('table', class_='run')
if not table:
    sys.exit('ERROR: could not find <table class="run"> on page')

# Each data row has 4 tds: [H, A, L, title]
zasa_entries: list[dict] = []
for tr in table.find_all('tr'):
    tds = tr.find_all('td')
    if len(tds) != 4:
        continue   # header rows have <th>
    title_td = tds[3]
    title    = title_td.get_text(strip=True)
    if not title:
        continue
    for td in tds[:3]:   # H, A, L columns
        parsed = parse_cell(td)
        if parsed and parsed['level'] in TARGET_LEVELS:
            zasa_entries.append({'zasa_title': title, **parsed})

print(f'Found {len(zasa_entries)} zasa entries at levels {sorted(TARGET_LEVELS)}')

# ── Load manifest ─────────────────────────────────────────────────────────────
manifest = pd.read_csv(MANIFEST_PATH)
manifest = manifest[manifest['level'].isin(TARGET_LEVELS)].copy()
manifest['_norm_title'] = manifest['title'].apply(normalize)
print(f'Manifest entries at target levels: {len(manifest)}')

# ── Match ─────────────────────────────────────────────────────────────────────
# Build lookup: (norm_title, level, diftype) -> manifest row index
manifest_keys = list(zip(
    manifest['_norm_title'],
    manifest['level'].astype(int),
    manifest['diftype'],
))

results = []
unmatched = []

for entry in zasa_entries:
    norm_zasa = normalize(entry['zasa_title'])
    level     = entry['level']
    diftype   = entry['diftype']

    # Resolve title: zasa-specific overrides first, then ereter alias dict.
    raw_zasa  = entry['zasa_title']
    short_dif = diftype.replace('[DP ', '').rstrip(']')
    canonical = (ZASA_OVERRIDES.get(raw_zasa)
                 or TITLE_ALIASES_LEVELED.get((raw_zasa, short_dif, level))
                 or TITLE_ALIASES.get(raw_zasa))
    if canonical is not None:
        norm_zasa = normalize(canonical)

    # Filter manifest candidates to same level + diftype
    candidates = [
        (manifest.index[i], manifest_keys[i][0])
        for i, k in enumerate(manifest_keys)
        if k[1] == level and k[2] == diftype
    ]
    if not candidates:
        unmatched.append({**entry, 'reason': 'no manifest entry at this level/diftype'})
        continue

    cand_indices  = [c[0] for c in candidates]
    cand_titles   = [c[1] for c in candidates]

    match = process.extractOne(
        norm_zasa, cand_titles,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=FUZZY_THRESH,
    )
    if match is None:
        unmatched.append({**entry, 'reason': f'no fuzzy match above {FUZZY_THRESH}'})
        continue

    matched_norm, score, idx_in_cands = match
    mrow = manifest.loc[cand_indices[idx_in_cands]]
    results.append({
        'title':          mrow['title'],
        'level':          int(level),
        'diftype':        diftype,
        'zasa_rating':    entry['zasa_rating'],
        'zasa_url':       entry['zasa_url'],
        'zasa_title':     entry['zasa_title'],
        'match_score':    score,
        'file_path':      mrow['file_path'],
    })

# ── Output ────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values(['level', 'diftype', 'title'])
df.to_csv(OUTPUT_PATH, index=False)

print(f'\nMatched:   {len(results):4d}')
print(f'Unmatched: {len(unmatched):4d}')
print(f'Saved to:  {OUTPUT_PATH}')

# Rating distribution by level
print('\nZasa rating distribution:')
for lvl, grp in df.groupby('level'):
    print(f'  lv{lvl}: n={len(grp):4d}  '
          f'mean={grp["zasa_rating"].mean():.2f}  '
          f'std={grp["zasa_rating"].std():.2f}  '
          f'min={grp["zasa_rating"].min():.1f}  '
          f'max={grp["zasa_rating"].max():.1f}')

# Low-confidence matches worth reviewing
low_conf = df[df['match_score'] < 90]
if len(low_conf):
    print(f'\nLow-confidence matches (score < 90) — review these:')
    for _, r in low_conf.iterrows():
        print(f'  [{r["match_score"]:.0f}] "{r["zasa_title"]}" → "{r["title"]}"')

# Sample of unmatched
if unmatched:
    print(f'\nSample unmatched (first 10):')
    for u in unmatched[:10]:
        print(f'  lv{u["level"]} {u["diftype"]}: "{u["zasa_title"]}" — {u["reason"]}')
