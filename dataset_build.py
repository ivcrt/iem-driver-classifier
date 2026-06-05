
"""
ty claude for this part Im not at that level of data science knowledge, I just knew I wanted to join those tables


- Features : Frequency Response Curves
- Labels   : Driver Config Linked to Crinacle's list
Joining the two tables on the IEM's name then fuzzy with threshold
"""
import json, re, csv
from collections import Counter
from rapidfuzz import process, fuzz
 
CURVES  = "data/curves.json"
RANKING = "data/raw.txt"
OUT     = "outputs/iem_driver_dataset.csv"
THRESHOLD = 92  # fuzzy similarity threshold
 
def norm(s):
    s = s.lower()
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'\b[lr]\b', ' ', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
 
def digits(s):
    return set(re.findall(r'\d+', s))
 
def driver_class(cfg):
    c = cfg.lower()
    if 'planar' in c: return 'Planar'
    if 'est' in c:    return 'Tribrid/EST'
    if 'bc' in c:     return 'Other'
    has_ba, has_dd = 'ba' in c, 'dd' in c
    if has_dd and has_ba: return 'Hybrid'
    if has_ba: return 'BA'
    if has_dd: return 'DD'
    return 'Other'
 
crin = {}
for line in open(RANKING, encoding='utf-8', errors='replace'):
    p = line.rstrip('\n').split('\t')
    if len(p) < 11: continue
    model, cfg = p[2].strip(), p[8].strip()
    if not model or not cfg: continue
    crin[norm(model)] = (model, cfg, driver_class(cfg))
crin_keys = list(crin.keys())
print(f"crinacle List: {len(crin)} models")
 
data = json.load(open(CURVES, encoding='utf-8'))
freqs = data['meta']['frequencies']
curves = data['curves']
EXCLUDE = {'crinacle', 'crinacle5128'}
 
rows = []
exact_hits = fuzzy_hits = 0
matched_models = set()
 
for key, val in curves.items():
    if '::' not in key: continue
    src, model = key.split('::', 1)
    if src in EXCLUDE: continue
    d = val.get('d')
    if not d or len(d) != len(freqs): continue
    nm = norm(model)
    hit = crin.get(nm)
    if hit is not None:
        exact_hits += 1
    else:
        best = process.extractOne(nm, crin_keys, scorer=fuzz.token_set_ratio)
        if best and best[1] >= THRESHOLD and digits(nm) == digits(best[0]):
            hit = crin[best[0]]; fuzzy_hits += 1
        else:
            continue
    canon, cfg, cls = hit
    matched_models.add(canon)
    rows.append({'model': canon, 'source': src, 'driver_raw': cfg,
                 'driver_class': cls, 'fr': d})
 
print(f"Exact matchs : {exact_hits} | fuzzy : {fuzzy_hits} | total curves : {len(rows)}")
print(f"Unique models that have labels : {len(matched_models)}")
print("Distribution classes :", Counter(r['driver_class'] for r in rows))
 
fcols = [f"f_{round(x)}" for x in freqs]
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'source', 'driver_class', 'driver_raw'] + fcols)
    for r in rows:
        w.writerow([r['model'], r['source'], r['driver_class'], r['driver_raw']] + r['fr'])
print(f"\nWrites -> {OUT}  ({len(fcols)}  FR points)")