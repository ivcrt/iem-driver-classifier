
"""
ty claude for this part Im not at that level of data science knowledge, I just knew I wanted to join those tables


- Features : Frequency Response Curves
- Labels   : Driver Config Linked to Crinacle's list
Joining the two tables on the IEM's name 
"""
import json, re, csv
from collections import Counter

CURVES = "data/curves.json"
RANKING = "data/raw.txt"
OUT = "outputs/iem_driver_dataset.csv"

#normalising names 
def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

# parsing crinacle's list
def driver_class(cfg):
    c = cfg.lower()
    has_planar = 'planar' in c
    has_est    = 'est' in c            # electrostatic
    has_bc     = 'bc' in c             # bone conduction
    has_ba     = bool(re.search(r'\bba\b|\d+\s*ba|/\s*\d*\s*ba|ba', c)) or 'ba' in c
    has_dd     = 'dd' in c
    if has_planar: return 'Planar'
    if has_est:    return 'Tribrid/EST'
    if has_bc:     return 'Other'      # rare (bone conduction)
    if has_dd and has_ba: return 'Hybrid'
    if has_ba and not has_dd: return 'BA'
    if has_dd and not has_ba: return 'DD'
    return 'Other'

crin = {}
with open(RANKING, encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 11:     
            continue
        model = parts[2].strip()
        cfg   = parts[8].strip()
        if not model or not cfg:
            continue
        crin[norm(model)] = (model, cfg, driver_class(cfg))

print(f"Liste crinacle : {len(crin)} modeles avec config driver")
print("Distribution des classes (cote labels) :", Counter(v[2] for v in crin.values()))

# load curves (excluding crinacle because we dont have his measurements)
with open(CURVES, encoding='utf-8') as f:
    data = json.load(f)
freqs = data['meta']['frequencies']
curves = data['curves']
EXCLUDE_SRC = {'crinacle', 'crinacle5128'}

rows = []
matched_models = set()
for key, val in curves.items():
    if '::' not in key:
        continue
    src, model = key.split('::', 1)
    if src in EXCLUDE_SRC:
        continue
    d = val.get('d')
    if not d or len(d) != len(freqs):
        continue
    hit = crin.get(norm(model))
    if hit is None:
        continue
    crin_model, cfg, cls = hit
    matched_models.add(norm(model))
    rows.append({'model': model, 'source': src, 'driver_raw': cfg,
                 'driver_class': cls, 'fr': d})

print(f"\nCourbes (hors crinacle) appariees : {len(rows)}")
print(f"Modeles uniques labellises        : {len(matched_models)}")
print("Distribution des classes (dataset final, par courbe) :",
      Counter(r['driver_class'] for r in rows))

# write the csv
fcols = [f"f_{round(x)}" for x in freqs]
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'source', 'driver_class', 'driver_raw'] + fcols)
    for r in rows:
        w.writerow([r['model'], r['source'], r['driver_class'], r['driver_raw']] + r['fr'])

print(f"\nEcrit -> {OUT}")
print(f"Colonnes : 4 meta + {len(fcols)} points FR")