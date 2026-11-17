"""Compact cross-cell overview of a benchmark results folder: medians, paired win rates and optimism (R1 tables).

Usage: `python benchmarks/overview.py benchmarks/results/<run_id>/results.jsonl`. Prints markdown: per problem x noise
ratio the median D9-pick regret with protocol on / off (paired win rate of on), the TPE best-ever regret, the D14
correction win rate and TPE vs random; then mean win rates across cells; then the optimism of the picks. Problems
without a known optimum use the best true value seen in any run as reference.
"""

import json, collections, statistics as st, math, sys
R=collections.defaultdict(dict)
for l in open(sys.argv[1]):
    x=json.loads(l)
    if x.get("error"): continue
    R[(x["problem"],x["noise_ratio"],x["arm"])][x["seed"]]=x
# problems without a known optimum (sklearn_svm): regret against the best true value seen in any run
best_seen=collections.defaultdict(lambda: -math.inf)
for d in R.values():
    for x in d.values(): best_seen[x["problem"]]=max(best_seen[x["problem"]], x["ever_true"])
for d in R.values():
    for x in d.values():
        if x["d9_regret"] is None:
            ref=best_seen[x["problem"]]
            x["d9_regret"], x["bs_regret"], x["ever_regret"] = ref-x["d9_true"], ref-x["bs_true"], ref-x["ever_true"]
cells=sorted({(p,n) for p,n,a in R}, key=lambda c:(c[0], c[1] or 0))
def win(a,b): return sum((x<y)+0.5*(x==y) for x,y in zip(a,b))/len(a)
agg=collections.defaultdict(list)
print("| problem | noise | random on / off | lhs on / off | tpe on / off | gp on / off | tpe best-ever | D14 tpe on vs raw | tpe vs random |")
print("|---|---|---|---|---|---|---|---|---|")
for p,n in cells:
    def med(arm,key="d9_regret"):
        d=R.get((p,n,arm)); return st.median(v[key] for v in d.values()) if d else float('nan')
    def w(a,b):
        A=R.get((p,n,a)); B=R.get((p,n,b))
        if not A or not B: return float('nan')
        s=sorted(set(A)&set(B)); return win([A[i]["d9_regret"] for i in s],[B[i]["d9_regret"] for i in s])
    row=[]
    for s in ("random","lhs","tpe","gp"):
        on,off,ww=med(f"{s}-on"),med(f"{s}-off"),w(f"{s}-on",f"{s}-off")
        if not math.isnan(ww): agg[f"protocol on vs off: {s}"].append(ww)
        row.append(f"{on:.3g} / {off:.3g} ({ww:.0%})" if not math.isnan(on) else "-")
    d14=w("tpe-on","tpe-raw"); agg["D14 correction (tpe)"].append(d14)
    agg["D14 correction (random)"].append(w("random-on","random-raw"))
    tv=w("tpe-on","random-on"); agg["tpe vs random (protocol on)"].append(tv)
    agg["lhs vs random (protocol on)"].append(w("lhs-on","random-on"))
    print(f"| {p} | {n if n is not None else 'native'} | "+" | ".join(row)+f" | {med('tpe-on','ever_regret'):.3g} | {d14:.0%} | {tv:.0%} |")
print()
print("| comparison (D9-pick regret, paired by seed) | mean win rate | cells > 50 % | cells < 50 % |")
print("|---|---|---|---|")
for k,v in agg.items():
    v=[x for x in v if not math.isnan(x)]
    print(f"| {k} | {100*st.mean(v):.0f} % | {sum(x>0.5 for x in v)} / {len(v)} | {sum(x<0.5 for x in v)} / {len(v)} |")
print()
print("| problem | noise | optimism of pick, tpe-on D9 | optimism, tpe-off best single |")
print("|---|---|---|---|")
for p,n in cells:
    A=R.get((p,n,"tpe-on")); B=R.get((p,n,"tpe-off"))
    if A and B: print(f"| {p} | {n if n is not None else 'native'} | {st.median(v['d9_optimism'] for v in A.values()):.3g} | {st.median(v['bs_optimism'] for v in B.values()):.3g} |")
