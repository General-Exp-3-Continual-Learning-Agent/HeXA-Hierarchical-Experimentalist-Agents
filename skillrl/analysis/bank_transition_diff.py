import json,sys,re,glob,os
d=sys.argv[1]; LEV=sys.argv[2] if len(sys.argv)>2 else 'pass_the_parcel'
ns=sorted(int(re.search(r'_(\d+)\.json$',p).group(1)) for p in glob.glob(os.path.join(d,'skill_bank_evolving_*.json')))
def get(n):
    b=json.load(open(os.path.join(d,f'skill_bank_evolving_{n}.json')))
    return {x['skill_id']:x for x in b['level_skills'].get(LEV,[])}
def txt(x): return x['title']+'|'+x.get('principle','')+'|'+x.get('when_to_apply','')+'|'+x.get('example','')
print('ROUNDS',ns)
for n in ns: 
    b=get(n); print(f'--- r{n} n={len(b)}')
    for i,x in b.items(): print(f'    {i:<16} g{x["generation"]} c{x["confidence"]:.2f} {x["title"]}')
print('\n======== TRANSITIONS')
for a_,b_ in zip(ns,ns[1:]):
    a,b=get(a_),get(b_)
    gone=[i for i in a if i not in b]; new=[i for i in b if i not in a]
    chg=[i for i in a if i in b and txt(a[i])!=txt(b[i])]
    if not(gone or new): print(f'r{a_}->r{b_}: stable, {len(chg)} edited'); continue
    print(f'r{a_}->r{b_}:')
    for i in gone: print(f'    GONE {i}  {a[i]["title"][:72]}')
    for i in new:  print(f'    NEW  {i}  {b[i]["title"][:72]}')
    print(f'    edited: {", ".join(chg)}')
