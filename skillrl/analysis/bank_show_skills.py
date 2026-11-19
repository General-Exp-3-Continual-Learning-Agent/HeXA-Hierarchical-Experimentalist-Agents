import json,sys,os,textwrap
d=sys.argv[1]
for spec in sys.argv[2:]:
    n,ids=spec.split(':'); n=int(n)
    b={x['skill_id']:x for x in json.load(open(os.path.join(d,f'skill_bank_evolving_{n}.json')))['level_skills']['pass_the_parcel']}
    for i in ids.split(','):
        if i not in b: print(f'  [r{n} {i}] ABSENT\n'); continue
        x=b[i]; print(f'=== r{n} {i} :: {x["title"]}')
        for f in ('principle','when_to_apply','example'):
            if x.get(f): print(textwrap.fill(f'  {f}: {x[f]}',128))
        print()
