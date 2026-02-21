import csv

with open('data/manifest.csv','r',encoding='utf-8') as f:
    reader=csv.DictReader(f)
    print('fieldnames:', reader.fieldnames)
    try:
        r=next(reader)
        print('sample row keys:', list(r.keys()))
        print('sample row:', r)
    except StopIteration:
        print('manifest is empty')

