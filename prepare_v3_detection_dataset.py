from pathlib import Path
from collections import Counter
import shutil

ROOT = Path(r'C:\Users\bdhul\Documents\Fruit_Grading_Project')
SRC = ROOT / 'Dataset'
OUT = ROOT / 'Dataset_V3_Detection3Class'
SPLITS = ('train', 'valid', 'test')
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
# Original 32 classes: Apple 0-3, Orange 20-23, Pear 28-31.
TO_FRUIT = {**{i: 0 for i in range(0, 4)}, **{i: 1 for i in range(20, 24)}, **{i: 2 for i in range(28, 32)}}
NAMES = {0: 'Apple', 1: 'Orange', 2: 'Pear'}

def parse_line(line):
    p = line.split()
    if len(p) < 5:
        raise ValueError('fewer than 5 fields')
    cf = float(p[0]); cid = int(cf)
    if cf != cid:
        raise ValueError('class ID is not an integer')
    c = [float(x) for x in p[1:]]
    if any(v < 0 or v > 1 for v in c):
        raise ValueError('coordinate outside [0,1]')
    if len(c) == 4:
        x, y, w, h = c
        if w <= 0 or h <= 0:
            raise ValueError('box width/height must be positive')
        return cid, (x, y, w, h), False
    if len(c) < 6 or len(c) % 2:
        raise ValueError('polygon needs 3+ x/y pairs')
    xs, ys = c[0::2], c[1::2]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    if w <= 0 or h <= 0:
        raise ValueError('polygon has zero area')
    return cid, ((xmin+xmax)/2, (ymin+ymax)/2, w, h), True

def main():
    if not SRC.exists():
        raise SystemExit(f'Source dataset not found: {SRC}')
    if OUT.exists():
        if input(f'{OUT} exists. Type OVERWRITE to rebuild only this output folder: ').strip() != 'OVERWRITE':
            print('Cancelled; no changes made.'); return
        shutil.rmtree(OUT)
    stats = Counter(); counts = {s: Counter() for s in SPLITS}; issues = []
    for split in SPLITS:
        imgdir, labdir = SRC/split/'images', SRC/split/'labels'
        oi, ol = OUT/split/'images', OUT/split/'labels'
        oi.mkdir(parents=True, exist_ok=True); ol.mkdir(parents=True, exist_ok=True)
        if not imgdir.exists():
            issues.append(f'{split}: missing image directory {imgdir}'); continue
        images = sorted(p for p in imgdir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        stems = {p.stem for p in images}
        for im in images:
            shutil.copy2(im, oi/im.name); stats['images_copied'] += 1
            lab = labdir/f'{im.stem}.txt'; output = []
            if not lab.exists():
                stats['missing_label_files'] += 1
                issues.append(f'{split}/{im.name}: no matching label file; output label left empty')
            else:
                for n, line in enumerate(lab.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                    if not line.strip(): continue
                    try: cid, (x,y,w,h), poly = parse_line(line)
                    except Exception as e:
                        stats['invalid_lines'] += 1; issues.append(f'{split}/{lab.name}:{n}: {e}: {line[:120]}'); continue
                    if cid not in TO_FRUIT:
                        stats['unsupported_instances_skipped'] += 1; continue
                    new_id = TO_FRUIT[cid]
                    output.append(f'{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
                    counts[split][new_id] += 1; stats['boxes_written'] += 1
                    stats['polygons_converted' if poly else 'boxes_preserved'] += 1
            (ol/f'{im.stem}.txt').write_text(('\n'.join(output)+'\n') if output else '', encoding='utf-8')
            if not output: stats['empty_output_labels'] += 1
        if labdir.exists():
            for lab in labdir.glob('*.txt'):
                if lab.stem not in stems:
                    stats['orphan_label_files'] += 1; issues.append(f'{split}: label has no matching image: {lab.name}')
        else: issues.append(f'{split}: missing label directory {labdir}')
    yaml_text = f'''path: {str(OUT).replace(chr(92), '/')}
train: train/images
val: valid/images
test: test/images
nc: 3
names:
  0: Apple
  1: Orange
  2: Pear
'''
    (OUT/'data.yaml').write_text(yaml_text, encoding='utf-8')
    report = ['V3 DETECTION DATASET REPORT', f'Source untouched: {SRC}', f'Output: {OUT}', '',
              'Mapping: original IDs 0-3 -> Apple; 20-23 -> Orange; 28-31 -> Pear.',
              'All ripeness/rotten states of these fruits are retained only as fruit-location boxes.',
              'Other fruit types are skipped. Original train/valid/test split is preserved.', '']
    for s in SPLITS:
        report += [f'{s}:']
        for cid, name in NAMES.items(): report.append(f'  {name} boxes: {counts[s][cid]}')
    report += ['', 'Overall:'] + [f'  {k}: {v}' for k,v in stats.items()] + ['', 'Issues to review:']
    report += [f'  - {x}' for x in issues] if issues else ['  None reported.']
    (OUT/'preparation_report.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('\n'.join(report)); print(f'\nCreated: {OUT/"data.yaml"}'); print(f'Report: {OUT/"preparation_report.txt"}')
    print('\nReview invalid lines and counts before training. Source Dataset was not modified.')

if __name__ == '__main__': main()
