"""Plan or apply the user-requested MedRAG consolidation.

Default is read-only with a local CSV plan. Run --apply only in a session
with write access to BOTH named project roots. No files are overwritten.
Duplicates are removed only after verifying an identical destination hash.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import shutil

PARENT = Path(r'D:\Code_Reserach')
CURRENT = PARENT / '2026-2027-code-MDPI-rag-Med-question-answering'
TARGET = PARENT / '2026-2027-hold-submit-Electronics-Med-RAG'
CORE = {'01_data', '02_code', '03_result', 'MDPI_template_ACS',
        'run_web.bat', 'run_web.vbs', 'stop_web.vbs'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def plan():
    virtual = {}
    rows = []
    for source, label in [(TARGET/'back-up', 'legacy-back-up'), (CURRENT, 'previous-project')]:
        if not source.exists():
            continue
        for path in sorted(source.rglob('*')):
            if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
                raise RuntimeError(f'Refusing linked filesystem entry: {path}')
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            if rel.parts[0] in CORE:
                dest = TARGET/rel
            elif source == CURRENT and (rel.parts[0] == 'audit_medrag' or path.name in {'audit_medrag_alignment.py','consolidate_medrag.py'}):
                dest = TARGET/'04_audit'/rel
            else:
                dest = TARGET/'archive'/label/rel
            checksum = digest(path)
            if dest not in virtual and dest.exists():
                virtual[dest] = digest(dest)
            if dest in virtual and virtual[dest] != checksum:
                dest = TARGET/'archive'/label/rel
                if dest not in virtual and dest.exists():
                    virtual[dest] = digest(dest)
                if dest in virtual and virtual[dest] != checksum:
                    dest = dest.with_name(dest.stem+'.'+checksum[:12]+dest.suffix)
                    if dest.exists() and digest(dest) != checksum:
                        raise RuntimeError(f'Archive collision: {dest}')
            action = 'deduplicate' if dest in virtual else 'move'
            virtual[dest] = checksum
            assert inside(dest,TARGET) and inside(path,source)
            rows.append({'source':str(path),'destination':str(dest),'action':action,
                         'bytes':path.stat().st_size,'sha256':checksum})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    rows = plan()
    # In planning mode all writes remain inside the currently writable project.
    out = (TARGET/'04_audit') if args.apply else (CURRENT/'audit_medrag')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'consolidation-manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=['source','destination','action','bytes','sha256'])
        writer.writeheader(); writer.writerows(rows)
    print(f'{len(rows)} files; {sum(r["action"]=="move" for r in rows)} moves; '
          f'{sum(r["action"]=="deduplicate" for r in rows)} verified duplicate candidates')
    if not args.apply:
        print('PLAN ONLY. No source or destination files changed.')
        return
    for row in rows:
        src,dst = Path(row['source']),Path(row['destination'])
        if not inside(src,CURRENT) and not inside(src,TARGET/'back-up'):
            raise RuntimeError(f'Unexpected source: {src}')
        if not inside(dst,TARGET) or digest(src)!=row['sha256']:
            raise RuntimeError(f'Path or source content changed: {src}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if digest(dst)!=row['sha256']:
                raise RuntimeError(f'Destination changed, refusing overwrite: {dst}')
            src.unlink()
        else:
            shutil.move(str(src),str(dst))
            if digest(dst)!=row['sha256']:
                raise RuntimeError(f'Moved file hash mismatch: {dst}')
    # Recheck every mapped destination before removing any empty directories.
    for row in rows:
        if digest(Path(row['destination']))!=row['sha256']:
            raise RuntimeError('Final verification failed: '+row['destination'])
    for root in [TARGET/'back-up',CURRENT]:
        if not root.exists():
            continue
        for directory in sorted((p for p in root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            if not inside(directory,root):
                raise RuntimeError(f'Unexpected directory: {directory}')
            if not any(directory.iterdir()):
                directory.rmdir()
        if not any(root.iterdir()):
            root.rmdir()
    (TARGET/'README_START_HERE.md').write_text(
        '# MedRAG consolidated research project\n\n'
        '- Manuscript: `MDPI_template_ACS/Med-rag-highlight.tex`\n'
        '- Research code and frozen artifacts: `02_code/`\n'
        '- Source data: `01_data/`\n'
        '- Experiment results: `03_result/`\n'
        '- Audit and SHA-256 migration manifest: `04_audit/`\n'
        '- Historical files and conflicting older versions: `archive/`\n\n'
        'Use the revision code promoted from back-up. Consult the audit before '
        'using the older manuscript latency table. Historical path references '
        'inside logs and alignment documentation may still name the previous layout.\n',encoding='utf-8')
    print('All mapped files verified. Consolidation complete.')


if __name__ == '__main__':
    main()
