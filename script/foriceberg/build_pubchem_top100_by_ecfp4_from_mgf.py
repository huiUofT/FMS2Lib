#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mgf", type=Path, required=True)
    p.add_argument("--pubchem-hdf5", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--max-topk", type=int, default=100)
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--chunk-size", type=int, default=200, help="Blocks per worker task when --workers > 1")
    return p.parse_args()

def canon(s:str)->str:
    try:
        m=Chem.MolFromSmiles(s); return Chem.MolToSmiles(m, canonical=True) if m else ""
    except Exception: return ""

def ecfp4_fp(s:str):
    m=Chem.MolFromSmiles(s); return AllChem.GetMorganFingerprintAsBitVect(m,2,2048) if m else None

_EL_RE=re.compile(r"([A-Z][a-z]?)(\d*)")
def normalize_formula(formula:str)->str:
    if not isinstance(formula,str): return ""
    s=formula.strip()
    if not s: return ""
    charge=""
    m=re.search(r"([+-]\d*|(\d+[+-]))$", s)
    if m: charge=m.group(0); s=s[:-len(charge)].strip()
    parts=_EL_RE.findall(s)
    if not parts: return formula.strip().replace(" ","")
    d:Dict[str,int]={}
    for el,cnt in parts: d[el]=d.get(el,0)+(int(cnt) if cnt else 1)
    out=[]
    for k in ("C","H"):
        if k in d:
            c=d.pop(k); out.append(f"{k}{'' if c==1 else c}")
    for el in sorted(d): 
        c=d[el]; out.append(f"{el}{'' if c==1 else c}")
    core="".join(out)
    return core+charge if charge else core

def parse_mgf_blocks(mgf:Path)->List[Dict[str,List[str]]]:
    blks=[]; cur={}
    with mgf.open("r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line=line.rstrip("\n")
            if line=="BEGIN IONS": cur={"header":[],"peaks":[]}
            elif line=="END IONS":
                if cur: blks.append(cur)
                cur={}
            else:
                if not cur: continue
                if "=" in line and not line[:1].isdigit(): cur["header"].append(line)
                else: cur["peaks"].append(line)
    return blks

def extract(header:List[str], key:str)->str:
    p=key+"="; 
    for h in header:
        if h.startswith(p): return h[len(p):].strip()
    return ""

def first_nonempty(header:List[str], keys:List[str])->str:
    for k in keys:
        v=extract(header, k)
        if v:
            return v
    return ""

def _split_candidates(payload)->List[str]:
    def coerce(s:str)->List[str]:
        s=s.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"') and s.endswith('"')):
            try:
                val=json.loads(s)
                if isinstance(val,str): return [val]
                if isinstance(val,list):
                    flat=[]
                    for it in val:
                        if isinstance(it,list): flat.extend([str(x) for x in it])
                        else: flat.append(str(it))
                    return flat
            except Exception: pass
        if "|" in s: return [t for t in s.split("|") if t]
        if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
        return [s] if s else []
    if hasattr(payload,"__len__") and not isinstance(payload,(bytes,bytearray,str)):
        out=[]; 
        for it in payload:
            s=it.decode() if isinstance(it,(bytes,bytearray)) else str(it)
            out.extend(coerce(s))
        return out
    else:
        s=payload.decode() if isinstance(payload,(bytes,bytearray)) else str(payload)
        return coerce(s)

def _collect_smiles_recursive(obj)->List[str]:
    out=[]
    def add(ds):
        try: data=ds[()]
        except Exception: return
        out.extend(_split_candidates(data))
    def walk(o):
        if isinstance(o,h5py.Dataset): add(o); return
        if isinstance(o,h5py.Group):
            for k,v in o.items():
                name=str(k).lower()
                if isinstance(v,h5py.Dataset) and name in ("smiles","smiles_list","smiles_str"):
                    add(v)
                else:
                    walk(v)
    walk(obj)
    inchi_re=re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
    seen=set(); ret=[]
    for s in out:
        ss=s.strip()
        if ss and not inchi_re.match(ss) and ss not in seen:
            seen.add(ss); ret.append(ss)
    return ret

def read_pubchem_smiles_for_formula(h5:h5py.File, norm_f:str, raw_f:str)->List[str]:
    if norm_f and norm_f in h5: return _collect_smiles_recursive(h5[norm_f])
    if raw_f and raw_f in h5: return _collect_smiles_recursive(h5[raw_f])
    return []

def build_rows_for_block(h5:h5py.File, b:Dict[str, List[str]], max_topk:int)->List[Dict[str, object]]:
    ident=first_nonempty(b["header"], ["TITLE","IDENTIFIER","FEATURE_ID"])
    t=extract(b["header"],"SMILES")
    f_raw=extract(b["header"],"FORMULA")
    if not (ident and t):
        return []
    if not f_raw:
        m=Chem.MolFromSmiles(t)
        f_raw=CalcMolFormula(m) if m else ""
    if not f_raw:
        return []
    f_norm=normalize_formula(f_raw)
    cands=read_pubchem_smiles_for_formula(h5, f_norm, f_raw)
    if not cands:
        return []
    tfp=ecfp4_fp(t)
    if tfp is None:
        return []
    scored=[]
    for s in cands:
        fpi=ecfp4_fp(s)
        if fpi is None:
            continue
        scored.append((DataStructs.TanimotoSimilarity(tfp,fpi), s))
    if not scored:
        return []
    scored.sort(key=lambda x:(-x[0], x[1]))
    picked=[]; seen_can=set()
    for sim,s in scored:
        sc=canon(s)
        if not sc or sc in seen_can:
            continue
        picked.append((sim,s)); seen_can.add(sc)
        if len(picked)>=max_topk:
            break
    merged=[(1.0, t)] + picked
    dedup=[]; seen=set()
    for sim,s in merged:
        sc=canon(s)
        if not sc or sc in seen:
            continue
        seen.add(sc)
        dedup.append((sim,s))
    out=[]; rank=1
    for sim,s in dedup:
        out.append({"identifier":ident,"true_formula":(f_norm or f_raw),"true_smiles":t,"candidate_smiles":s,"tanimoto":sim,"rank":rank})
        rank+=1
    return out

def process_chunk(chunk:List[Dict[str, List[str]]], h5_path:str, max_topk:int)->List[Dict[str, object]]:
    rows=[]
    with h5py.File(h5_path,"r") as h5:
        for b in chunk:
            rows.extend(build_rows_for_block(h5, b, max_topk))
    return rows

def main():
    args=parse_args()
    blocks=parse_mgf_blocks(args.mgf)
    rows=[]
    if args.workers <= 1:
        with h5py.File(args.pubchem_hdf5,"r") as h5:
            for b in tqdm(blocks, desc="Processing spectra", unit="spec"):
                rows.extend(build_rows_for_block(h5, b, args.max_topk))
    else:
        chunk_size=max(1, int(args.chunk_size))
        chunks=[blocks[i:i+chunk_size] for i in range(0, len(blocks), chunk_size)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs=[ex.submit(process_chunk, ch, str(args.pubchem_hdf5), args.max_topk) for ch in chunks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing chunks", unit="chunk"):
                rows.extend(fut.result())
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

if __name__=='__main__': main()
