#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem
import re


_SCRIPT_DIR = Path(__file__).resolve().parent
_GITHUB_DIR = _SCRIPT_DIR.parent
_DEFAULT_GEN_CKPT = _GITHUB_DIR / "mode" / "generation" / "best.ckpt"
_DEFAULT_INTEN_CKPT = _GITHUB_DIR / "mode" / "intensity" / "best.ckpt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="One-pass pipeline: from MGF + susdat_F.csv -> candidates -> predict -> cosine -> standardized SMILES evaluation."
	)
	parser.add_argument("--mgf", type=Path, required=True, help="Input MGF with TITLE= and FORMULA= per spectrum")
	parser.add_argument(
		"--susdat",
		type=Path,
		required=True,
		help="SusDat CSV with columns formula, Name, SMILES (or Molecular_Formula, Name, SMILES for Norman export)",
	)
	parser.add_argument(
		"--gen-checkpoint",
		type=Path,
		default=_DEFAULT_GEN_CKPT,
		help=f"GEN checkpoint (default: {_DEFAULT_GEN_CKPT})",
	)
	parser.add_argument(
		"--inten-checkpoint",
		type=Path,
		default=_DEFAULT_INTEN_CKPT,
		help=f"INTEN checkpoint (default: {_DEFAULT_INTEN_CKPT})",
	)
	parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
	parser.add_argument(
		"--predict-script",
		type=Path,
		default=_GITHUB_DIR / "src" / "ms_pred" / "dag_pred" / "predict_eval_end2end_pubchem_candidates_cosine.py",
		help="End2end prediction + cosine driver (under github/src/ms_pred/dag_pred/)",
	)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--sparse-k", type=int, default=100)
	parser.add_argument("--max-nodes", type=int, default=5000)
	parser.add_argument("--threshold", type=float, default=0.05)
	return parser.parse_args()


def parse_mgf_blocks(mgf_path: Path) -> List[Dict[str, List[str]]]:
	blocks: List[Dict[str, List[str]]] = []
	curr: Dict[str, List[str]] = {}
	with mgf_path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			line = line.rstrip("\n")
			if line == "BEGIN IONS":
				curr = {"header": [], "peaks": []}
			elif line == "END IONS":
				blocks.append(curr)
				curr = {}
			else:
				if not curr:
					continue
				if "=" in line:
					curr.setdefault("header", []).append(line)
				else:
					if line and line[0].isdigit():
						curr.setdefault("peaks", []).append(line)
	return blocks


def extract_field(header_lines: List[str], key: str) -> str:
	prefix = key + "="
	for h in header_lines:
		if h.startswith(prefix):
			return h[len(prefix):].strip()
	return ""


def write_mgf(blocks: List[Dict[str, List[str]]], out_path: Path) -> None:
	with out_path.open("w", encoding="utf-8") as w:
		for b in blocks:
			w.write("BEGIN IONS\n")
			for h in b.get("header", []):
				w.write(h + "\n")
			for p in b.get("peaks", []):
				w.write(p + "\n")
			w.write("END IONS\n")


def canon_smiles(s: str) -> str:
	try:
		m = Chem.MolFromSmiles(s)
		return Chem.MolToSmiles(m, canonical=True) if m else ""
	except Exception:
		return ""

_EL_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
def normalize_formula(formula: str) -> str:
	"""
	Normalize a chemical formula string to a simple canonical form:
	- Remove spaces and non-alphanumeric (+/- kept as is at the end if present)
	- Parse element symbols and integer counts
	- Sort elements in Hill system: C, then H, then alphabetical for the rest
	- Omit count '1'
	Notes:
	- This is a heuristic; complex annotations (isotopes, charges in the middle) are not fully supported.
	"""
	if not isinstance(formula, str):
		return ""
	s = formula.strip()
	if not s:
		return ""
	charge = ""
	# Extract trailing charge like +, -, 2+, 2-
	m = re.search(r"([+-]\d*|(\d+[+-]))$", s)
	if m:
		charge = m.group(0)
		s = s[: -len(charge)].strip()
	parts = _EL_RE.findall(s)
	if not parts:
		return formula.strip().replace(" ", "")
	el_to_count: Dict[str, int] = {}
	for el, cnt in parts:
		n = int(cnt) if cnt else 1
		el_to_count[el] = el_to_count.get(el, 0) + n
	ordered: List[str] = []
	def add(el: str):
		if el in el_to_count:
			c = el_to_count.pop(el)
			ordered.append(f"{el}{'' if c==1 else c}")
	# Hill: C, H, then others alpha
	add("C")
	add("H")
	for el in sorted(el_to_count.keys()):
		add(el)
	core = "".join(ordered)
	return (core + charge) if charge else core

def main() -> None:
	args = parse_args()
	for label, p in (("GEN", args.gen_checkpoint), ("INTEN", args.inten_checkpoint)):
		if not p.is_file():
			print(f"ERROR: {label} checkpoint not found: {p}", file=sys.stderr)
			sys.exit(1)
	args.out_dir.mkdir(parents=True, exist_ok=True)

	# Load susdat and build formula->candidates, name->true_smiles
	df = pd.read_csv(args.susdat, low_memory=False)
	if "formula" not in df.columns and "Molecular_Formula" in df.columns:
		df = df.rename(columns={"Molecular_Formula": "formula"})
	required = {"formula", "Name", "SMILES"}
	miss = required - set(df.columns)
	if miss:
		raise ValueError(f"susdat missing columns: {miss}")
	formula_to_cands: Dict[str, List[str]] = {}
	for rec in df.itertuples(index=False):
		form = normalize_formula(str(rec.formula))
		smi = str(rec.SMILES)
		if form and smi:
			formula_to_cands.setdefault(form, [])
			if smi not in formula_to_cands[form]:
				formula_to_cands[form].append(smi)
	name_to_true: Dict[str, str] = {}
	name_to_true_ci: Dict[str, str] = {}
	for rec in df.itertuples(index=False):
		name = str(rec.Name)
		smi = str(rec.SMILES)
		if name and smi and name not in name_to_true:
			name_to_true[name] = smi
		if name and smi:
			key = name.strip().casefold()
			if key not in name_to_true_ci:
				name_to_true_ci[key] = smi

	# Parse MGF and assemble compat CSV + filtered MGF
	blocks = parse_mgf_blocks(args.mgf)
	kept_blocks: List[Dict[str, List[str]]] = []
	compat_rows: List[Dict[str, str]] = []
	stats = {"total_blocks": 0, "kept_with_candidates": 0, "skipped_no_formula": 0, "skipped_no_candidates": 0}
	for b in blocks:
		stats["total_blocks"] += 1
		title = extract_field(b.get("header", []), "TITLE")
		formula_raw = extract_field(b.get("header", []), "FORMULA")
		formula = normalize_formula(formula_raw) if formula_raw else ""
		if not formula:
			stats["skipped_no_formula"] += 1
			continue
		cands = formula_to_cands.get(formula, [])
		if not cands:
			stats["skipped_no_candidates"] += 1
			continue
		# Prefer SMILES from MGF header if present; else fallback to susdat Name mapping (exact, then case-insensitive)
		true_smi = extract_field(b.get("header", []), "SMILES")
		if not true_smi:
			true_smi = name_to_true.get(title, "")
		if not true_smi and title:
			true_smi = name_to_true_ci.get(title.strip().casefold(), "")
		# Insert true SMILES (if present) first, then formula-based candidates; deduplicate while preserving order
		merged_cands = []
		seen = set()
		for sm in ([true_smi] if true_smi else []) + list(cands):
			if sm and sm not in seen:
				seen.add(sm)
				merged_cands.append(sm)
		row = {"identifier": title, "SMILES": true_smi}
		for i, s in enumerate(merged_cands, start=1):
			row[f"pubchem_smiles_{i}"] = s
		compat_rows.append(row)
		kept_blocks.append(b)
	if not compat_rows:
		( args.out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
		print("No entries with candidates. Stats written.")
		return
	compat_csv = args.out_dir / "candidates_compat_by_formula_from_mgf.csv"
	pd.DataFrame(compat_rows).to_csv(compat_csv, index=False)
	mgf_filtered = args.out_dir / "targets_with_candidates.mgf"
	write_mgf(kept_blocks, mgf_filtered)
	(args.out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))

	# Also write per-entry candidate counts (without inserting true_smiles)
	compat_df = pd.DataFrame(compat_rows)
	def _count_cands(row: pd.Series) -> int:
		return sum(1 for c in row.index if isinstance(c, str) and c.startswith("pubchem_smiles_") and pd.notna(row[c]) and str(row[c]).strip())
	counts_df = compat_df[["identifier", "SMILES"]].copy()
	counts_df["num_candidates"] = compat_df.apply(_count_cands, axis=1)
	counts_df.to_csv(args.out_dir / "per_entry_candidate_counts.csv", index=False)

	# Call predictor script
	run_dir = args.out_dir / "pred_run"
	run_dir.mkdir(parents=True, exist_ok=True)
	cmd = [
		sys.executable,
		str(args.predict_script),
		"--input_csv", str(compat_csv),
		"--target_mgf", str(mgf_filtered),
		"--gen_checkpoint", str(args.gen_checkpoint),
		"--inten_checkpoint", str(args.inten_checkpoint),
		"--out_dir", str(run_dir),
		"--batch_size", str(args.batch_size),
		"--num_workers", str(args.num_workers),
		"--sparse_k", str(args.sparse_k),
		"--max_nodes", str(args.max_nodes),
		"--threshold", str(args.threshold),
	]
	subprocess.run(cmd, check=True)

	# Evaluation with standardized SMILES and coverage
	top1_csv = run_dir / "end2end_pubchem_candidates_top1.csv"
	if not top1_csv.exists():
		print("Prediction did not produce top1 CSV. Exiting.", file=sys.stderr)
		return
	top1 = pd.read_csv(top1_csv)
	compat = pd.read_csv(compat_csv)
	# Coverage: use the actual candidates used for prediction (labels/index), not the prebuild compat CSV
	cand_index_path = run_dir / "candidates_index.csv"
	id_to_cands: Dict[str, List[str]] = {}
	if cand_index_path.exists():
		cand_idx = pd.read_csv(cand_index_path)
		for ident, group in cand_idx.groupby("identifier", sort=False):
			id_to_cands[ident] = [str(s) for s in group["cand_smiles"].astype(str).tolist() if str(s)]
	else:
		# Fallback to compat CSV columns if index not found
		cand_cols = [c for c in compat.columns if isinstance(c, str) and c.startswith("pubchem_smiles_")]
		for r in compat.itertuples(index=False):
			id_to_cands[str(r.identifier)] = [str(getattr(r, c)) for c in cand_cols if hasattr(r, c) and pd.notna(getattr(r, c)) and str(getattr(r, c)).strip()]
	# Build coverage rows with canonical comparison
	has_true_rows = []
	for r in compat.itertuples(index=False):
		ident = str(r.identifier)
		cand_list = id_to_cands.get(ident, [])
		cand_raw_set = set(map(str, cand_list))
		cand_canon_set = set(filter(None, (canon_smiles(str(s)) for s in cand_list)))
		true_raw = str(getattr(r, "SMILES", "")) if hasattr(r, "SMILES") else ""
		true_canon = canon_smiles(true_raw) if true_raw else ""
		has_true_raw = bool(true_raw) and (true_raw in cand_raw_set)
		has_true_canon = bool(true_canon) and (true_canon in cand_canon_set)
		has_true_rows.append({
			"identifier": ident,
			"has_true_in_candidates_raw": has_true_raw,
			"has_true_in_candidates_canonical": has_true_canon,
		})
	has_true_df = pd.DataFrame(has_true_rows)
	# Merge, standardize, compare
	df_eval = top1.merge(compat[["identifier", "SMILES"]], on="identifier", how="left").merge(has_true_df, on="identifier", how="left")
	df_eval["true_canon"] = df_eval["SMILES"].astype(str).map(canon_smiles)
	df_eval["pred_canon"] = df_eval["best_smiles"].astype(str).map(canon_smiles)
	df_eval["retrieved_correct"] = (df_eval["true_canon"] == df_eval["pred_canon"])
	total = len(df_eval)
	retrieved = int(df_eval["retrieved_correct"].sum())
	covered_canon = int(df_eval["has_true_in_candidates_canonical"].fillna(False).astype(bool).sum())
	summary = {
		"evaluated_total": total,
		"has_true_in_candidates_canonical": f"{covered_canon}/{total}",
		"has_true_in_candidates_canonical_rate": (covered_canon / total) if total else 0.0,
		"retrieved_correct_std": f"{retrieved}/{total}",
		"retrieved_correct_std_rate": (retrieved / total) if total else 0.0,
	}
	(run_dir / "retrieval_eval_standardized_summary.json").write_text(json.dumps(summary, indent=2))
	df_eval.to_csv(run_dir / "eval_with_canonical_smiles.csv", index=False)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()

