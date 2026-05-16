#!/usr/bin/env python3
"""
在测试集上评估模型性能，使用官方的Cosine Similarity计算方法
基于 retrieval_benchmark_torchmetrics.py 中的 dist_bin 函数
"""
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd
import h5py
from functools import partial
from typing import Dict, List

# 仓库根目录：本脚本位于 ``script/``，根目录为其上一级（含 ``src/``、``data/``）
from foriceberg_paths import data_spec_datasets, github_root, pythonpath_src, repo_root

project_root = repo_root()
sys.path.insert(0, pythonpath_src())

# 尝试导入tqdm，如果失败则使用简单的进度显示
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    print("⚠ 警告: tqdm未安装，将使用简单的进度显示")
    USE_TQDM = False
    def tqdm(iterable, desc=None, total=None):
        """简单的进度显示替代tqdm"""
        if desc:
            print(desc)
        if total:
            print(f"总共: {total} 项")
        count = 0
        for item in iterable:
            count += 1
            if count % 100 == 0:
                print(f"  已处理: {count}/{total if total else '?'}")
            yield item
        if desc:
            print(f"完成: {count} 项")

import ms_pred.common as common
from ms_pred.common.splitter import get_splits


def dist_bin(cand_preds_dict: List[Dict], true_spec_dict: dict, sparse=True, ignore_peak=None, func='cos', selected_evs=None, agg=True) -> np.ndarray:
    """
    官方的cosine distance计算函数（来自 retrieval_benchmark_torchmetrics.py）
    
    Args:
        cand_preds_dict: List of candidate predictions (每个元素是一个dict，key是碰撞能量，value是spectrum)
        true_spec_dict: 真实谱图字典 (key是碰撞能量，value是binned spectrum array)
        sparse: 预测是否为sparse格式
        ignore_peak: 忽略的peak index（通常是precursor peak）
        func: 距离函数类型 ('cos' 或 'entropy')
        selected_evs: 选择的碰撞能量列表
        agg: 是否聚合多个碰撞能量
        
    Returns:
        np.ndarray: cosine distances for each candidate
    """
    dist = []
    true_npeaks = []

    if selected_evs:
        true_spec_dict = {k: v for k, v in true_spec_dict.items() if str(k) in selected_evs}
    
    # 标准化碰撞能量键为字符串格式（保留0位小数，与官方代码一致）
    true_spec_dict = {f'{float(k):.0f}': v for k, v in true_spec_dict.items()}
    cand_preds_dict = [{f'{float(k):.0f}': v for k, v in cand_dict.items()} for cand_dict in cand_preds_dict]
    
    for colli_eng in true_spec_dict.keys():
        true_spec = true_spec_dict[colli_eng]
        try:
            cand_preds = np.stack([i.get(colli_eng, np.zeros_like(true_spec)) for i in cand_preds_dict], axis=0)
            stacked = True
        except ValueError:
            cand_preds = [i.get(colli_eng, np.zeros((0,2))) for i in cand_preds_dict]
            stacked = False
        
        if sparse:
            # 将sparse格式转换为dense格式
            pred_specs = np.zeros((len(cand_preds), true_spec.shape[0]))
            if stacked:
                inds = cand_preds[:, :, 0].astype(int)
                pos_1 = np.repeat(np.arange(inds.shape[0])[:, np.newaxis], inds.shape[1], axis=1)
                valid_mask = inds < pred_specs.shape[1]
                pred_specs[pos_1[valid_mask], inds[valid_mask]] = cand_preds[:, :, 1][valid_mask]
            else:
                for i, cand_pred in enumerate(cand_preds):
                    if cand_pred.ndim == 2 and cand_pred.shape[1] == 2:
                        inds = cand_pred[:, 0].astype(int)
                        valid_mask = inds < pred_specs.shape[1]
                        pred_specs[i, inds[valid_mask]] = cand_pred[valid_mask, 1]
        else:
            pred_specs = cand_preds

        if ignore_peak:
            pred_specs[:, int(ignore_peak):] = 0
            true_spec = true_spec.copy()
            true_spec[int(ignore_peak):] = 0

        true_npeaks.append(np.sum(true_spec > 0))

        if func == 'cos':
            # Cosine distance = 1 - cosine similarity
            norm_pred = norm(pred_specs, axis=-1) + 1e-22
            norm_true = norm(true_spec, axis=-1) + 1e-22
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        elif func == 'entropy':
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

            def entropy(prob):
                return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            zeros = (pred_specs.sum(axis=-1) == 0)
            entropy_pred = entropy(norm_pred)
            entropy_targ = entropy(norm_true)
            entropy_mix = entropy((norm_pred + norm_true) / 2)
            entropy_dists = (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
            entropy_dists[zeros] = 1  # mask empty spectra to be distance 1
            dist.append(entropy_dists)

    dist = np.array(dist)  # num of colli energy x number of candidates
    
    # 权重计算：如果>=5 peaks: weight=4, elif >=1 peaks: weight=1, else: weight=0
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    weights = weights / (weights.sum() + 1e-9)
    
    if agg:
        return np.sum(dist * weights[:, None], axis=0)
    else:
        # return both
        dist = dist[weights > 0]  # exclude any objectives that have zero-weights based on filter
        return dist, np.sum(dist * weights[:, None], axis=0)


# 定义cosine distance函数
cos_dist_bin = partial(dist_bin, func='cos')


def load_true_spectrum(spec_file: Path, spec_name: str, num_bins: int = 15000, upper_limit: int = 1500) -> Dict[str, np.ndarray]:
    """
    加载真实谱图并转换为binned格式
    
    真实谱图文件存储的是MGF格式的字符串，需要使用parse_spectra_mgf的逻辑来解析
    
    Args:
        spec_file: HDF5文件路径
        spec_name: 谱图名称
        num_bins: bin数量
        upper_limit: 最大m/z
        
    Returns:
        Dict[str, np.ndarray]: {碰撞能量: binned_spectrum}
    """
    # 使用HDF5Dataset来读取字符串数据
    spec_h5 = common.HDF5Dataset(spec_file)
    
    try:
        # 读取谱图字符串
        spec_str = spec_h5.read_str(spec_name)
        
        # 解析SIRIUS格式或MGF格式字符串
        spec_lines = spec_str.split('\n')
        all_spectra_tuples = []
        meta = {}
        cur_spectra = []
        current_ce = None
        
        # 检查格式类型
        is_mgf_format = "BEGIN IONS" in spec_str
        is_sirius_format = any(line.strip().startswith(">collision") for line in spec_lines)
        
        if is_mgf_format:
            # MGF格式解析
            in_ions = False
            for line in spec_lines:
                line = line.strip()
                if line == "BEGIN IONS":
                    in_ions = True
                    meta = {}  # 重置meta
                    cur_spectra = []  # 重置spectra
                    continue
                elif line == "END IONS":
                    in_ions = False
                    # 处理当前谱图
                    if len(cur_spectra) > 0:
                        spec_data = np.vstack(cur_spectra)
                        collision_energy = meta.get('COLLISION_ENERGY', meta.get('collision_energy', '30.0'))
                        all_spectra_tuples.append((collision_energy, spec_data))
                    continue
                elif not line:
                    continue
                elif "=" in line:
                    k, v = [i.strip() for i in line.split("=", 1)]
                    meta[k] = v
                elif in_ions:
                    # 这是m/z和intensity行
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intens = float(parts[1])
                            cur_spectra.append((mz, intens))
                        except ValueError:
                            continue
            
            # 如果没有找到END IONS，使用最后收集的数据
            if len(all_spectra_tuples) == 0 and len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                collision_energy = meta.get('COLLISION_ENERGY', meta.get('collision_energy', '30.0'))
                all_spectra_tuples = [(collision_energy, spec_data)]
        elif is_sirius_format:
            # SIRIUS格式解析
            for line in spec_lines:
                line = line.strip()
                if not line:
                    continue
                elif line.startswith("#"):
                    # 元数据行，如 #COLLISION ENERGY 30.0
                    if "COLLISION ENERGY" in line.upper():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.upper() == "ENERGY":
                                if i + 1 < len(parts):
                                    try:
                                        current_ce = float(parts[i + 1])
                                    except ValueError:
                                        current_ce = 30.0
                                    break
                elif line.startswith(">collision"):
                    # 碰撞能量标记行，如 >collision 30.0
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            current_ce = float(parts[1])
                        except ValueError:
                            current_ce = 30.0
                    # 如果之前有数据，先保存
                    if len(cur_spectra) > 0 and current_ce is not None:
                        spec_data = np.vstack(cur_spectra)
                        all_spectra_tuples.append((str(current_ce), spec_data))
                        cur_spectra = []
                else:
                    # m/z和intensity行
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intens = float(parts[1])
                            cur_spectra.append((mz, intens))
                        except ValueError:
                            continue
            
            # 保存最后一个谱图
            if len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                ce = str(current_ce) if current_ce is not None else '30.0'
                all_spectra_tuples.append((ce, spec_data))
            
            # 如果没有找到任何谱图，尝试从元数据中获取碰撞能量
            if len(all_spectra_tuples) == 0 and len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                ce = str(current_ce) if current_ce is not None else '30.0'
                all_spectra_tuples = [(ce, spec_data)]
        else:
            # 默认格式：假设所有行都是m/z和intensity对，使用默认碰撞能量
            for line in spec_lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(">"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intens = float(parts[1])
                        cur_spectra.append((mz, intens))
                    except ValueError:
                        continue
            
            if len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                all_spectra_tuples = [('30.0', spec_data)]
        
        spectra_tuples = all_spectra_tuples
        
        spec_h5.close()
        
        # 转换为binned格式
        true_spec_dict = {}
        for ce, spec_data in spectra_tuples:
            if len(spec_data) == 0:
                continue
            
            # 标准化碰撞能量键：转换为浮点数再转为字符串，确保格式一致
            if isinstance(ce, str):
                # 移除单位
                ce_clean = ce.replace(' eV', '').replace(' eV [imputed]', '').strip()
                try:
                    ce_float = float(ce_clean)
                except ValueError:
                    ce_float = 30.0  # 默认值
            else:
                ce_float = float(ce)
            
            # 标准化键格式：保留0位小数（与官方代码一致）
            ce_key = f'{ce_float:.0f}'
            
            binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
            true_spec_dict[ce_key] = binned
        
        return true_spec_dict
    except Exception as e:
        spec_h5.close()
        return {}


def load_predicted_spectrum(pred_file: Path, spec_name: str, num_bins: int = 15000, upper_limit: int = 1500) -> Dict[str, np.ndarray]:
    """
    加载预测谱图
    
    预测文件结构: pred_<spec_name>/ikey <inchikey>/collision <energy>/spec
    
    Args:
        pred_file: HDF5预测文件路径
        spec_name: 谱图名称
        num_bins: bin数量
        upper_limit: 最大m/z
        
    Returns:
        Dict[str, np.ndarray]: {碰撞能量: binned_spectrum}
    """
    with h5py.File(pred_file, 'r') as f:
        # 预测文件的键名格式是 pred_<spec_name>
        pred_key = f'pred_{spec_name}'
        
        if pred_key not in f:
            return {}
        
        pred_group = f[pred_key]
        pred_dict = {}
        
        # 检查是否为sparse输出（从文件属性）
        sparse_out = f.attrs.get('sparse_out', True)
        
        # 遍历所有ikey（通常只有一个）
        for ikey_key in pred_group.keys():
            ikey_group = pred_group[ikey_key]
            
            # 遍历所有碰撞能量
            for ce_key in ikey_group.keys():
                ce_group = ikey_group[ce_key]
                if 'spec' in ce_group:
                    spec_data = ce_group['spec'][:]
                    
                    # 提取碰撞能量值（从 "collision 30.0" 或 "30" 中提取）
                    ce_raw = ce_key.replace('collision ', '').strip()
                    
                    # 标准化碰撞能量键：转换为浮点数再转为字符串，确保格式一致
                    try:
                        ce_float = float(ce_raw)
                    except ValueError:
                        ce_float = 30.0  # 默认值
                    
                    # 标准化键格式：保留0位小数（与官方代码一致）
                    ce_key_normalized = f'{ce_float:.0f}'
                    
                    # 如果是sparse格式（2D数组，shape为[N, 2]），转换为binned
                    if sparse_out and len(spec_data.shape) == 2 and spec_data.shape[1] >= 2:
                        # sparse格式：[m/z_index, intensity] 或 [m/z, intensity]
                        # 需要转换为binned格式
                        binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
                        pred_dict[ce_key_normalized] = binned
                    elif len(spec_data.shape) == 1:
                        # 已经是binned格式（1D数组）
                        pred_dict[ce_key_normalized] = spec_data
                    else:
                        # 未知格式，尝试转换
                        if len(spec_data.shape) == 2:
                            binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
                            pred_dict[ce_key_normalized] = binned
                        else:
                            # 如果格式不对，创建空的binned spectrum
                            pred_dict[ce_key_normalized] = np.zeros(num_bins)
        
        return pred_dict


def main():
    parser = argparse.ArgumentParser(description='在测试集上评估模型性能')
    parser.add_argument('--gen-checkpoint', type=str, default=None, help='生成模型checkpoint路径（如果指定pred-file则可选）')
    parser.add_argument('--inten-checkpoint', type=str, default=None, help='强度模型checkpoint路径（如果指定pred-file则可选）')
    parser.add_argument('--pred-file', type=str, default=None, help='预测结果HDF5文件路径（如果指定，将使用此文件而不是在output-dir中查找）')
    parser.add_argument('--dataset-name', type=str, default='new_dataset', help='数据集名称')
    parser.add_argument('--split-name', type=str, default='split_1.tsv', help='划分文件名称')
    parser.add_argument('--output-dir', type=str, default='test/evaluation', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=0, help='工作进程数')
    parser.add_argument('--gpu', action='store_true', help='使用GPU')
    parser.add_argument('--num-bins', type=int, default=15000, help='Bin数量')
    parser.add_argument('--upper-limit', type=int, default=1500, help='最大m/z')
    parser.add_argument('--ignore-parent-peak', action='store_true', help='忽略前体离子峰')
    parser.add_argument('--test-labels', type=str, default=None, help='测试集标签文件路径（如果指定，将直接使用此文件而不是从split中获取）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.pred_file is None and (args.gen_checkpoint is None or args.inten_checkpoint is None):
        parser.error("必须指定 --pred-file 或同时指定 --gen-checkpoint 和 --inten-checkpoint")
    
    project_root = repo_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ICEBERG 测试集评估")
    print("=" * 80)
    if args.gen_checkpoint:
        print(f"生成模型: {args.gen_checkpoint}")
    if args.inten_checkpoint:
        print(f"强度模型: {args.inten_checkpoint}")
    if args.pred_file:
        print(f"预测文件: {args.pred_file}")
    print(f"数据集: {args.dataset_name}")
    if args.test_labels:
        print(f"测试集标签文件: {args.test_labels}")
    else:
        print(f"划分文件: {args.split_name}")
    print("=" * 80)
    
    # 加载数据集和划分
    dataset_path = data_spec_datasets(args.dataset_name)
    
    # 加载数据集的labels.tsv（用于获取precursor等信息）
    dataset_labels_file = dataset_path / "labels.tsv"
    dataset_labels = None
    if dataset_labels_file.exists():
        dataset_labels = pd.read_csv(dataset_labels_file, sep='\t')
        dataset_labels_dict = dict(zip(dataset_labels['spec'], dataset_labels.to_dict('records')))
    else:
        dataset_labels_dict = {}
    
    # 如果指定了测试集标签文件，直接使用
    if args.test_labels:
        test_labels_file = Path(args.test_labels)
        if not test_labels_file.exists():
            raise FileNotFoundError(f"测试集标签文件不存在: {test_labels_file}")
        test_labels = pd.read_csv(test_labels_file, sep='\t')
        print(f"\n从文件加载测试集: {len(test_labels)} 个样本")
    else:
        # 从split文件中获取测试集
        labels_file = dataset_path / "labels.tsv"
        split_file = dataset_path / "splits" / args.split_name
        
        if not labels_file.exists():
            raise FileNotFoundError(f"标签文件不存在: {labels_file}")
        if not split_file.exists():
            raise FileNotFoundError(f"划分文件不存在: {split_file}")
        
        labels_df = pd.read_csv(labels_file, sep='\t')
        # get_splits需要两个参数：names列表和split_file
        spec_names = labels_df['spec'].tolist()
        train_inds, val_inds, test_inds = get_splits(spec_names, str(split_file))
        test_indices = test_inds
        
        print(f"\n测试集大小: {len(test_indices)} 个样本")
        
        # 获取测试集的标签
        test_labels = labels_df.iloc[test_indices].copy()
        test_labels_file = output_dir / "test_labels.tsv"
    test_labels.to_csv(test_labels_file, sep='\t', index=False)
    
    # 检查是否已有预测结果，如果没有则提示用户手动运行预测
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找预测结果文件
    if args.pred_file:
        # 如果指定了pred-file，直接使用
        pred_file = Path(args.pred_file)
        if not pred_file.exists():
            raise FileNotFoundError(f"指定的预测文件不存在: {pred_file}")
        print(f"\n✓ 使用指定的预测结果文件: {pred_file}")
    else:
        # 否则在output_dir/predictions中查找
        pred_files = list(pred_dir.glob("preds*.hdf5"))
        
        if not pred_files:
            print("\n⚠ 未找到预测结果文件")
            print("=" * 80)
            print("请先手动运行预测命令生成预测结果：")
            print("=" * 80)
            print(f"python src/ms_pred/dag_pred/predict_smis.py \\")
            print(f"    --gen-checkpoint {args.gen_checkpoint} \\")
            print(f"    --inten-checkpoint {args.inten_checkpoint} \\")
            print(f"    --dataset-labels {test_labels_file} \\")
            print(f"    --batch-size {args.batch_size} \\")
            print(f"    --num-workers {args.num_workers} \\")
            print(f"    --sparse-out \\")
            print(f"    --sparse-k 100 \\")
            print(f"    --max-nodes 100 \\")
            print(f"    --threshold 0.1 \\")
            print(f"    --save-dir {pred_dir} \\")
            print(f"    --adduct-shift" + (" \\" if args.gpu else ""))
            if args.gpu:
                print(f"    --gpu")
            print("=" * 80)
            print("预测完成后，重新运行此评估脚本即可计算cosine similarity")
            return
        
        pred_file = pred_files[0]
        print(f"\n✓ 找到预测结果文件: {pred_file}")
    
    # 加载真实谱图
    spec_file = dataset_path / "spec_files.hdf5"
    if not spec_file.exists():
        print(f"❌ 真实谱图文件不存在: {spec_file}")
        return
    
    print("\n开始计算Cosine Similarity...")
    
    # 读取预测文件的元数据
    with h5py.File(pred_file, 'r') as f:
        num_bins = f.attrs.get('num_bins', args.num_bins)
        upper_limit = f.attrs.get('upper_limit', args.upper_limit)
    
    results = []
    
    # 检查是否有已保存的部分结果
    partial_results_file = output_dir / "evaluation_results_partial.tsv"
    processed_specs = set()
    if partial_results_file.exists():
        print(f"\n发现部分结果文件: {partial_results_file}")
        try:
            partial_df = pd.read_csv(partial_results_file, sep='\t')
            processed_specs = set(partial_df['spec'].tolist())
            results = partial_df.to_dict('records')
            print(f"已处理 {len(processed_specs)} 个样本，将继续处理剩余样本...")
        except Exception as e:
            print(f"读取部分结果文件失败: {e}，将从头开始")
    
    # 遍历测试集样本
    iterator = tqdm(test_labels.itertuples(), total=len(test_labels), desc="计算Cosine Similarity") if USE_TQDM else test_labels.itertuples()
    
    processed_count = 0
    for row in iterator:
        spec_name = row.spec
        
        # 跳过已处理的样本
        if spec_name in processed_specs:
            continue
        spec_name = row.spec
        
        # 获取precursor mass，优先从row获取，否则从数据集labels中获取，最后从smiles计算
        precursor_mass = None
        if hasattr(row, 'precursor') and pd.notna(row.precursor):
            precursor_mass = row.precursor
        elif spec_name in dataset_labels_dict:
            # 从数据集的labels.tsv中获取
            spec_info = dataset_labels_dict[spec_name]
            if 'precursor' in spec_info and pd.notna(spec_info['precursor']):
                precursor_mass = spec_info['precursor']
        
        # 如果还是没有，从smiles计算
        if precursor_mass is None:
            try:
                from ms_pred.common import chem_utils
                smiles = None
                adduct = None
                
                # 获取smiles
                if hasattr(row, 'smiles') and pd.notna(row.smiles):
                    smiles = row.smiles
                elif spec_name in dataset_labels_dict and 'smiles' in dataset_labels_dict[spec_name]:
                    smiles = dataset_labels_dict[spec_name]['smiles']
                
                # 获取adduct
                if hasattr(row, 'ionization') and pd.notna(row.ionization):
                    adduct = row.ionization
                elif spec_name in dataset_labels_dict and 'ionization' in dataset_labels_dict[spec_name]:
                    adduct = dataset_labels_dict[spec_name]['ionization']
                
                if smiles:
                    precursor_mass = chem_utils.mass_from_smi(smiles)
                    if adduct:
                        precursor_mass += common.ion2mass.get(adduct, 0)
            except Exception as e:
                print(f"⚠ 警告: 无法计算precursor mass for {spec_name}: {e}")
                precursor_mass = None
        
        if precursor_mass is None:
            print(f"⚠ 警告: 无法获取precursor mass for {spec_name}，跳过")
            continue
        
        # 加载真实谱图
        true_spec_dict = load_true_spectrum(spec_file, spec_name, num_bins=num_bins, upper_limit=upper_limit)
        
        if not true_spec_dict:
            print(f"⚠ 警告: 无法加载真实谱图 {spec_name}")
            continue
        
        # 加载预测谱图
        pred_spec_dict = load_predicted_spectrum(pred_file, spec_name, num_bins=num_bins, upper_limit=upper_limit)
        
        if not pred_spec_dict:
            print(f"⚠ 警告: 无法加载预测谱图 {spec_name}")
            continue
        
        # 计算cosine distance
        # 将pred_spec_dict转换为list格式（因为dist_bin期望list of dicts）
        pred_list = [pred_spec_dict]
        
        # 计算ignore_peak（如果指定）
        ignore_peak = None
        if args.ignore_parent_peak:
            # 计算precursor peak的bin index
            ignore_peak = int((precursor_mass - 1) * num_bins / upper_limit)
        
        try:
            cosine_dist = cos_dist_bin(
                cand_preds_dict=pred_list,
                true_spec_dict=true_spec_dict,
                sparse=False,  # 已经转换为binned格式
                ignore_peak=ignore_peak,
                agg=True
            )
            
            # cosine similarity = 1 - cosine distance
            cosine_sim = 1 - cosine_dist[0]
            
            # 从SMILES计算inchikey（如果labels中没有）
            smiles = getattr(row, 'smiles', 'N/A')
            try:
                if smiles != 'N/A':
                    inchikey = common.inchikey_from_smiles(smiles)
                else:
                    inchikey = 'N/A'
            except:
                inchikey = 'N/A'
            
            results.append({
                'spec': spec_name,
                'smiles': smiles,
                'inchikey': inchikey,
                'precursor_mass': precursor_mass,
                'cosine_distance': float(cosine_dist[0]),
                'cosine_similarity': float(cosine_sim),
            })
            processed_count += 1
            
            # 每处理100个样本保存一次中间结果
            if processed_count % 100 == 0:
                results_df = pd.DataFrame(results)
                results_df.to_csv(partial_results_file, sep='\t', index=False)
                if USE_TQDM:
                    iterator.set_postfix({'已保存': len(results)})
        except Exception as e:
            print(f"\n⚠ 警告: 计算 {spec_name} 的cosine similarity时出错: {e}")
            continue
    
    if not results:
        print("❌ 没有成功计算的结果")
        return
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_file = output_dir / "evaluation_results.tsv"
    results_df.to_csv(results_file, sep='\t', index=False)
    print(f"\n✓ 结果已保存到: {results_file}")
    
    # 计算统计信息
    cosine_sims = results_df['cosine_similarity'].values
    
    stats = {
        'num_samples': len(results),
        'mean_cosine_similarity': float(np.mean(cosine_sims)),
        'std_cosine_similarity': float(np.std(cosine_sims)),
        'median_cosine_similarity': float(np.median(cosine_sims)),
        'min_cosine_similarity': float(np.min(cosine_sims)),
        'max_cosine_similarity': float(np.max(cosine_sims)),
        'q25_cosine_similarity': float(np.percentile(cosine_sims, 25)),
        'q75_cosine_similarity': float(np.percentile(cosine_sims, 75)),
    }
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("评估结果统计")
    print("=" * 80)
    print(f"样本数量: {stats['num_samples']}")
    print(f"平均Cosine Similarity: {stats['mean_cosine_similarity']:.4f}")
    print(f"标准差: {stats['std_cosine_similarity']:.4f}")
    print(f"中位数: {stats['median_cosine_similarity']:.4f}")
    print(f"最小值: {stats['min_cosine_similarity']:.4f}")
    print(f"最大值: {stats['max_cosine_similarity']:.4f}")
    print(f"25%分位数: {stats['q25_cosine_similarity']:.4f}")
    print(f"75%分位数: {stats['q75_cosine_similarity']:.4f}")
    print("=" * 80)
    
    # 保存统计信息
    stats_file = output_dir / "evaluation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n统计信息已保存到: {stats_file}")


if __name__ == "__main__":
    main()
