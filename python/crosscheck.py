import torch
from scipy.io import loadmat
import numpy as np  # 仅用于把 .mat 中的 cell 转成数组后再喂给 torch

# ---- 你的 PyTorch 版 RAHT_param（与我们之前给的一致） ----
@torch.no_grad()
def RAHT_param_torch(V: torch.Tensor,
                     minV: torch.Tensor,
                     width: float,
                     depth: int,
                     return_one_based: bool = False):
    device = V.device
    N = V.shape[0]
    Q = width / (2 ** depth)
    V = V.to(torch.float64)
    minV = minV.to(torch.float64, device=device)
    Vint = torch.floor((V - minV) / Q).to(torch.int64)

    MC = torch.zeros(N, dtype=torch.int64, device=device)
    tri = torch.tensor([1, 2, 4], dtype=torch.int64, device=device)
    for i in range(1, depth + 1):
        bits = ((Vint >> (i - 1)) & 1)
        MC = MC + (bits[:, [2, 1, 0]].to(torch.int64) * tri).sum(dim=1)
        tri = tri * 8

    Nbits = 3 * depth
    List, Flags, weights = [], [], []

    curr_list = (torch.arange(1, N + 1, device=device, dtype=torch.int64)
                 if return_one_based else
                 torch.arange(N, device=device, dtype=torch.int64))

    def take_MC(idx: torch.Tensor) -> torch.Tensor:
        return MC[idx - 1] if return_one_based else MC[idx]

    j = 1
    while True:
        List.append(curr_list)

        end_sentinel = (N + 1) if return_one_based else N
        next_starts = torch.cat([
            curr_list[1:],
            torch.tensor([end_sentinel], dtype=curr_list.dtype, device=device)
        ])
        w = (next_starts - curr_list).to(torch.int64)
        weights.append(w)

        Mj = take_MC(curr_list).to(torch.int64)
        if Mj.numel() <= 1:
            Flags.append(torch.tensor([False], dtype=torch.bool, device=device))
            break

        diff  = torch.bitwise_xor(Mj[:-1], Mj[1:])
        mask  = (torch.tensor(1, dtype=torch.int64, device=device) << (3*depth)) - \
                (torch.tensor(1, dtype=torch.int64, device=device) << j)
        masked = torch.bitwise_and(diff, mask)
        flag_j = torch.cat([masked.eq(0), torch.tensor([False], device=device, dtype=torch.bool)])
        Flags.append(flag_j)

        prev_flags = torch.cat([torch.tensor([False], device=device, dtype=torch.bool),
                                flag_j[:-1]])
        tmp_list = curr_list[~prev_flags]

        if tmp_list.numel() == 1:
            curr_list = tmp_list
            j += 1
            if j > 64:
                break
            continue

        curr_list = tmp_list
        j += 1
        if j > 64:
            break

    return List, Flags, weights

# ---- 将 MATLAB .mat 里的 cell 转为 torch 列表 ----
def load_raht_param_from_mat(path: str, device=None, one_based=True):
    """
    读取 ref_raht_param.mat，返回 List/Flags/weights（torch 列表）
    """
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    def _cell_to_list(md_value, dtype='long'):
        # 可能是 object 数组或 python list，统一变成 python list
        if isinstance(md_value, np.ndarray) and md_value.dtype == object:
            items = md_value.tolist()
        elif isinstance(md_value, list):
            items = md_value
        else:
            # squeeze_me=True 时单元素 cell 可能被直接 squeeze 成数组
            items = [md_value]

        out = []
        for arr in items:
            a = np.asarray(arr).reshape(-1)  # 列向量/行向量都压成 1D
            if dtype == 'bool':
                t = torch.from_numpy(a.astype(np.bool_))
            else:
                t = torch.from_numpy(a.astype(np.int64))
            if device is not None:
                t = t.to(device)
            out.append(t)
        return out

    L = _cell_to_list(md['ListC'], dtype='long')       # MATLAB 是 1-based
    F = _cell_to_list(md['FlagsC'], dtype='bool')
    W = _cell_to_list(md['weightsC'], dtype='long')
    return L, F, W

# ---- 通用比较函数 ----
def compare_raht_param(L_ref, F_ref, W_ref, L_py, F_py, W_py, one_based=True):
    ok = True
    # 如果 Python 结果是 0-based 而参考是 1-based，统一到 1-based 对比
    if L_py and (L_py[0].min().item() == 0) and one_based:
        L_py = [t + 1 for t in L_py]

    if len(L_ref) != len(L_py):
        print(f"[X] levels length mismatch: ref={len(L_ref)} vs py={len(L_py)}")
        ok = False

    nlevels = min(len(L_ref), len(L_py))
    for j in range(nlevels):
        # List
        if L_ref[j].numel() != L_py[j].numel() or not torch.equal(L_ref[j].reshape(-1), L_py[j].reshape(-1)):
            ok = False
            # 找不同位置
            a, b = L_ref[j].reshape(-1), L_py[j].reshape(-1)
            m = min(a.numel(), b.numel())
            diff_pos = (a[:m] != b[:m]).nonzero().flatten()
            print(f"[X] List[{j}] mismatch. size ref={a.numel()}, py={b.numel()}, first diffs at {diff_pos[:5].tolist()}")
        # Flags
        if F_ref[j].numel() != F_py[j].numel() or not torch.equal(F_ref[j].reshape(-1), F_py[j].reshape(-1)):
            ok = False
            a, b = F_ref[j].reshape(-1), F_py[j].reshape(-1)
            m = min(a.numel(), b.numel())
            diff_pos = (a[:m] != b[:m]).nonzero().flatten()
            print(f"[X] Flags[{j}] mismatch. size ref={a.numel()}, py={b.numel()}, first diffs at {diff_pos[:5].tolist()}")
        # weights
        if W_ref[j].numel() != W_py[j].numel() or not torch.equal(W_ref[j].reshape(-1), W_py[j].reshape(-1)):
            ok = False
            a, b = W_ref[j].reshape(-1), W_py[j].reshape(-1)
            m = min(a.numel(), b.numel())
            diff_pos = (a[:m] != b[:m]).nonzero().flatten()
            print(f"[X] weights[{j}] mismatch. size ref={a.numel()}, py={b.numel()}, first diffs at {diff_pos[:5].tolist()}")

    if ok:
        print("[✓] RAHT_param sanity check PASSED: List/Flags/weights 全部一致。")
    else:
        print("[!] RAHT_param sanity check FAILED（见上面的差异）。")
    return ok

def _ensure_1d_cpu_long(t: torch.Tensor, dtype: torch.dtype):
    """标准化张量：展平成 1D、搬到 CPU、转换类型。"""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    t = t.reshape(-1).to('cpu')
    if dtype is not None:
        t = t.to(dtype)
    return t

def _auto_align_index_base(L_ref, L_py):
    """
    自动检测并对齐索引基（0/1-based）。
    若发现 L_ref[k] 与 L_py[k]+1 完全相等（取第一个尺寸相等的 level），
    则对所有 L_py 统一 +1。
    """
    for k in range(min(len(L_ref), len(L_py))):
        a, b = L_ref[k], L_py[k]
        if a.numel() == b.numel() and a.numel() > 0:
            if torch.equal(a, b):
                return L_ref, L_py, "aligned"
            if torch.equal(a, b + 1):
                L_py = [x + 1 for x in L_py]
                return L_ref, L_py, "python+1"
            if torch.equal(a + 1, b):
                # 这种一般不常见（MATLAB 比 Python 少 1），也处理一下
                L_ref = [x + 1 for x in L_ref]
                return L_ref, L_py, "matlab+1"
            # 找到第一个能判断的层后即可返回
            return L_ref, L_py, "unaligned"
    return L_ref, L_py, "unknown"

def _report_diff(name, level, a: torch.Tensor, b: torch.Tensor, max_show=10):
    """打印差异报告（位置与值）。"""
    print(f"[X] {name}[{level}] mismatch: size ref={a.numel()}, py={b.numel()}")
    m = min(a.numel(), b.numel())
    if m == 0:
        return
    diff_mask = (a[:m] != b[:m])
    n_diff = int(diff_mask.sum().item())
    if n_diff == 0 and a.numel() != b.numel():
        return
    idx = torch.nonzero(diff_mask, as_tuple=False).reshape(-1)
    print(f"    diffs: {n_diff} (show up to {max_show})")
    for i in idx[:max_show]:
        i = int(i.item())
        print(f"    - pos {i}: ref={a[i].item()}  py={b[i].item()}")

def compare_raht_param_torch(L_ref_raw, F_ref_raw, W_ref_raw,
                             L_py_raw,  F_py_raw,  W_py_raw,
                             strict_types: bool = True) -> bool:
    """
    严格比较 MATLAB 与 Python 的 RAHT_param 结果（List/Flags/weights）：
    - 比较 level 数
    - 逐层比较长度
    - **逐值**比较是否完全一致
    - 自动对齐 0/1-based 索引（若检测到整体偏移）
    返回 True 表示全部一致；False 表示存在差异并已打印报告。
    """
    # 1) 标准化为 1D、CPU、期望 dtype
    L_ref = [_ensure_1d_cpu_long(t, torch.long) for t in L_ref_raw]
    F_ref = [_ensure_1d_cpu_long(t, torch.bool) for t in F_ref_raw]
    W_ref = [_ensure_1d_cpu_long(t, torch.long) for t in W_ref_raw]

    L_py  = [_ensure_1d_cpu_long(t, torch.long) for t in L_py_raw]
    F_py  = [_ensure_1d_cpu_long(t, torch.bool) for t in F_py_raw]
    W_py  = [_ensure_1d_cpu_long(t, torch.long) for t in W_py_raw]

    ok = True

    # 2) 对齐索引基（只对 List 做）
    L_ref, L_py, align_flag = _auto_align_index_base(L_ref, L_py)
    if align_flag != "aligned":
        print(f"[!] index-base alignment hint: {align_flag}")

    # 3) 先比 level 数
    if len(L_ref) != len(L_py):
        print(f"[X] levels(List) mismatch: ref={len(L_ref)}, py={len(L_py)}")
        ok = False
    if len(F_ref) != len(F_py):
        print(f"[X] levels(Flags) mismatch: ref={len(F_ref)}, py={len(F_py)}")
        ok = False
    if len(W_ref) != len(W_py):
        print(f"[X] levels(weights) mismatch: ref={len(W_ref)}, py={len(W_py)}")
        ok = False

    nlevels = min(len(L_ref), len(L_py), len(F_ref), len(F_py), len(W_ref), len(W_py))

    # 4) 逐层逐值比较
    for j in range(nlevels):
        # 类型严格性检查
        if strict_types:
            if L_ref[j].dtype != L_py[j].dtype:
                print(f"[!] dtype(List[{j}]): ref={L_ref[j].dtype}, py={L_py[j].dtype}")
            if F_ref[j].dtype != F_py[j].dtype:
                print(f"[!] dtype(Flags[{j}]): ref={F_ref[j].dtype}, py={F_py[j].dtype}")
            if W_ref[j].dtype != W_py[j].dtype:
                print(f"[!] dtype(weights[{j}]): ref={W_ref[j].dtype}, py={W_py[j].dtype}")

        # List
        if (L_ref[j].numel() != L_py[j].numel()) or (not torch.equal(L_ref[j], L_py[j])):
            ok = False
            _report_diff("List", j, L_ref[j], L_py[j])

        # Flags（布尔）
        if (F_ref[j].numel() != F_py[j].numel()) or (not torch.equal(F_ref[j], F_py[j])):
            ok = False
            _report_diff("Flags", j, F_ref[j].to(torch.int64), F_py[j].to(torch.int64))

        # weights（整型）
        if (W_ref[j].numel() != W_py[j].numel()) or (not torch.equal(W_ref[j], W_py[j])):
            ok = False
            _report_diff("weights", j, W_ref[j], W_py[j])

    if ok:
        print("[✓] RAHT_param sanity check PASSED: 维度 & 数值完全一致。")
    else:
        print("[!] RAHT_param sanity check FAILED：见上方差异报告。")
    return ok

def compare_tensor(a: torch.Tensor, b: torch.Tensor, name="", rtol=0.0, atol=0.0, max_show=10):
    """
    返回 True/False，并在不一致时打印详细差异（位置和值）。
    - 对整型/布尔：使用完全相等（torch.equal）
    - 对浮点：使用 allclose（rtol/atol 可调，默认严格相等）
    """
    ok = True
    if a.shape != b.shape:
        print(f"[X] {name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        ok = False

    if a.dtype != b.dtype:
        print(f"[!] {name}: dtype mismatch {a.dtype} vs {b.dtype}（继续比较数值）")

    # 对齐到 CPU，展平成 1D
    a1 = a.reshape(-1).to('cpu')
    b1 = b.reshape(-1).to('cpu')

    # 选择比较方式
    if torch.is_floating_point(a1) or torch.is_floating_point(b1):
        equal = torch.allclose(a1, b1, rtol=rtol, atol=atol) and (a1.numel() == b1.numel())
        diff_mask = torch.isfinite(a1) & torch.isfinite(b1) & (torch.abs(a1 - b1) > (atol + rtol * torch.abs(b1)))
    else:
        equal = torch.equal(a1, b1)
        # 若shape相同但不等，逐元素找不同
        diff_mask = (a1 != b1) if a1.numel() == b1.numel() else torch.ones(min(a1.numel(), b1.numel()), dtype=torch.bool)

    if not equal:
        ok = False
        m = min(a1.numel(), b1.numel())
        if m == 0:
            print(f"[X] {name}: one tensor is empty.")
            return False
        idx = torch.nonzero(diff_mask[:m], as_tuple=False).reshape(-1)
        n_diff = int(idx.numel())
        print(f"[X] {name}: value mismatch, diffs={n_diff}（show up to {max_show}）")
        for i in idx[:max_show]:
            i = int(i)
            av = a1[i].item()
            bv = b1[i].item()
            print(f"    - pos {i}: ref={av}  py={bv}")

    if ok:
        print(f"[✓] {name}: equal.")
    return ok

def compare_lists(ref_list, py_list, base_name, rtol=0.0, atol=0.0):
    """
    比较由多层张量构成的列表（如 List/Flags/weights）。
    """
    ok = True
    if len(ref_list) != len(py_list):
        print(f"[X] {base_name}: levels mismatch ref={len(ref_list)} vs py={len(py_list)}")
        ok = False
    n = min(len(ref_list), len(py_list))
    for j in range(n):
        ok &= compare_tensor(ref_list[j], py_list[j], name=f"{base_name}[{j}]", rtol=rtol, atol=atol)
    return ok

def load_raht_out_mat(path: str, device='cpu'):
    """读取 MATLAB 的 T/w。T -> float64, w -> int64 (列向量形状)。"""
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    T = torch.from_numpy(np.asarray(md['Coeff'], dtype=np.float64)).to(device)
    w = torch.from_numpy(np.asarray(md['w']).reshape(-1).astype(np.int64)).to(device).unsqueeze(1)
    return T, w

def _to1d_cpu(t: torch.Tensor, dtype=None):
    t = torch.as_tensor(t)
    if dtype is not None:
        t = t.to(dtype)
    return t.detach().cpu().reshape(-1)

def load_matlab_raht_out(path, device='cpu'):
    """
    读取 MATLAB 保存的 [T, w] 输出结果 (.mat 文件, 推荐 -v7 格式)
    返回:
        T: torch.FloatTensor [N, D]
        w: torch.LongTensor  [N, 1]
    """
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    T = torch.as_tensor(md['T'], dtype=torch.float64, device=device)
    w = torch.as_tensor(md['w'], dtype=torch.float64, device=device)
    if w.ndim == 1:
        w = w.view(-1, 1)
    w = w.to(torch.int64)
    return T, w


# ==============================================================
# 2. sanity check: 比较 MATLAB 与 Python 输出
# ==============================================================
def compare_RAHT_outputs(T_ref: torch.Tensor, w_ref: torch.Tensor,
                         T_py:  torch.Tensor, w_py:  torch.Tensor,
                         rtol: float = 1e-12, atol: float = 1e-12,
                         max_show: int = 10) -> bool:
    """
    对比 MATLAB 与 Python 的 [T, w] 输出：
    - T: 浮点 allclose（数值近似）
    - w: 整数严格相等
    打印详细差异；返回 True/False。
    """

    ok = True

    # ---- 形状检查 ----
    if T_ref.shape != T_py.shape:
        print(f"[X] T shape mismatch: ref{tuple(T_ref.shape)} vs py{tuple(T_py.shape)}")
        ok = False
    if w_ref.shape != w_py.shape:
        print(f"[X] w shape mismatch: ref{tuple(w_ref.shape)} vs py{tuple(w_py.shape)}")
        ok = False

    # ---- 统一 dtype/设备 ----
    T_ref64 = T_ref.to(torch.float64).cpu()
    T_py64  = T_py.to(torch.float64).cpu()
    w_ref64 = w_ref.to(torch.int64).cpu().reshape(-1)
    w_py64  = w_py.to(torch.int64).cpu().reshape(-1)

    # ---- T 浮点对比 ----
    same_T = torch.allclose(T_ref64, T_py64, rtol=rtol, atol=atol) and (T_ref64.numel() == T_py64.numel())
    if not same_T:
        ok = False
        diff = torch.abs(T_ref64 - T_py64)
        max_diff = diff.max().item() if diff.numel() else float('nan')

        # 找最大误差的位置
        if diff.numel() > 0:
            idx_flat = torch.argmax(diff).item()
            idx2 = np.unravel_index(int(idx_flat), T_ref64.shape)
            i, j = int(idx2[0]), int(idx2[1]) if len(idx2) >= 2 else (int(idx2[0]), 0)
            ref_val = T_ref64[i, j].item() if T_ref64.dim() == 2 else T_ref64[i].item()
            py_val  = T_py64[i, j].item() if T_ref64.dim() == 2 else T_py64[i].item()
            print(f"[X] T mismatch: max|Δ|={max_diff:.3e} at (row {i}, col {j})")
            print(f"    ref={ref_val:.6g}, py={py_val:.6g}")

        # 列出前若干处差异
        a1 = T_ref64.reshape(-1)
        b1 = T_py64.reshape(-1)
        thr = atol + rtol * torch.abs(b1)
        bad = (torch.abs(a1 - b1) > thr)
        where = torch.nonzero(bad, as_tuple=False).reshape(-1)[:max_show]
        print(f"    first {where.numel()} diffs (flat index):")
        for k in where.tolist():
            print(f"      pos {k}: ref={a1[k].item():.6g}, py={b1[k].item():.6g}, |Δ|={abs(a1[k]-b1[k]).item():.3e}")
    else:
        print("[✓] T allclose within tolerance.")

    # ---- w 整数对比 ----
    same_w = (w_ref64.numel() == w_py64.numel()) and torch.equal(w_ref64, w_py64)
    if not same_w:
        ok = False
        print(f"[X] w mismatch.")
        m = min(w_ref64.numel(), w_py64.numel())
        if m > 0:
            bad = (w_ref64[:m] != w_py64[:m])
            where = torch.nonzero(bad, as_tuple=False).reshape(-1)[:max_show]
            print(f"    first {where.numel()} diffs (index):")
            for k in where.tolist():
                print(f"      idx {k}: ref={int(w_ref64[k].item())}, py={int(w_py64[k].item())}")
        if w_ref64.numel() != w_py64.numel():
            print(f"    size: ref={w_ref64.numel()}, py={w_py64.numel()}")
    else:
        print("[✓] w exactly equal.")

    if ok:
        print("[✓] RAHT sanity check PASSED.")
    else:
        print("[!] RAHT sanity check FAILED.")
    return ok