
# %% auto 0
__all__ = ['rng', 'bytes2size', 'b2s', 'memmap2cache', 'cache_memmap', 'a', 'b', 'np_save', 'create_empty_array', 'alphabet',
           'ALPHABET', 'random_choice', 'random_randint', 'random_rand', 'is_nparray', 'is_tensor', 'is_zarr',
           'is_dask', 'is_memmap', 'is_slice', 'totensor', 'toarray', 'toL', 'to3dtensor', 'to2dtensor', 'to1dtensor',
           'to3darray', 'to2darray', 'to1darray', 'to3d', 'to2d', 'to1d', 'to2dPlus', 'to3dPlus', 'to2dPlusTensor',
           'to2dPlusArray', 'to3dPlusTensor', 'to3dPlusArray', 'todtype', 'bytes2str', 'get_size', 'get_dir_size',
           'get_file_size', 'is_np_view', 'is_file', 'is_dir', 'delete_all_in_dir', 'reverse_dict', 'is_tuple',
           'itemify', 'isnone', 'exists', 'ifelse', 'is_not_close', 'test_not_close', 'test_type', 'test_ok',
           'test_not_ok', 'test_error', 'test_eq_nan', 'assert_fn', 'test_gt', 'test_ge', 'test_lt', 'test_le', 'stack',
           'stack_pad', 'pad_sequences', 'match_seq_len', 'random_shuffle', 'cat2int', 'cycle_dl', 'cycle_dl_to_device',
           'cycle_dl_estimate', 'cache_data', 'get_func_defaults', 'get_idx_from_df_col_vals', 'get_sublist_idxs',
           'flatten_list', 'display_pd_df', 'ttest', 'kstest', 'tscore', 'pcc', 'scc', 'remove_fn', 'npsave',
           'permute_2D', 'random_normal', 'random_half_normal', 'random_normal_tensor', 'random_half_normal_tensor',
           'default_dpi', 'get_plot_fig', 'fig2buf', 'plot_scatter', 'get_idxs', 'apply_cmap', 'torch_tile',
           'to_tsfresh_df', 'pcorr', 'scorr', 'torch_diff', 'get_outliers_IQR', 'clip_outliers', 'get_percentile',
           'torch_clamp', 'get_robustscale_params', 'torch_slice_by_dim', 'torch_nanmean', 'torch_nanstd', 'concat',
           'reduce_memory_usage', 'cls_name', 'roll2d', 'roll3d', 'random_roll2d', 'random_roll3d', 'rotate_axis0',
           'rotate_axis1', 'rotate_axis2', 'chunks_calculator', 'is_memory_shared', 'assign_in_chunks', 'create_array',
           'np_save_compressed', 'np_load_compressed', 'np2memmap', 'torch_mean_groupby', 'torch_flip',
           'torch_nan_to_num', 'torch_masked_to_num', 'mpl_trend', 'int2digits', 'array2digits', 'sincos_encoding',
           'linear_encoding', 'encode_positions', 'sort_generator', 'get_subset_dict', 'create_dir', 'remove_dir',
           'named_partial', 'attrdict2dict', 'dict2attrdict', 'dict2yaml', 'yaml2dict', 'get_config', 'str2list',
           'str2index', 'get_cont_cols', 'get_cat_cols', 'get_mapping', 'map_array', 'log_tfm', 'to_sincos_time',
           'plot_feature_dist', 'rolling_moving_average', 'ffill_sequence', 'bfill_sequence', 'fbfill_sequence',
           'dummify', 'shuffle_along_axis', 'analyze_feature', 'analyze_array', 'get_relpath', 'get_root',
           'to_root_path', 'split_in_chunks', 'save_object', 'load_object', 'get_idxs_to_keep', 'zerofy', 'feat2list',
           'smallest_dtype', 'plot_forecast']

# %% ../nbs/002_utils.ipynb 3
from .imports import *
import joblib
import string
import yaml
from numbers import Integral
from numpy.random import default_rng
from scipy.stats import ttest_ind, ks_2samp, pearsonr, spearmanr, normaltest, linregress
warnings.filterwarnings("ignore", category=FutureWarning)

# %% ../nbs/002_utils.ipynb 4
rng = default_rng()
def random_choice(
    a, # 1-D array-like or int. The values from which to draw the samples.
    size=None, # int or tuple of ints, optional. The shape of the output.
    replace=True, # bool, optional. Whether or not to allow the same value to be drawn multiple times.
    p=None, # 1-D array-like, optional. The probabilities associated with each entry in a.
    axis=0, # int, optional. The axis along which the samples are drawn.
    shuffle=True, # bool, optional. Whether or not to shuffle the samples before returning them.
    dtype=None, # data type of the output.
    seed=None, # int or None, optional. Seed for the random number generator.
):
    "Same as np.random.choice but with a faster random generator, dtype and seed"
    rand_gen = default_rng(seed) if seed is not None else rng
    result = rand_gen.choice(a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)
    if dtype is None:
        return result
    return result.astype(dtype=dtype, copy=False)


def random_randint(
    low, # int, lower endpoint of interval (inclusive)
    high=None, # int, upper endpoint of interval (exclusive), or None for a single-argument form of low.
    size=None, # int or tuple of ints, optional. Output shape.
    dtype=int, # data type of the output.
    endpoint=False, # bool, optional. If True, `high` is an inclusive endpoint. If False, the range is open on the right.
    seed=None,  # int or None, optional. Seed for the random number generator.
):
    "Same as np.random.randint but with a faster random generator and seed"
    rand_gen = default_rng(seed) if seed is not None else rng
    return rand_gen.integers(low, high, size=size, dtype=dtype, endpoint=endpoint)


def random_rand(
    *d, # int or tuple of ints, optional. The dimensions of the returned array, must be non-negative.
    dtype=None, # data type of the output.
    out=None, # ndarray, optional. Alternative output array in which to place the result.
    seed=None # int or None, optional. Seed for the random number generator.
):
    "Same as np.random.rand but with a faster random generator, dtype and seed"
    rand_gen = rng if seed is None else default_rng(seed)
    if out is None:
        return rand_gen.random(d, dtype=dtype)
    else:
        rand_gen.random(d, dtype=dtype, out=out)

# %% ../nbs/002_utils.ipynb 8
def is_nparray(o): return isinstance(o, np.ndarray)
def is_tensor(o): return isinstance(o, torch.Tensor)
def is_zarr(o): return hasattr(o, 'oindex')
def is_dask(o): return hasattr(o, 'compute')
def is_memmap(o): return isinstance(o, np.memmap)
def is_slice(o): return isinstance(o, slice)

# %% ../nbs/002_utils.ipynb 10
def totensor(o):
    if isinstance(o, torch.Tensor): return o
    elif isinstance(o, np.ndarray):  return torch.from_numpy(o)
    elif isinstance(o, pd.DataFrame): return torch.from_numpy(o.values)
    else: 
        try: return torch.tensor(o)
        except: warn(f"Can't convert {type(o)} to torch.Tensor", Warning)


def toarray(o):
    if isinstance(o, np.ndarray): return o
    elif isinstance(o, torch.Tensor): return o.cpu().numpy()
    elif isinstance(o, pd.DataFrame): return o.values
    else:
        try: return np.asarray(o)
        except: warn(f"Can't convert {type(o)} to np.array", Warning)
    
    
def toL(o):
    if isinstance(o, L): return o
    elif isinstance(o, (np.ndarray, torch.Tensor)): return L(o.tolist())
    else:
        try: return L(o)
        except: warn(f'passed object needs to be of type L, list, np.ndarray or torch.Tensor but is {type(o)}', Warning)


def to3dtensor(o):
    o = totensor(o)
    if o.ndim == 3: return o
    elif o.ndim == 1: return o[None, None]
    elif o.ndim == 2: return o[:, None]
    assert False, f'Please, review input dimensions {o.ndim}'


def to2dtensor(o):
    o = totensor(o)
    if o.ndim == 2: return o
    elif o.ndim == 1: return o[None]
    elif o.ndim == 3: return o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to1dtensor(o):
    o = totensor(o)
    if o.ndim == 1: return o
    elif o.ndim == 3: return o[0,0]
    if o.ndim == 2: return o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to3darray(o):
    o = toarray(o)
    if o.ndim == 3: return o
    elif o.ndim == 1: return o[None, None]
    elif o.ndim == 2: return o[:, None]
    assert False, f'Please, review input dimensions {o.ndim}'


def to2darray(o):
    o = toarray(o)
    if o.ndim == 2: return o
    elif o.ndim == 1: return o[None]
    elif o.ndim == 3: return o[0]
    assert False, f'Please, review input dimensions {o.ndim}'


def to1darray(o):
    o = toarray(o)
    if o.ndim == 1: return o
    elif o.ndim == 3: o = o[0,0]
    elif o.ndim == 2: o = o[0]
    assert False, f'Please, review input dimensions {o.ndim}'
    
    
def to3d(o):
    if o.ndim == 3: return o
    if isinstance(o, (np.ndarray, pd.DataFrame)): return to3darray(o)
    if isinstance(o, torch.Tensor): return to3dtensor(o)
    
    
def to2d(o):
    if o.ndim == 2: return o
    if isinstance(o, np.ndarray): return to2darray(o)
    if isinstance(o, torch.Tensor): return to2dtensor(o)
    
    
def to1d(o):
    if o.ndim == 1: return o
    if isinstance(o, np.ndarray): return to1darray(o)
    if isinstance(o, torch.Tensor): return to1dtensor(o)
    
    
def to2dPlus(o):
    if o.ndim >= 2: return o
    if isinstance(o, np.ndarray): return to2darray(o)
    elif isinstance(o, torch.Tensor): return to2dtensor(o)
    
    
def to3dPlus(o):
    if o.ndim >= 3: return o
    if isinstance(o, np.ndarray): return to3darray(o)
    elif isinstance(o, torch.Tensor): return to3dtensor(o)
    
    
def to2dPlusTensor(o):
    return to2dPlus(totensor(o))


def to2dPlusArray(o):
    return to2dPlus(toarray(o))


def to3dPlusTensor(o):
    return to3dPlus(totensor(o))


def to3dPlusArray(o):
    return to3dPlus(toarray(o))


def todtype(dtype):
    def _to_type(o, dtype=dtype):
        if o.dtype == dtype: return o
        elif isinstance(o, torch.Tensor): o = o.to(dtype=dtype)
        elif isinstance(o, np.ndarray): o = o.astype(dtype)
        return o
    return _to_type

# %% ../nbs/002_utils.ipynb 13
def bytes2str(
    size_bytes : int, # Number of bytes 
    decimals=2 # Number of decimals in the output
    )->str:
    if size_bytes == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    # s = round(size_bytes / p, decimals)
    return f'{size_bytes / p:.{decimals}f} {size_name[i]}'

bytes2size = bytes2str
b2s = bytes2str


def get_size(
    o,                  # Any python object 
    return_str = False, # True returns size in human-readable format (KB, MB, GB, ...). False in bytes.
    decimals   = 2,     # Number of decimals in the output
):
    if hasattr(o, "base") and o.base is not None: # if it's a view
        return get_size(o.base, return_str=return_str, decimals=decimals)
    if isinstance(o, np.ndarray):
        size = o.nbytes
    elif isinstance(o, torch.Tensor):
        size = sys.getsizeof(o.storage())
    elif isinstance(o, pd.DataFrame):
        size = o.memory_usage(deep=True).sum()
    elif isinstance(o, (list, tuple)):
        size = sum(get_size(i) for i in o)
    elif isinstance(o, dict):
        size = sum(get_size(k) + get_size(v) for k, v in o.items())
    else:
        size = sys.getsizeof(o)
    if return_str: 
        return bytes2str(size, decimals=decimals)
    else:
        return size

def get_dir_size(
    dir_path : str,  # path to directory 
    return_str : bool = True, # True returns size in human-readable format (KB, MB, GB, ...). False in bytes.
    decimals : int = 2, # Number of decimals in the output
    verbose : bool = False, # Controls verbosity
    ):
    assert os.path.isdir(dir_path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                fp_size = os.path.getsize(fp)
                total_size += fp_size
                pv(f'file: {fp[-50:]:50} size: {fp_size}', verbose)
    if return_str: 
        return bytes2str(total_size, decimals=decimals)
    return total_size

def get_file_size(
    file_path : str,  # path to file 
    return_str : bool = True, # True returns size in human-readable format (KB, MB, GB, ...). False in bytes.
    decimals : int = 2, # Number of decimals in the output
    ):
    assert os.path.isfile(file_path)
    fsize = os.path.getsize(file_path)
    if return_str: 
        return bytes2str(fsize, decimals=decimals)
    return fsize

# %% ../nbs/002_utils.ipynb 15
def is_np_view(
    o# a numpy array
):
    return hasattr(o, "base") and o.base is not None

# %% ../nbs/002_utils.ipynb 17
def is_file(path):
    return os.path.isfile(path)

def is_dir(path):
    return os.path.isdir(path)

# %% ../nbs/002_utils.ipynb 19
def delete_all_in_dir(tgt_dir, exception=None):
    import shutil
    if exception is not None and len(L(exception)) > 1: exception = tuple(exception)
    for file in os.listdir(tgt_dir):
        if exception is not None and file.endswith(exception): continue
        file_path = os.path.join(tgt_dir, file)
        if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
        elif os.path.isdir(file_path): shutil.rmtree(file_path)

# %% ../nbs/002_utils.ipynb 20
def reverse_dict(dictionary): 
    return {v: k for k, v in dictionary.items()}

# %% ../nbs/002_utils.ipynb 21
def is_tuple(o): return isinstance(o, tuple)

# %% ../nbs/002_utils.ipynb 22
def itemify(*o, tup_id=None): 
    o = [o_ for o_ in L(*o) if o_ is not None]
    items = L(o).zip()
    if tup_id is not None: return L([item[tup_id] for item in items])
    else: return items

# %% ../nbs/002_utils.ipynb 24
def isnone(o):
    return o is None

def exists(o): return o is not None

def ifelse(a, b, c):
    "`b` if `a` is True else `c`"
    return b if a else c

# %% ../nbs/002_utils.ipynb 26
def is_not_close(a, b, eps=1e-5):
    "Is `a` within `eps` of `b`"
    if hasattr(a, '__array__') or hasattr(b, '__array__'):
        return (abs(a - b) > eps).all()
    if isinstance(a, (Iterable, Generator)) or isinstance(b, (Iterable, Generator)):
        return is_not_close(np.array(a), np.array(b), eps=eps)
    return abs(a - b) > eps


def test_not_close(a, b, eps=1e-5):
    "`test` that `a` is within `eps` of `b`"
    test(a, b, partial(is_not_close, eps=eps), 'not_close')


def test_type(a, b):
    return test_eq(type(a), type(b))


def test_ok(f, *args, **kwargs):
    try: 
        f(*args, **kwargs)
        e = 0
    except: 
        e = 1
        pass
    test_eq(e, 0)
    
def test_not_ok(f, *args, **kwargs):
    try: 
        f(*args, **kwargs)
        e = 0
    except: 
        e = 1
        pass
    test_eq(e, 1)
    
def test_error(error, f, *args, **kwargs):
    try: f(*args, **kwargs)
    except Exception as e: 
        test_eq(str(e), error)
        
def test_eq_nan(a,b):
    "`test` that `a==b` excluding nan values (valid for torch.Tensor and np.ndarray)"
    mask_a = torch.isnan(a) if isinstance(a, torch.Tensor) else np.isnan(a)
    mask_b = torch.isnan(b) if isinstance(b, torch.Tensor) else np.isnan(b)
    test(a[~mask_a],b[~mask_b],equals, '==')

# %% ../nbs/002_utils.ipynb 27
def assert_fn(*args, **kwargs): assert False, 'assertion test'
test_error('assertion test', assert_fn, 35, a=3)

# %% ../nbs/002_utils.ipynb 28
def test_gt(a,b):
    "`test` that `a>b`"
    test(a,b,gt,'>')

def test_ge(a,b):
    "`test` that `a>=b`"
    test(a,b,ge,'>')
    
def test_lt(a,b):
    "`test` that `a>b`"
    test(a,b,lt,'<')

def test_le(a,b):
    "`test` that `a>b`"
    test(a,b,le,'<=')

# %% ../nbs/002_utils.ipynb 31
def stack(o, axis=0, retain=True):
    if hasattr(o, '__array__'): return o
    if isinstance(o[0], torch.Tensor):
        return retain_type(torch.stack(tuple(o), dim=axis),  o[0]) if retain else torch.stack(tuple(o), dim=axis)
    else:
        return retain_type(np.stack(o, axis), o[0]) if retain else np.stack(o, axis)
    
    
def stack_pad(o, padding_value=np.nan):
    'Converts a an iterable into a numpy array using padding if necessary'
    if not is_listy(o) or not is_array(o):
        if not hasattr(o, "ndim"): o = np.asarray([o])
        else: o = np.asarray(o)
    o_ndim = 1
    if o.ndim > 1:
        o_ndim = o.ndim
        o_shape = o.shape
        o = o.flatten()
    o = [oi if (is_array(oi) and oi.ndim > 0) or is_listy(oi) else [oi] for oi in o]
    row_length = len(max(o, key=len))
    result = np.full((len(o), row_length), padding_value)
    for i,row in enumerate(o):
        result[i, :len(row)] = row
    if o_ndim > 1:
        if row_length == 1:
            result = result.reshape(*o_shape)
        else:
            result = result.reshape(*o_shape, row_length)
    return result

# %% ../nbs/002_utils.ipynb 35
def pad_sequences(
    o, # Iterable object
    maxlen:int=None, # Optional max length of the output. If None, max length of the longest individual sequence.
    dtype:(str, type)=np.float64, # Type of the output sequences. To pad sequences with variable length strings, you can use object.
    padding:str='pre', # 'pre' or 'post' pad either before or after each sequence.
    truncating:str='pre', # 'pre' or 'post' remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.
    padding_value:float=np.nan, # Value used for padding.
):
    "Transforms an iterable with sequences into a 3d numpy array using padding or truncating sequences if necessary"
    
    assert padding in ['pre', 'post']
    assert truncating in ['pre', 'post']
    assert is_iter(o)

    if not is_array(o):
        o = [to2darray(oi) for oi in o]
    seq_len = maxlen or max(o, key=len).shape[-1]
    result = np.full((len(o), o[0].shape[-2], seq_len), padding_value, dtype=dtype)
    for i,values in enumerate(o):
        if truncating == 'pre':
            values = values[..., -seq_len:]
        else:
            values = values[..., :seq_len]
        if padding == 'pre':
            result[i, :, -values.shape[-1]:] = values
        else:
            result[i, :, :values.shape[-1]] = values        
    return result

# %% ../nbs/002_utils.ipynb 44
def match_seq_len(*arrays):
    max_len = stack([x.shape[-1] for x in arrays]).max()
    return [np.pad(x, pad_width=((0,0), (0,0), (max_len - x.shape[-1], 0)), mode='constant', constant_values=0) for x in arrays]

# %% ../nbs/002_utils.ipynb 46
def random_shuffle(o, random_state=None):
    import sklearn
    res = sklearn.utils.shuffle(o, random_state=random_state)
    if isinstance(o, L): return L(list(res))
    return res

# %% ../nbs/002_utils.ipynb 48
def cat2int(o):
    from fastai.data.transforms import Categorize
    from fastai.data.core import TfmdLists
    cat = Categorize()
    cat.setup(o)
    return stack(TfmdLists(o, cat)[:])

# %% ../nbs/002_utils.ipynb 51
def cycle_dl(dl, show_progress_bar=True):
    try:
        if show_progress_bar:
            for _ in progress_bar(dl): _
        else:
            for _ in dl: _
    except KeyboardInterrupt:
        pass

        
def cycle_dl_to_device(dl, show_progress_bar=True):
    try:
        if show_progress_bar: 
            for bs in progress_bar(dl): [b.to(default_device()) for b in bs]
        else:
            for bs in dl: [b.to(default_device()) for b in bs]
    except KeyboardInterrupt:
        pass
        
def cycle_dl_estimate(dl, iters=10):
    iters = min(iters, len(dl))
    iterator = iter(dl)
    timer.start(False)
    try:
        for _ in range(iters): next(iterator)
    except KeyboardInterrupt:
        pass
    t = timer.stop()
    return (t/iters * len(dl)).total_seconds()

# %% ../nbs/002_utils.ipynb 52
def cache_data(o, slice_len=10_000, verbose=False):
    start = 0
    n_loops = (len(o) - 1) // slice_len + 1
    pv(f'{n_loops} loops', verbose)
    timer.start(False)
    for i in range(n_loops):
        o[slice(start,start + slice_len)]        
        if verbose and (i+1) % 10 == 0: print(f'{i+1:4} elapsed time: {timer.elapsed()}')
        start += slice_len
    pv(f'{i+1:4} total time  : {timer.stop()}\n', verbose)
    
memmap2cache =  cache_data
cache_memmap = cache_data

# %% ../nbs/002_utils.ipynb 53
def get_func_defaults(f): 
    import inspect
    fa = inspect.getfullargspec(f)
    if fa.defaults is None: return dict(zip(fa.args, [''] * (len(fa.args))))
    else: return dict(zip(fa.args, [''] * (len(fa.args) - len(fa.defaults)) + list(fa.defaults)))

# %% ../nbs/002_utils.ipynb 54
def get_idx_from_df_col_vals(df, col, val_list):
    return [df[df[col] == val].index[0] for val in val_list]

# %% ../nbs/002_utils.ipynb 55
def get_sublist_idxs(aList, bList):
    "Get idxs that when applied to aList will return bList. aList must contain all values in bList"
    sorted_aList = aList[np.argsort(aList)]
    return np.argsort(aList)[np.searchsorted(sorted_aList, bList)]

# %% ../nbs/002_utils.ipynb 57
def flatten_list(l):
    return [item for sublist in l for item in sublist]

# %% ../nbs/002_utils.ipynb 58
def display_pd_df(df, max_rows:Union[bool, int]=False, max_columns:Union[bool, int]=False):
    if max_rows:
        old_max_rows = pd.get_option('display.max_rows')
        if max_rows is not True and isinstance(max_rows, Integral): pd.set_option('display.max_rows', max_rows)
        else: pd.set_option('display.max_rows', df.shape[0])
    if max_columns:
        old_max_columns = pd.get_option('display.max_columns')
        if max_columns is not True and isinstance(max_columns, Integral): pd.set_option('display.max_columns', max_columns)
        else: pd.set_option('display.max_columns', df.shape[1])
    display(df)
    if max_rows: pd.set_option('display.max_rows', old_max_rows)
    if max_columns: pd.set_option('display.max_columns', old_max_columns)

# %% ../nbs/002_utils.ipynb 60
def ttest(data1, data2, equal_var=False):
    "Calculates t-statistic and p-value based on 2 sample distributions"
    t_stat, p_value = ttest_ind(data1, data2, equal_var=equal_var)
    return t_stat, np.sign(t_stat) * p_value

def kstest(data1, data2, alternative='two-sided', mode='auto', by_axis=None):
    """Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
    
    Parameters
    data1, data2: Two arrays of sample observations assumed to be drawn from a continuous distributions. Sample sizes can be different.
    alternative: {‘two-sided’, ‘less’, ‘greater’}, optional. Defines the null and alternative hypotheses. Default is ‘two-sided’. 
    mode: {‘auto’, ‘exact’, ‘asymp’}, optional. Defines the method used for calculating the p-value. 
    by_axis (optional, int): for arrays with more than 1 dimension, the test will be run for each variable in that axis if by_axis is not None.
    """
    if by_axis is None:
        stat, p_value = ks_2samp(data1.flatten(), data2.flatten(), alternative=alternative, mode=mode)
        return stat, np.sign(stat) * p_value
    else:
        assert data1.shape[by_axis] == data2.shape[by_axis], f"both arrays must have the same size along axis {by_axis}"
        stats, p_values = [], []
        for i in range(data1.shape[by_axis]):
            d1 = np.take(data1, indices=i, axis=by_axis)
            d2 = np.take(data2, indices=i, axis=by_axis)
            stat, p_value = ks_2samp(d1.flatten(), d2.flatten(), alternative=alternative, mode=mode)
            stats.append(stat) 
            p_values.append(np.sign(stat) * p_value)
        return stats, p_values  
        

def tscore(o): 
    if o.std() == 0: return 0
    else: return np.sqrt(len(o)) * o.mean() / o.std()

# %% ../nbs/002_utils.ipynb 66
def pcc(a, b):
    return pearsonr(a, b)[0]

def scc(a, b):
    return spearmanr(a, b)[0]

a = np.random.normal(0.5, 1, 100)
b = np.random.normal(0.15, .5, 100)
pcc(a, b), scc(a, b)

# %% ../nbs/002_utils.ipynb 67
def remove_fn(fn, verbose=False):
    "Removes a file (fn) if exists"
    try: 
        os.remove(fn)
        pv(f'{fn} file removed', verbose)
    except OSError: 
        pv(f'{fn} does not exist', verbose)
        pass

# %% ../nbs/002_utils.ipynb 68
def npsave(array_fn, array, verbose=True):
    remove_fn(array_fn, verbose)
    pv(f'saving {array_fn}...', verbose)
    np.save(array_fn, array)
    pv(f'...{array_fn} saved', verbose)
    
np_save = npsave

# %% ../nbs/002_utils.ipynb 70
def permute_2D(array, axis=None):
    "Permute rows or columns in an array. This can be used, for example, in feature permutation"
    if axis == 0: return array[np.random.randn(*array.shape).argsort(axis=0), np.arange(array.shape[-1])[None, :]] 
    elif axis == 1 or axis == -1: return array[np.arange(len(array))[:,None], np.random.randn(*array.shape).argsort(axis=1)] 
    return array[np.random.randn(*array.shape).argsort(axis=0), np.random.randn(*array.shape).argsort(axis=1)] 

# %% ../nbs/002_utils.ipynb 72
def random_normal():
    "Returns a number between -1 and 1 with a normal distribution"
    while True:
        o = np.random.normal(loc=0., scale=1/3)
        if abs(o) <= 1: break
    return o

def random_half_normal():
    "Returns a number between 0 and 1 with a half-normal distribution"
    while True:
        o = abs(np.random.normal(loc=0., scale=1/3))
        if o <= 1: break
    return o

def random_normal_tensor(shape=1, device=None):
    "Returns a tensor of a predefined shape between -1 and 1 with a normal distribution"
    return torch.empty(shape, device=device).normal_(mean=0, std=1/3).clamp_(-1, 1)

def random_half_normal_tensor(shape=1, device=None):
    "Returns a tensor of a predefined shape between 0 and 1 with a half-normal distribution"
    return abs(torch.empty(shape, device=device).normal_(mean=0, std=1/3)).clamp_(0, 1)

# %% ../nbs/002_utils.ipynb 73
from matplotlib.backends.backend_agg import FigureCanvasAgg

def default_dpi():
    DPI = plt.gcf().get_dpi()
    plt.close()
    return int(DPI)

def get_plot_fig(size=None, dpi=default_dpi()):
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, frameon=False) if size else plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    config = plt.gcf()
    plt.close('all')
    return config

def fig2buf(fig):
    canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]

# %% ../nbs/002_utils.ipynb 75
def plot_scatter(x, y, deg=1):
    linreg = linregress(x, y)
    plt.scatter(x, y, label=f'R2:{linreg.rvalue:.2f}', color='lime', edgecolor='black', alpha=.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, deg))(np.unique(x)), color='r')
    plt.legend(loc='best')
    plt.show()

# %% ../nbs/002_utils.ipynb 77
def get_idxs(o, aList): return array([o.tolist().index(v) for v in aList])

# %% ../nbs/002_utils.ipynb 79
def apply_cmap(o, cmap):
    o = toarray(o)
    out = plt.get_cmap(cmap)(o)[..., :3]
    out = tensor(out).squeeze(1)
    return out.permute(0, 3, 1, 2)

# %% ../nbs/002_utils.ipynb 81
def torch_tile(a, n_tile, dim=0):
    if ismin_torch("1.10") and dim == 0:
        if isinstance(n_tile, tuple): 
            return torch.tile(a, n_tile)
        return torch.tile(a, (n_tile,))
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cat([init_dim * torch.arange(n_tile) + i for i in range(init_dim)]).to(device=a.device)
    return torch.index_select(a, dim, order_index)

# %% ../nbs/002_utils.ipynb 83
def to_tsfresh_df(ts):
    r"""Prepares a time series (Tensor/ np.ndarray) to be used as a tsfresh dataset to allow feature extraction"""
    ts = to3d(ts)
    if isinstance(ts, np.ndarray):
        ids = np.repeat(np.arange(len(ts)), ts.shape[-1]).reshape(-1,1)
        joint_ts =  ts.transpose(0,2,1).reshape(-1, ts.shape[1])
        cols = ['id'] + np.arange(ts.shape[1]).tolist()
        df = pd.DataFrame(np.concatenate([ids, joint_ts], axis=1), columns=cols)
    elif isinstance(ts, torch.Tensor):
        ids = torch_tile(torch.arange(len(ts)), ts.shape[-1]).reshape(-1,1)
        joint_ts =  ts.transpose(1,2).reshape(-1, ts.shape[1])
        cols = ['id']+np.arange(ts.shape[1]).tolist()
        df = pd.DataFrame(torch.cat([ids, joint_ts], dim=1).numpy(), columns=cols)
    df['id'] = df['id'].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df

# %% ../nbs/002_utils.ipynb 85
def pcorr(a, b): 
    return pearsonr(a, b)

def scorr(a, b): 
    corr = spearmanr(a, b)
    return corr[0], corr[1]

# %% ../nbs/002_utils.ipynb 86
def torch_diff(t, lag=1, pad=True, append=0):
    import torch.nn.functional as F
    diff = t[..., lag:] - t[..., :-lag]
    if pad: 
        return F.pad(diff, (lag, append))
    else: 
        return diff

# %% ../nbs/002_utils.ipynb 88
def get_outliers_IQR(o, axis=None, quantile_range=(25.0, 75.0)):
    if isinstance(o, torch.Tensor):
        Q1 = torch.nanquantile(o, quantile_range[0]/100, axis=axis, keepdims=axis is not None)
        Q3 = torch.nanquantile(o, quantile_range[1]/100, axis=axis, keepdims=axis is not None)
    else:
        Q1 = np.nanpercentile(o, quantile_range[0], axis=axis, keepdims=axis is not None)
        Q3 = np.nanpercentile(o, quantile_range[1], axis=axis, keepdims=axis is not None)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

def clip_outliers(o, axis=None):
    min_outliers, max_outliers = get_outliers_IQR(o, axis=axis)
    if isinstance(o, (np.ndarray, pd.core.series.Series)):
        return np.clip(o, min_outliers, max_outliers)
    elif isinstance(o, torch.Tensor):
        return torch.clamp(o, min_outliers, max_outliers)

def get_percentile(o, percentile, axis=None):
    if isinstance(o, torch.Tensor): 
        return torch.nanquantile(o, percentile/100, axis=axis, keepdims=axis is not None)
    else: 
        return np.nanpercentile(o, percentile, axis=axis, keepdims=axis is not None)

def torch_clamp(o, min=None, max=None):
    r"""Clamp torch.Tensor using 1 or multiple dimensions"""
    if min is not None: o = torch.max(o, min)
    if max is not None: o = torch.min(o, max)
    return o

# %% ../nbs/002_utils.ipynb 90
def get_robustscale_params(o, sel_vars=None, not_sel_vars=None, by_var=True, percentiles=(25, 75), eps=1e-6):
    "Calculates median and inter-quartile range required to robust scaler inputs"
    assert o.ndim == 3
    if by_var: 
        axis=(0,2)
        keepdims=True
    else:
        axis=None
        keepdims=False
    median = np.nanpercentile(o, 50, axis=axis, keepdims=keepdims)
    Q1 = np.nanpercentile(o, percentiles[0], axis=axis, keepdims=keepdims)
    Q3 = np.nanpercentile(o, percentiles[1], axis=axis, keepdims=keepdims)
    IQR = Q3 - Q1

    if eps is not None: 
        IQR = np.clip(IQR, eps, None)
        
    if sel_vars is not None:
        not_sel_vars = np.asarray([v for v in np.arange(o.shape[1]) if v not in sel_vars])
        
    if not_sel_vars is not None:
        median[:, not_sel_vars] = 0
        IQR[:, not_sel_vars] = 1
        
    return median, IQR


# %% ../nbs/002_utils.ipynb 92
def torch_slice_by_dim(t, index, dim=-1, **kwargs):
    if not isinstance(index, torch.Tensor): index = torch.Tensor(index)
    assert t.ndim == index.ndim, "t and index must have the same ndim"
    index = index.long()
    return torch.gather(t, dim, index, **kwargs)

# %% ../nbs/002_utils.ipynb 94
def torch_nanmean(o, dim=None, keepdim=False):
    """There's currently no torch.nanmean function"""
    mask = torch.isnan(o)
    if mask.any():
        output = torch.from_numpy(np.asarray(np.nanmean(o.cpu().numpy(), axis=dim, keepdims=keepdim))).to(o.device)
        if output.shape == mask.shape:
            output[mask] = 0
        return output
    else:
        return torch.mean(o, dim=dim, keepdim=keepdim) if dim is not None else torch.mean(o)


def torch_nanstd(o, dim=None, keepdim=False):
    """There's currently no torch.nanstd function"""
    mask = torch.isnan(o)
    if mask.any():
        output = torch.from_numpy(np.asarray(np.nanstd(o.cpu().numpy(), axis=dim, keepdims=keepdim))).to(o.device)
        if output.shape == mask.shape:
            output[mask] = 1
        return output
    else:
        return torch.std(o, dim=dim, keepdim=keepdim) if dim is not None else torch.std(o)

# %% ../nbs/002_utils.ipynb 96
def concat(*ls, dim=0):
    "Concatenate tensors, arrays, lists, or tuples by a dimension"
    if not len(ls): return []
    it = ls[0]
    if isinstance(it, torch.Tensor): return torch.cat(ls, dim=dim)
    elif isinstance(it, np.ndarray): return np.concatenate(ls, axis=dim)
    else:
        res = np.concatenate(ls, axis=dim).tolist()
        return retain_type(res, typ=type(it))

# %% ../nbs/002_utils.ipynb 97
def reduce_memory_usage(df):
    
    start_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_memory} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')
    
    end_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe after reduction {end_memory} MB")
    print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df

# %% ../nbs/002_utils.ipynb 98
def cls_name(o): return o.__class__.__name__

# %% ../nbs/002_utils.ipynb 100
def roll2d(o, roll1: Union[None, list, int] = None, roll2: Union[None, list, int] = None):
    """Rolls a 2D object on the indicated axis
    This solution is based on https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    """
    
    assert o.ndim == 2, "roll2D can only be applied to 2d objects"
    axis1, axis2 = np.ogrid[:o.shape[0], :o.shape[1]]
    if roll1 is not None:
        if isinstance(roll1, int): axis1 = axis1 - np.array(roll1).reshape(1,1)
        else: axis1 = np.array(roll1).reshape(o.shape[0],1)
    if roll2 is not None:
        if isinstance(roll2, int):  axis2 = axis2 - np.array(roll2).reshape(1,1)
        else: axis2 = np.array(roll2).reshape(1,o.shape[1])
    return o[axis1, axis2]


def roll3d(o, roll1: Union[None, list, int] = None, roll2: Union[None, list, int] = None, roll3: Union[None, list, int] = None):
    """Rolls a 3D object on the indicated axis
    This solution is based on https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    """
    
    assert o.ndim == 3, "roll3D can only be applied to 3d objects"
    axis1, axis2, axis3 = np.ogrid[:o.shape[0], :o.shape[1], :o.shape[2]]
    if roll1 is not None:
        if isinstance(roll1, int): axis1 = axis1 - np.array(roll1).reshape(1,1,1)
        else: axis1 = np.array(roll1).reshape(o.shape[0],1,1)
    if roll2 is not None:
        if isinstance(roll2, int):  axis2 = axis2 - np.array(roll2).reshape(1,1,1)
        else: axis2 = np.array(roll2).reshape(1,o.shape[1],1)
    if roll3 is not None:
        if isinstance(roll3, int):  axis3 = axis3 - np.array(roll3).reshape(1,1,1)
        else: axis3 = np.array(roll3).reshape(1,1,o.shape[2])
    return o[axis1, axis2, axis3]


def random_roll2d(o, axis=(), replace=False):
    """Rolls a 2D object on the indicated axis
    This solution is based on https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    """
    
    assert o.ndim == 2, "roll2D can only be applied to 2d objects"
    axis1, axis2 = np.ogrid[:o.shape[0], :o.shape[1]]
    if 0 in axis:
        axis1 = random_choice(np.arange(o.shape[0]), o.shape[0], replace).reshape(-1, 1)
    if 1 in axis:
        axis2 = random_choice(np.arange(o.shape[1]), o.shape[1], replace).reshape(1, -1)
    return o[axis1, axis2]


def random_roll3d(o, axis=(), replace=False):
    """Randomly rolls a 3D object along the indicated axes
    This solution is based on https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    """
    
    assert o.ndim == 3, "random_roll3d can only be applied to 3d objects"
    axis1, axis2, axis3 = np.ogrid[:o.shape[0], :o.shape[1], :o.shape[2]]
    if 0 in axis:
        axis1 = random_choice(np.arange(o.shape[0]), o.shape[0], replace).reshape(-1, 1, 1)
    if 1 in axis:
        axis2 = random_choice(np.arange(o.shape[1]), o.shape[1], replace).reshape(1, -1, 1)
    if 2 in axis:
        axis3 = random_choice(np.arange(o.shape[2]), o.shape[2], replace).reshape(1, 1, -1)
    return o[axis1, axis2, axis3]

def rotate_axis0(o, steps=1):
    return o[np.arange(o.shape[0]) - steps]

def rotate_axis1(o, steps=1):
    return o[:, np.arange(o.shape[1]) - steps]

def rotate_axis2(o, steps=1):
    return o[:, :, np.arange(o.shape[2]) - steps]

# %% ../nbs/002_utils.ipynb 105
def chunks_calculator(shape, dtype='float32', n_bytes=1024**3):
    """Function to calculate chunks for a given size of n_bytes (default = 1024**3 == 1GB). 
    It guarantees > 50% of the chunk will be filled"""
    
    X  = np.random.rand(1, *shape[1:]).astype(dtype)
    byts = get_size(X, return_str=False)
    n = n_bytes // byts
    if shape[0] / n <= 1: return False
    remainder = shape[0] % n
    if remainder / n < .5: 
        n_chunks = shape[0] // n
        n += np.ceil(remainder / n_chunks).astype(int)
    return (n, -1, -1)

# %% ../nbs/002_utils.ipynb 107
def is_memory_shared(a, b):
    "Check if 2 array-like objects share memory"
    assert is_array(a) and is_array(b)
    return np.shares_memory(a, b)

# %% ../nbs/002_utils.ipynb 109
def assign_in_chunks(a, b, chunksize='auto', inplace=True, verbose=True):
    """Assigns values in b to an array-like object a using chunks to avoid memory overload.
    The resulting a retains it's dtype and share it's memory.
    a: array-like object
    b: may be an integer, float, str, 'rand' (for random data), or another array like object.
    chunksize: is the size of chunks. If 'auto' chunks will have around 1GB each.
    """

    if b != 'rand' and not isinstance(b, (Iterable, Generator)):
        a[:] = b
    else:
        shape = a.shape
        dtype = a.dtype
        if chunksize == "auto":
            chunksize = chunks_calculator(shape, dtype)
            chunksize = shape[0] if not chunksize else  chunksize[0]
            if verbose: 
                print(f'auto chunksize: {chunksize}')
        for i in progress_bar(range((shape[0] - 1) // chunksize + 1), display=verbose, leave=False):
            start, end = i * chunksize, min(shape[0], (i + 1) * chunksize)
            if start >= shape[0]: break
            if b == 'rand':
                a[start:end] = np.random.rand(end - start, *shape[1:])
            else:
                if is_dask(b):
                    a[start:end] = b[start:end].compute()
                else:
                    a[start:end] = b[start:end]
    if not inplace: return a

# %% ../nbs/002_utils.ipynb 112
def create_array(shape, fname=None, path='./data', on_disk=True, dtype='float32', mode='r+', fill_value='rand', chunksize='auto', verbose=True, **kwargs):
    """
    mode:
        ‘r’:  Open existing file for reading only.
        ‘r+’: Open existing file for reading and writing.
        ‘w+’: Create or overwrite existing file for reading and writing.
        ‘c’:  Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
    fill_value: 'rand' (for random numbers), int or float
    chunksize = 'auto' to calculate chunks of 1GB, or any integer (for a given number of samples)
    """
    if on_disk:
        assert fname is not None, 'you must provide a fname (filename)'
        path = Path(path)
        if not fname.endswith('npy'): fname = f'{fname}.npy'
        filename = path/fname
        filename.parent.mkdir(parents=True, exist_ok=True)
        # Save a small empty array
        _temp_fn = path/'temp_X.npy'
        np.save(_temp_fn, np.empty(0))
        # Create  & save file
        arr = np.memmap(_temp_fn, dtype=dtype, mode='w+', shape=shape, **kwargs)
        np.save(filename, arr)
        del arr
        os.remove(_temp_fn)
        # Open file in selected mode
        arr = np.load(filename, mmap_mode=mode)
    else:
        arr = np.empty(shape, dtype=dtype, **kwargs)
    if fill_value != 0:
        assign_in_chunks(arr, fill_value, chunksize=chunksize, inplace=True, verbose=verbose)
    return arr

create_empty_array = partial(create_array, fill_value=0)

# %% ../nbs/002_utils.ipynb 115
import gzip

def np_save_compressed(arr, fname=None, path='./data', verbose=False, **kwargs):
    assert fname is not None, 'you must provide a fname (filename)'
    if fname.endswith('npy'): fname = f'{fname}.gz'
    elif not fname.endswith('npy.gz'): fname = f'{fname}.npy.gz'
    filename = Path(path)/fname
    filename.parent.mkdir(parents=True, exist_ok=True)
    f = gzip.GzipFile(filename, 'w', **kwargs)
    np.save(file=f, arr=arr)
    f.close()
    pv(f'array saved to {filename}', verbose)
    
def np_load_compressed(fname=None, path='./data', **kwargs):
    assert fname is not None, 'you must provide a fname (filename)'
    if fname.endswith('npy'): fname = f'{fname}.gz'
    elif not fname.endswith('npy.gz'): fname = f'{fname}.npy.gz'
    filename = Path(path)/fname
    f = gzip.GzipFile(filename, 'r', **kwargs)
    arr = np.load(f)
    f.close()
    return arr

# %% ../nbs/002_utils.ipynb 117
def np2memmap(arr, fname=None, path='./data', dtype='float32', mode='c', **kwargs):
    """ Function that turns an ndarray into a memmap ndarray
    mode:
        ‘r’:  Open existing file for reading only.
        ‘r+’: Open existing file for reading and writing.
        ‘w+’: Create or overwrite existing file for reading and writing.
        ‘c’:  Copy-on-write: assignments affect data in memory, but changes are not saved to disk. The file on disk is read-only.
    """
    assert fname is not None, 'you must provide a fname (filename)'
    if not fname.endswith('npy'): fname = f'{fname}.npy'
    filename = Path(path)/fname
    filename.parent.mkdir(parents=True, exist_ok=True)
    # Save file
    np.save(filename, arr)
    # Open file in selected mode
    arr = np.load(filename, mmap_mode=mode)
    return arr

# %% ../nbs/002_utils.ipynb 119
def torch_mean_groupby(o, idxs):
    """Computes torch mean along axis 0 grouped by the idxs. 
    Need to ensure that idxs have the same order as o"""
    if is_listy(idxs[0]): idxs = flatten_list(idxs)
    flattened_idxs = torch.tensor(idxs)
    idxs, vals = torch.unique(flattened_idxs, return_counts=True)
    vs = torch.split_with_sizes(o, tuple(vals))
    return torch.cat([v.mean(0).unsqueeze(0) for k,v in zip(idxs, vs)])

# %% ../nbs/002_utils.ipynb 121
def torch_flip(t, dims=-1):
    if dims == -1: return t[..., np.arange(t.shape[dims])[::-1].copy()]
    elif dims == 0: return t[np.arange(t.shape[dims])[::-1].copy()]
    elif dims == 1: return t[:, np.arange(t.shape[dims])[::-1].copy()]
    elif dims == 2: return t[:, :, np.arange(t.shape[dims])[::-1].copy()]

# %% ../nbs/002_utils.ipynb 123
def torch_nan_to_num(o, num=0, inplace=False):
    if ismin_torch("1.8") and not inplace: 
        return torch.nan_to_num(o, num)
    mask = torch.isnan(o)
    return torch_masked_to_num(o, mask, num=num, inplace=inplace)

def torch_masked_to_num(o, mask, num=0, inplace=False):
    if inplace: 
        o[:] = o.masked_fill(mask, num)
    else: 
        return o.masked_fill(mask, num)

# %% ../nbs/002_utils.ipynb 127
def mpl_trend(x, y, deg=1): 
    return np.poly1d(np.polyfit(x, y, deg))(x)

# %% ../nbs/002_utils.ipynb 129
def int2digits(o, n_digits=None, normalize=True):
    if n_digits is not None:
        iterable = '0' * (n_digits - len(str(abs(o)))) + str(abs(o))
    else:
        iterable = str(abs(o))
    sign = np.sign(o)
    digits = np.array([sign * int(d) for d in iterable])
    if normalize:
        digits = digits / 10
    return digits


def array2digits(o, n_digits=None, normalize=True):
    output = np.array(list(map(partial(int2digits, n_digits=n_digits), o)))
    if normalize:
        output = output / 10
    return output

# %% ../nbs/002_utils.ipynb 131
def sincos_encoding(seq_len, device=None, to_np=False):
    if to_np:
        sin = np.sin(np.arange(seq_len) / seq_len * 2 * np.pi)
        cos = np.cos(np.arange(seq_len) / seq_len * 2 * np.pi)
    else:
        if device is None: device = default_device()
        sin = torch.sin(torch.arange(seq_len, device=device) / seq_len * 2 * np.pi)
        cos = torch.cos(torch.arange(seq_len, device=device) / seq_len * 2 * np.pi)
    return sin, cos

# %% ../nbs/002_utils.ipynb 133
def linear_encoding(seq_len, device=None, to_np=False, lin_range=(-1,1)):
    if to_np:
        enc =  np.linspace(lin_range[0], lin_range[1], seq_len)
    else:
        if device is None: device = default_device()
        enc = torch.linspace(lin_range[0], lin_range[1], seq_len, device=device)
    return enc

# %% ../nbs/002_utils.ipynb 135
def encode_positions(pos_arr, min_val=None, max_val=None, linear=False, lin_range=(-1,1)):
    """ Encodes an array with positions using a linear or sincos methods
    """
    
    if min_val is None:
        min_val = np.nanmin(pos_arr)
    if max_val is None:
        max_val = np.nanmax(pos_arr)
        
    if linear: 
        return (((pos_arr - min_val)/(max_val - min_val)) * (lin_range[1] - lin_range[0]) + lin_range[0])
    else:
        sin = np.sin((pos_arr - min_val)/(max_val - min_val) * 2 * np.pi)
        cos = np.cos((pos_arr - min_val)/(max_val - min_val) * 2 * np.pi)
        return sin, cos

# %% ../nbs/002_utils.ipynb 138
def sort_generator(generator, bs):
    g = list(generator)
    for i in range(len(g)//bs + 1): g[bs*i:bs*(i+1)] = np.sort(g[bs*i:bs*(i+1)])
    return (i for i in g)

# %% ../nbs/002_utils.ipynb 140
def get_subset_dict(d, keys):
    return dict((k,d[k]) for k in listify(keys) if k in d)

# %% ../nbs/002_utils.ipynb 142
def create_dir(directory, verbose=True): 
    if not is_listy(directory): directory = [directory]
    for d in directory:
        d = Path(d)
        if d.exists():
            if verbose: print(f"{d} directory already exists.")
        else: 
            d.mkdir(parents=True, exist_ok=True)
            assert d.exists(),  f"a problem has occurred while creating {d}"
            if verbose: print(f"{d} directory created.")


def remove_dir(directory, verbose=True):
    import shutil
    if not is_listy(directory): directory = [directory]
    for d in directory:
        d = Path(d)
        if d.is_file(): d = d.parent
        if not d.exists():
            if verbose: print(f"{d} directory doesn't exist.")
        else:
            shutil.rmtree(d)
            assert not d.exists(), f"a problem has occurred while deleting {d}"
            if verbose: print(f"{d} directory removed.")

# %% ../nbs/002_utils.ipynb 147
class named_partial(object):
    """Create a partial function with a __name__"""
    
    def __init__(self, name, func, *args, **kwargs):
        self._func = partial(func, *args, **kwargs)
        self.__name__ = name
    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
    def __repr__(self):
        return self.__name__

# %% ../nbs/002_utils.ipynb 149
def attrdict2dict(
    d: dict,  # a dict
):
    "Converts a (nested) AttrDict dict to a dict."
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = attrdict2dict(d[k])
        elif is_listy(v):
            d[k] = list(v)  # convert L to list
    return d


def dict2attrdict(
    d: dict,  # a dict
):
    "Converts a (nested) dict to an AttrDict."
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2attrdict(d[k])
        elif is_listy(v):
            d[k] = list(v)  # convert L to list
    return AttrDict(d)

# %% ../nbs/002_utils.ipynb 151
def dict2yaml(
    d, # a dict
    file_path, # a path to a yaml file
    sort_keys=False, # if True, sort the keys
):
    "Converts a dict to a yaml file."
    file_path = Path(file_path)
    if not file_path.suffix == '.yaml':
        file_path = file_path.with_suffix(".yaml")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as outfile:
        yaml.dump(d, outfile, default_flow_style=False, sort_keys=sort_keys)


def yaml2dict(
    file_path, # a path to a yaml file
    attrdict=True, # if True, convert output to AttrDict
):
    "Converts a yaml file to a dict (optionally AttrDict)."
    file_path = Path(file_path)
    if not file_path.suffix == '.yaml':
        file_path = file_path.with_suffix(".yaml")
    with open(file_path, "r") as infile:
        d = yaml.load(infile, Loader=yaml.FullLoader)
    if not d:  # if file is empty
        return {}
    return dict2attrdict(d) if attrdict else d


def get_config(file_path):
    "Gets a config from a yaml file."
    file_path = Path(file_path)
    if not file_path.suffix == ".yaml":
        file_path = file_path.with_suffix(".yaml")
    cfg = yaml2dict(file_path)
    config = cfg.get("config") or cfg
    config = dict2attrdict(config)
    return config

# %% ../nbs/002_utils.ipynb 154
def str2list(o):
    if o is None: return []
    elif o is not None and not isinstance(o, (list, L)):
        if isinstance(o, pd.core.indexes.base.Index): o = o.tolist()
        else: o = [o]
    return o

def str2index(o):
    if o is None: return o
    o = str2list(o)
    if len(o) == 1: return o[0]
    return o

def get_cont_cols(df):
    return df._get_numeric_data().columns.tolist()

def get_cat_cols(df):
    cols = df.columns.tolist()
    cont_cols = df._get_numeric_data().columns.tolist()
    return [col for col in cols if col not in cont_cols]

# %% ../nbs/002_utils.ipynb 155
alphabet = L(list(string.ascii_lowercase))
ALPHABET = L(list(string.ascii_uppercase))

# %% ../nbs/002_utils.ipynb 156
def get_mapping(arr, dim=1, return_counts=False):
    maps = [L(np.unique(np.take(arr, i, dim)).tolist()) for i in range(arr.shape[dim])]
    if return_counts:
        counts = [len(m) for m in maps]
        return maps, counts
    return maps

def map_array(arr, dim=1):
    out = stack([np.unique(np.take(arr, i, dim), return_inverse=True)[1] for i in range(arr.shape[dim])])
    if dim == 1: out = out.T
    return out

# %% ../nbs/002_utils.ipynb 159
def log_tfm(o, inplace=False):
    "Log transforms an array-like object with positive and/or negative values"
    if isinstance(o, torch.Tensor):
        pos_o = torch.log1p(o[o > 0])
        neg_o = -torch.log1p(torch.abs(o[o < 0]))
    else: 
        pos_o = np.log1p(o[o > 0])
        neg_o = -np.log1p(np.abs(o[o < 0]))
    if inplace:
        o[o > 0] = pos_o
        o[o < 0] = neg_o
        return o
    else:
        if hasattr(o, "clone"): output = o.clone()
        elif hasattr(o, "copy"): output = o.copy()
        output[output > 0] = pos_o
        output[output < 0] = neg_o
        return output

# %% ../nbs/002_utils.ipynb 162
def to_sincos_time(arr, max_value):
    sin = np.sin(arr / max_value * 2 * np.pi)
    cos = np.cos(arr / max_value * 2 * np.pi)
    return sin, cos

# %% ../nbs/002_utils.ipynb 164
def plot_feature_dist(X, percentiles=[0,0.1,0.5,1,5,10,25,50,75,90,95,99,99.5,99.9,100]):
    for i in range(X.shape[1]):
        ys = []
        for p in percentiles:
            ys.append(np.percentile(X[:, i].flatten(), p))
        plt.plot(percentiles, ys)
        plt.xticks(percentiles, rotation='vertical')
        plt.grid(color='gainsboro', linewidth=.5)
        plt.title(f"var_{i}")
        plt.show()

# %% ../nbs/002_utils.ipynb 166
def rolling_moving_average(o, window=2):
    if isinstance(o, torch.Tensor):
        cunsum = torch.cumsum(o, axis=-1) # nancumsum not available (can't be used with missing data!)
        lag_cunsum = torch.cat([torch.zeros((o.shape[0], o.shape[1], window), device=o.device), torch.cumsum(o[..., :-window], axis=-1)], -1)
        count = torch.clip(torch.ones_like(o).cumsum(-1), max=window)
        return (cunsum - lag_cunsum) / count
    else:
        cunsum = np.nancumsum(o, axis=-1)
        lag_cunsum = np.concatenate([np.zeros((o.shape[0], o.shape[1], window)), np.nancumsum(o[..., :-window], axis=-1)], -1)
        count = np.minimum(np.ones_like(o).cumsum(-1), window)
        return (cunsum - lag_cunsum) / count

# %% ../nbs/002_utils.ipynb 168
def ffill_sequence(o):
    """Forward fills an array-like object alongside sequence dimension"""
    if isinstance(o, torch.Tensor):
        mask = torch.isnan(o)
        idx = torch.where(~mask, torch.arange(mask.shape[-1], device=o.device), 0)
        idx = torch.cummax(idx, dim=-1).values
        return o[torch.arange(o.shape[0], device=o.device)[:,None,None], torch.arange(o.shape[1], device=o.device)[None,:,None], idx]
    else:
        mask = np.isnan(o)
        idx = np.where(~mask, np.arange(mask.shape[-1]), 0)
        idx = np.maximum.accumulate(idx, axis=-1)
        return o[np.arange(o.shape[0])[:,None,None], np.arange(o.shape[1])[None,:,None], idx]

def bfill_sequence(o):
    """Backward fills an array-like object alongside sequence dimension"""
    if isinstance(o, torch.Tensor):
        o = torch.flip(o, (-1,))
        o = ffill_sequence(o)
        return torch.flip(o, (-1,))
    else:
        o = o[..., ::-1]
        o = ffill_sequence(o)
        return o[..., ::-1]

def fbfill_sequence(o):
    """Forward and backward fills an array-like object alongside sequence dimension"""
    o = ffill_sequence(o)
    o = bfill_sequence(o)
    return o

# %% ../nbs/002_utils.ipynb 173
def dummify(o:Union[np.ndarray, torch.Tensor], by_var:bool=True, inplace:bool=False, skip:Optional[list]=None, random_state=None):
    """Shuffles an array-like object along all dimensions or dimension 1 (variables) if by_var is True."""
    if not inplace: 
        if isinstance(o, np.ndarray): o_dummy = o.copy()
        elif isinstance(o, torch.Tensor): o_dummy = o.clone()
    else: o_dummy = o
    if by_var:
        for k in progress_bar(range(o.shape[1]), leave=False):
            if skip is not None and k in listify(skip): continue
            o_dummy[:, k] = random_shuffle(o[:, k].flatten(), random_state=random_state).reshape(o[:, k].shape)
    else:
        o_dummy[:] = random_shuffle(o.flatten(), random_state=random_state).reshape(o.shape)
    if not inplace: 
        return o_dummy

# %% ../nbs/002_utils.ipynb 176
def shuffle_along_axis(o, axis=-1, random_state=None):
    if isinstance(o, torch.Tensor): size = o.numel()
    else: size = np.size(o)
    for ax in listify(axis):
        idx = random_shuffle(np.arange(size), random_state=random_state).reshape(*o.shape).argsort(axis=ax)
        o = np.take_along_axis(o, idx, axis=ax)
    return o

# %% ../nbs/002_utils.ipynb 178
def analyze_feature(feature, bins=100, density=False, feature_name=None, clip_outliers_plot=False, quantile_range=(25.0, 75.0), 
           percentiles=[1, 25, 50, 75, 99], text_len=12, figsize=(10,6)):
    non_nan_feature = feature[~np.isnan(feature)]
    nan_perc = np.isnan(feature).mean()
    print(f"{'dtype':>{text_len}}: {feature.dtype}")
    print(f"{'nan values':>{text_len}}: {nan_perc:.1%}")
    print(f"{'max':>{text_len}}: {np.nanmax(feature)}")
    for p in percentiles:
        print(f"{p:>{text_len}.0f}: {get_percentile(feature, p)}")
    print(f"{'min':>{text_len}}: {np.nanmin(feature)}")
    min_outliers, max_outliers = get_outliers_IQR(feature, quantile_range=quantile_range)
    print(f"{'outlier min':>{text_len}}: {min_outliers}")
    print(f"{'outlier max':>{text_len}}: {max_outliers}")
    print(f"{'outliers':>{text_len}}: {((non_nan_feature < min_outliers) | (non_nan_feature > max_outliers)).mean():.1%}")
    print(f"{'mean':>{text_len}}: {np.nanmean(feature)}")
    print(f"{'std':>{text_len}}: {np.nanstd(feature)}")
    print(f"{'normal dist':>{text_len}}: {normaltest(non_nan_feature, axis=0, nan_policy='propagate')[1] > .05}")
    plt.figure(figsize=figsize)
    if clip_outliers_plot:
        plt.hist(np.clip(non_nan_feature, min_outliers, max_outliers), bins, density=density, color='lime', edgecolor='black')
    else: 
        plt.hist(non_nan_feature, bins, density=density, color='lime', edgecolor='black')
    plt.axvline(min_outliers, lw=1, ls='--', color='red')
    plt.axvline(max_outliers, lw=1, ls='--', color='red')
    plt.title(f"feature: {feature_name}")
    plt.show()
    
def analyze_array(o, bins=100, density=False, feature_names=None, clip_outliers_plot=False, quantile_range=(25.0, 75.0), 
           percentiles=[1, 25, 50, 75, 99], text_len=12, figsize=(10,6)):
    if percentiles:
        percentiles = np.sort(percentiles)[::-1]
    print(f"{'array shape':>{text_len}}: {o.shape}")
    if o.ndim > 1:
        for f in range(o.shape[1]):
            feature_name = f"{feature_names[f]}" if feature_names is not None else f
            print(f"\n{f:3} {'feature':>{text_len - 4}}: {feature_name}\n")
            analyze_feature(o[:, f].flatten(), feature_name=feature_name)
    else:
        analyze_feature(o.flatten(), feature_name=feature_names)        

# %% ../nbs/002_utils.ipynb 181
def get_relpath(path):
    current_path = os.getcwd()
    if is_listy(path):
        relpaths = []
        for p in path:
            relpaths.append(os.path.relpath(p, current_path))
        return relpaths
    else:
        return os.path.relpath(path, current_path)

# %% ../nbs/002_utils.ipynb 182
def get_root():
    "Returns the root directory of the git repository."
    import subprocess
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return git_root


def to_root_path(path):
    "Converts a path to an absolute path from the root directory of the repository."
    if path is None:
        return path
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return Path(get_root()) / path

# %% ../nbs/002_utils.ipynb 183
def split_in_chunks(o, chunksize, start=0, shuffle=False, drop_last=False):
    stop = ((len(o) - start)//chunksize*chunksize) if drop_last else None
    chunk_list = []
    for s in np.arange(len(o))[start:stop:chunksize]:
        chunk_list.append(np.random.permutation(o[slice(s, s+chunksize)]) if shuffle else o[slice(s, s+chunksize)])
    if shuffle: random.shuffle(chunk_list)
    return chunk_list

# %% ../nbs/002_utils.ipynb 185
def save_object(o, file_path, verbose=True):
    file_path = Path(file_path)
    if not file_path.suffix == '.pkl':
        file_path = file_path.parent / (file_path.name + '.pkl')
    create_dir(file_path.parent, verbose)
    joblib.dump(o, file_path, )
    pv(f'{type(o).__name__} saved as {file_path}', verbose)
    
def load_object(file_path):
    file_path = Path(file_path)
    if not file_path.suffix == '.pkl':
        file_path = file_path.parent / (file_path.name + '.pkl')
    return joblib.load(file_path)

# %% ../nbs/002_utils.ipynb 188
def get_idxs_to_keep(o, cond, crit='all', invert=False, axis=(1,2), keepdims=False):
    idxs_to_keep = cond(o)
    if isinstance(o, torch.Tensor):
        axis = tuplify(axis)
        for ax in axis[::-1]: 
            if crit == 'all':
                idxs_to_keep = torch.all(idxs_to_keep, axis=ax, keepdim=keepdims)
            elif crit == 'any':
                idxs_to_keep = torch.any(idxs_to_keep, axis=ax, keepdim=keepdims)
        if invert: idxs_to_keep =  ~idxs_to_keep
        return idxs_to_keep
    else: 
        if crit == 'all':
            idxs_to_keep = np.all(idxs_to_keep, axis=axis, keepdims=keepdims)
        elif crit == 'any':
            idxs_to_keep = np.any(idxs_to_keep, axis=axis, keepdims=keepdims)
        if invert: idxs_to_keep = ~idxs_to_keep
        return idxs_to_keep

# %% ../nbs/002_utils.ipynb 190
def zerofy(a, stride, keep=False):
    "Create copies of an array setting individual/ group values to zero "
    if keep:
        a_copy = a.copy()[None]
    a = a[None]
    add_steps = np.int32(np.ceil(a.shape[2] / stride) * stride - a.shape[2])
    if add_steps > 0:
        a = np.concatenate([np.zeros((a.shape[0], a.shape[1], add_steps)), a], -1)
    a = a.repeat(a.shape[1] * a.shape[2] / stride, 0)
    a0 = np.arange(a.shape[0])[:, None]
    a1 = np.repeat(np.arange(a.shape[1]), a.shape[0] // a.shape[1])[:, None]
    a2 = np.lib.stride_tricks.sliding_window_view(np.arange(a.shape[-1]), stride, 0)[::stride]
    a2 = np.repeat(a2[None], stride * a.shape[0] / a.shape[2], axis=0).reshape(-1, stride)
    a[a0, a1, a2] = 0
    if add_steps > 0:
        a = a[..., add_steps:]
    if keep:
        return np.concatenate([a_copy, a])
    else: 
        return a

# %% ../nbs/002_utils.ipynb 192
def feat2list(o):
    if o is None: return []
    elif isinstance(o, str): return [o]
    return list(o)

# %% ../nbs/002_utils.ipynb 194
def smallest_dtype(num, use_unsigned=False):
    "Find the smallest dtype that can safely hold `num`"
    if use_unsigned:
        int_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
        float_dtypes = ['float16', 'float32']
        float_bounds = [2**11, 2**24] # 2048, 16777216
    else:
        int_dtypes = ['int8', 'int16', 'int32', 'int64']
        float_dtypes = ['float16', 'float32', 'float64']
        float_bounds = [2**11, 2**24, 2**53] # 2048, 16777216, 9007199254740992
    if isinstance(num, Integral):
        for dtype in int_dtypes:
            if np.iinfo(dtype).min <= num <= np.iinfo(dtype).max: 
                return np.dtype(dtype)
        raise ValueError("No dtype found")
    elif isinstance(num, float):
        for dtype, bound in zip(float_dtypes, float_bounds):
            num = round(num)
            if -bound <= num <= bound: 
                return np.dtype(dtype)
        raise ValueError("No dtype found")
    else:
        raise ValueError("Input is not a number")

# %% ../nbs/002_utils.ipynb 196
def plot_forecast(X_true, y_true, y_pred, sel_vars=None, idx=None, figsize=(8, 4), n_samples=1):
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    def _plot_forecast(X_true, y_true, y_pred, sel_var=None, idx=None, figsize=(8, 4)):
        if idx is None:
            idx = np.random.randint(0, len(X_true))
        if sel_var is None:
            title = f'sample: {idx}'
        else:
            title = f'sample: {idx} sel_var: {sel_var}'
        if sel_var is None: sel_var = slice(None)
        pred = np.concatenate([X_true[idx, sel_var], y_pred[idx, sel_var]], -1)
        pred[..., :X_true.shape[-1]] = np.nan

        true = np.concatenate([X_true[idx, sel_var], y_true[idx, sel_var]], -1)
        true_hist = true.copy()
        true_fut = true.copy()
        true_hist[..., X_true.shape[-1]:] = np.nan
        true_fut[..., :X_true.shape[-1]] = np.nan

        plt.figure(figsize=figsize)
        plt.plot(pred.T, color='orange', lw=1, linestyle='--')
        plt.plot(true_hist.T, color='purple', lw=1)
        plt.plot(true_fut.T, color='purple', lw=1, linestyle='--')
        plt.axvline(X_true.shape[-1] - 1, color='gray', lw=.5, linestyle='--')
        
        plt.title(title)
        plt.xlim(0, X_true.shape[-1] + y_true.shape[-1])
        pred_patch = mpatches.Patch(color='orange', label='pred')
        true_patch = mpatches.Patch(color='purple', label='true')
        plt.legend(handles=[true_patch, pred_patch], loc='best')
        plt.show()
        
    assert X_true.shape[:-1] == y_true.shape[:-1] == y_pred.shape[:-1]
    assert y_true.shape[-1] == y_pred.shape[-1]
    
    if idx is not None:
        idx = listify(idx)
        n_samples = len(idx)
        iterator = idx
    else:
        iterator = random_randint(len(X_true), size=n_samples)
    
    if sel_vars is None:
        for idx in iterator:
            _plot_forecast(X_true, y_true, y_pred, sel_var=None, idx=idx, figsize=figsize)
    else:
        for idx in iterator:
            if sel_vars is True:
                sel_vars = np.arange(y_true.shape[1])
            else:
                sel_vars = listify(sel_vars)
            for sel_var in sel_vars:
                _plot_forecast(X_true, y_true, y_pred, sel_var=sel_var, idx=idx, figsize=figsize)
