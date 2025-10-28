import pickle
from glob import glob
from itertools import product
from prettytable import PrettyTable

def create_table(
                results:list[dict] | list[tuple[str,dict[str,dict]]] | dict[str,dict[str,dict]]
                 ,cols=None,sub_cols=None,ids=None):
    if cols is None:
        if isinstance(results,dict):
            cols = [x for x in results.keys()]
        else:
            row1 = results[0]
            if isinstance(row1,tuple):
                cols = list(row1[1].keys())
            elif isinstance(row1,dict):
                cols = list(row1.keys())
            else:
                cols = []
        print(f'{cols=}')

    if sub_cols is not None:
        all_cols =list([x for x in product(cols, sub_cols)])
        col_names = [f'{col}_{sub_col}' for col, sub_col in all_cols]
    else:
        col_names = cols
    print(f'{col_names=}')
    
    fin_cols = ['ID'] + col_names
    myTable = PrettyTable(fin_cols)

    if ids is not None:
        if isinstance(results[0],tuple):
            ids = [x[0] for x in results]
            rows = [x[1] for x in results]
            results = dict(zip(ids,rows))
        elif isinstance(results,dict):
            ids = [x for x in results.keys()]
        elif isinstance(results,list):
            results = dict([(idx,x) for idx,x in enumerate(results)])

    for row_id, row_data in results:
        row = [row_data[col] for col in col_names]
        if sub_cols is not None:
            row = [row[col] for col in sub_cols]
        myTable.add_row([row_id] + row)
    print(myTable)
    return myTable

base_dir = f'/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/'
# sub_dir = 'ext_detail/projected_areas_prod'
sub_dir = 'cluster_joining/projected_areas/'
files = glob(f'{base_dir}/{sub_dir}/*/all_metrics*.pkl')
data = {}
for file in files:
    seed = file.split('/')[-2]
    with open(file, 'rb') as f:
        data[seed] = pickle.load(f)
final_data = []
for seed, data in data.items():
    print(f'{seed=}')
    leaf_area_std = data['epis'] + data['leaves'] + data['wood']
    total_area = data.get('total_area')
    whole_area = data['whole']
    lai_std = leaf_area_std/whole_area
    lai_adj = (data['wood'] + data['leaves'])/whole_area
    eai = (data['epis'])/whole_area
    
    final_data.append((seed, { #'wood': data.get('wood_singular'),
                        # 'Tree ID': seed, 
                        'Epiphyte Area': data['epis'],
                        'Leaf Area': data['leaves'],
                        'Wood (Overlap)': data['wood'],
                        'Total Area': total_area,
                        'Whole Area': whole_area,
                        'LAI (Standard)': lai_std,
                        'LAI (Proposed)': lai_adj,
                        'EAI': eai,}))
myTable = create_table(final_data)
print(myTable)
breakpoint()


###### Notes on first skio/tl join projection run
# i - additional isolation
# e - extend further 
# c - fix clustering (both of the above)
# s - correct segmentation
# n - none needed, good
skio_tl_first_run_corrections: { 'skio_0_tl_6': '',
                            'skio_0_tl_15': '',
                            'skio_0_tl_69': '',
                            'skio_0_tl_188': '',
                            'skio_0_tl_223': '',
                            'skio_0_tl_240': '',
                            'skio_0_tl_246': '',
                            'skio_0_tl_490': '',
                            'skio_33_tl_0': '',
                            'skio_107_tl_0': '',
                            'skio_108_tl_0': '',
                            'skio_111_tl_0': '',
                            'skio_112_tl_0': '',
                            'skio_114_tl_0': '',
                            'skio_115_tl_0': '',
                            'skio_116_tl_493': '',
                            'skio_133_tl_68': '',
                            'skio_134_tl_0': '',
                            'skio_135_tl_0': '',
                            'skio_136_tl_9': '',
                            'skio_137_tl_34': '',
                            'skio_138_tl_0': '',
                            'skio_189_tl_236': '',
                            'skio_190_tl_220': '',
                            'skio_191_tl_236': '', 
                            }