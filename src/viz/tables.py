import __future__ as future
from itertools import product
from prettytable import PrettyTable
from set_config import log, config

def create_table(
                results:list[dict] | list[tuple[str,dict[str,dict]]] | dict[str,dict[str,dict]]
                 ,cols=None,sub_cols=None,ids=None):
    if cols is None:
        cols = [x for x in results[0].keys()]
    if sub_cols is None:
        try:
            sub_cols = results[0][cols[0]].keys()
        except:
            log.info(f'no sub_cols found for {cols[0]}')
            sub_cols = results[0].keys()
    all_cols =list([x for x in product(cols, sub_cols)])

    col_names = [f'{col}_{sub_col}' for col, sub_col in all_cols]
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

    for row_id, row_data in results.items():
        row = [row_data[col] for col in col_names]
        if sub_cols is not None:
            row = [row[col] for col in sub_cols]
        myTable.add_row([row_id] + row)
    print(myTable)
    return myTable