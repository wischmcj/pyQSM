
def chunk_iter(iter, 
                chunk_size, 
                condition= lambda x:1==1,
                breakpt:bool = True, 
                ignore_ranges = []):
    interesting_intervals = []
    chunks = (len(iter)-len(iter)%chunk_size)/chunk_size
    for i in range(int(chunks)):
        frm=i*chunk_size
        to=(i+1)*chunk_size
        if (frm,to) not in ignore_ranges:
            if to>len(iter): 
                to = len(iter)
            for x in iter[frm:to]: x.paint_uniform_color([1,0,0])
            if condition(iter[frm:to]):
                if breakpt: breakpoint()
                interesting_intervals.append((frm,to))
    return interesting_intervals

def in_intervals(item,intervals):
    return any([a<=item<=b for a,b in intervals])