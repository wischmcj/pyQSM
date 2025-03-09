

def save_line_set(line_set, base_file = 'skel_stem20_topology'):
    import pickle 
    with open(f'{base_file}_lines.pkl','wb') as f:     pickle.dump(np.asarray(line_set.lines),f)
    with open(f'{base_file}_points.pkl','wb') as f:     pickle.dump(np.asarray(line_set.points),f)
    
def load_line_set(base_file = 'skel_stem20_topology'):
    import pickle 
    with open(f'{base_file}_lines.pkl','rb') as f: lines = pickle.load(f)
    with open(f'{base_file}_points.pkl','rb') as f: points = pickle.load(f)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set
