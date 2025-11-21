
######### Recovering intensity and color from original scan.
######### tl scans need to be translated by the mean of all of the points ( np.array([113.42908253, 369.58605156,   4.60074394]))
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/'
    remaining_files, remaining_keys = compare_dirs(base_dir + 'detail/', base_dir + 'color_int_tree_nbrs/',
                                file_pat1 = 'skio_*_tl_*.npz', file_pat2 = 'detail_feats_*.npz',
                                key_pat1 = '(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*', key_pat2 = '(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
    
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining'
    files = glob(f'{base_dir}/to_get_detail/*_joined.npz')
    # seed_pat = '.*seed([0-9]{1,3}).*'
    seed_pat = '.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*'
    # seed_pat = '.*/([0-9]{1,3})_joined_2.*'
    # files = ['/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/skio_193_tl_241_joined.npz']
    for file in files:
        print(f'{file=}')
        comp_file_name = re.match(re.compile(seed_pat),file).groups(1)[0]
        # comp_file_name = f'skio_0_tl_{comp_file_name}'
        log.info(f'processing {comp_file_name}')
        tile_dir = '/media/penguaman/backupSpace/lidar_sync/tls_lidar/SKIO/'
        tile_pattern = 'SKIO-RaffaiEtAlcolor_int_*.npz'
        # comp_pcd = read_pcd(file)
        comp_pcd = to_o3d(coords=np.load(file)['points'] ) # + np.array([113.42908253, 369.58605156,   4.60074394])
        get_nbrs_voxel_grid(comp_pcd,
                        comp_file_name,
                        tile_dir = tile_dir,
                        tile_pattern = tile_pattern,
                        invert=False,
                        out_folder= f'{base_dir}/detail')