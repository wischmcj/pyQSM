from pyQSM.canopy_metrics import (
    loop_over_files, np_feature,  width_at_height, np_to_o3d, read_pcd

)

from pyQSM.io import read_pcd
import re

from pyQSM.utils.io import load

if __name__ =="__main__":
    ##### Orig SKIO ext
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/'
    loop_over_files(width_at_height,
                    requested_seeds= ['190'],
                    parallel=False,
                    base_dir=base_dir,
                    data_file_config={ 
                        'src': {
                                'folder': 'ext_detail',
                                'file_pattern': f'full_ext_*_orig_detail.pcd',
                                'load_func': read_pcd, 
                            },
                    },
                    seed_pat = re.compile('.*seed([0-9]{1,3}).*'),
                    )
    skip_seeds = [  'skio_0_tl_306','skio_0_tl_157', 'skio_0_tl_377','skio_0_tl_49',
                    'skio_0_tl_167', 'skio_0_tl_158', 'skio_0_tl_23', 
                    'skio_0_tl_14', 'skio_0_tl_191', 'skio_0_tl_154', 'skio_0_tl_512',
                    # The below may be worth rerunning shift calc without vox
                    'skio_0_tl_192', 'skio_0_tl_307',
                    'skio_0_tl_669','skio_0_tl_163',
                    'skio_0_tl_202', 'skio_0_tl_297',]
    ###### SKIO/tl joins
    base_dir = '/media/penguaman/backupSpace/lidar_sync/pyqsm/skio/cluster_joining/'
    loop_over_files(width_at_height,
                    skip_seeds= skip_seeds,
                    parallel=False,
                    base_dir=base_dir,
                    data_file_config={ 
                        'src': {
                                'folder': 'detail',
                                'file_pattern': f'skio_*_tl_*.npz',
                                'load_func': np_to_o3d, 
                            },
                        'shift_one': {
                                'folder': 'shifts',
                                'file_pattern': 'skio_*_tl_*_shift.pkl',
                                'load_func': lambda x,root_dir: load(x,root_dir)[0], 
                                'kwargs': {'root_dir': '/'},
                            },
                        'intensity': {
                                'folder': 'detail',
                                'file_pattern': f'skio_*_tl_*.npz',
                                'load_func': np_feature('intensity'), # custom, for pickles
                            },
                    },
                    seed_pat = re.compile('.*(skio_[0-9]{1,3}_tl_[0-9]{1,3}).*')
    )