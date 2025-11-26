def tensor_board_color_analysis(file_content):
    log.info('identifying epiphytes')
    # user_input = 65
    # while user_input is not None:
    seed, pcd, clean_pcd, shift_one = file_content['seed'], file_content['src'], file_content['clean_pcd'], file_content['shift_one']
    logdir = "/media/penguaman/writable/lidar_sync/py_qsm/tensor_board/id_epi"
    writer = tf.summary.create_file_writer(logdir)
    params = [(lambda sc: sc + (1-sc)/3, 1, '33 inc, 1x'), 
                (lambda sc: sc + (1-sc)/2, 1, '50 inc, 1x'),
                (lambda sc: sc + (1-sc)/3, 1.5, '33 inc, 1.5x'), 
                (lambda sc: sc + (1-sc)/2, 1.5, '50 inc, 1.5x'),
                (lambda sc: sc + (1-sc)/3, .5, '33 inc, .5x'), 
                (lambda sc: sc + (1-sc)/2, .5, '50 inc, .5x')
                ]

    with writer.as_default():
        for sat_func, sat_cutoff, case_name in params:
            sat_pcd, sat_orig_colors = saturate_colors(pcd, cutoff=sat_cutoff, sc_func=sat_func)
            step+=1
            summary.add_3d('sat_test', to_dict_batch([sat_pcd]), step=step, logdir=logdir)