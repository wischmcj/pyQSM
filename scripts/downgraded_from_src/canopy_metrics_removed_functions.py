
def get_trunk(file_content):
    breakpoint()
    seed, pcd, clean_pcd, shift_one = file_content
    # file_base = f'skels3/skel_{str(contraction).replace('.','pt')}_{str(attraction).replace('.','pt')}_seed{seed}' #_vox{vox}_ds{ds}'0
    lowc,highc = split_on_pct(clean_pcd,70,shift=shift_one)
    draw([lowc])
    breakpoint()
    stem_pcd = get_stem_pcd(pcd=lowc)
    breakpoint()



def contraction_analysis(file, pcd, shift):

    green = get_green_surfaces(pcd)
    not_green = get_green_surfaces(pcd,True)
    # draw(lowc_pcd)
    draw(green)
    draw(not_green)

    c_mag = np.array([np.linalg.norm(x) for x in shift])
    z_mag = np.array([x[2] for x in shift])

    highc_idxs, highc, lowc = split_on_percentile(pcd,c_mag,70, color_on_percentile=True)

    z_cutoff = np.percentile(z_mag,80)
    log.info(f'{z_cutoff=}')
    low_idxs = np.where(z_mag<=z_cutoff)[0]
    lowc = clean_pcd.select_by_index(low_idxs)
    ztrimmed_shift = shift[low_idxs]
    ztrimmed_cmag = c_mag[low_idxs]
    draw(lowc)
    highc_idxs, highc, lowc = split_on_percentile(lowc,c_mag,70, color_on_percentile=True)

    # color_continuous_map(test,c_mag)
    highc_idxs = np.where(c_mag>np.percentile(c_mag,70))[0]
    highc_pcd = test.select_by_index(highc_idxs)
    lowc_pcd = test.select_by_index(highc_idxs,invert=True)
    draw([lowc_pcd])
    draw([highc_pcd])

    breakpoint()
    
    lowc_detail = get_neighbors_kdtree(pcd,lowc_pcd)
    draw(lowc_detail)
    breakpoint()



def draw_skel(pcd, 
                seed,
                shift=[], 
                down_sample=True,
                skeleton = None,
                save_gif=False, 
                out_path = None,
                on_frames=25,
                off_frames=25,
                addnl_frame_duration=.05,
                point_size=5):
    out_path = out_path or f'data/results/gif/{seed}'
    clean_pcd = pcd
    if down_sample: clean_pcd = get_downsample(pcd=pcd, normalize=False)
    if not skeleton:
        skeleton = contract(clean_pcd,shift)
    draw(skeleton)
    # center_and_rotate(skeleton)
    # center_and_rotate(clean_pcd)
    # _ = center_and_rotate(skeleton) #Rotation makes contraction harder
    gif_kwargs = {'on_frames': on_frames,'off_frames': off_frames, 'addnl_frame_duration':addnl_frame_duration,'point_size':point_size,'save':save_gif,'out_path':out_path}
    # rotating_compare_gif(skeleton,clean_pcd, **gif_kwargs)
    # rotating_compare_gif(contracted,clean_pcd, point_size=4, output=out_path,save = save_gif, on_frames = 50,off_frames=50, addnl_frame_duration=.03)
    return skeleton