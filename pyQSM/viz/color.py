from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np
from numpy import array as arr, mean
import scipy.spatial as sps
import  open3d as o3d
from open3d.io import read_point_cloud as read_pcd, write_point_cloud as write_pcd

from utils.io import load
from viz.viz_utils import color_continuous_map, draw
from set_config import log, config


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors   
import cv2
from math import floor
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv


def homog_colors(pcd):
    """
        Replaces the color of white points with the average color of their neighbors
        Intended to deal with bloom from overly reflective surfaces
    """
    colors = arr(pcd.colors)
    pts = arr(pcd.points)
    white_idxs = [idc for idc,color in enumerate(colors) if sum(color)>2.7]
    white_pts = [pts[idc] for idc in white_idxs]

    non_white_tree = pcd.select_by_index(white_idxs, invert=True)
    tree = sps.KDTree(arr(non_white_tree.points))
    white_nbrs = tree.query([white_pts],30)
    avg_neighbor_color = mean(colors[ white_nbrs[1]][0],axis=1)
    for white_idx,avg_color in zip(white_idxs,avg_neighbor_color):  colors[white_idx] = avg_color
    pcd.colors =  o3d.utility.Vector3dVector(colors)
    draw(pcd)
    
def remove_color_pts(pcd,
                        color_lambda = lambda x: sum(x)>2.7,
                        invert=False):
    colors = arr(pcd.colors)
    ids = [idc for idc, color in enumerate(colors) if color_lambda(color)]
    new_pcd = pcd.select_by_index(ids, invert = invert)
    return new_pcd

def get_green_surfaces(pcd,invert=False):
    green_pcd= remove_color_pts(pcd,    lambda rgb: rgb[1]>rgb[0] and rgb[1]>rgb[2] and 0.5<(rgb[0]/rgb[2])<2, invert)
    return green_pcd

def mute_colors(pcd):
    colors = arr(pcd.colors)
    rnd_colors = [[round(a,2) for a in col] for col in colors]
    pcd.colors= o3d.utility.Vector3dVector(rnd_colors)
    draw(pcd)

def bin_colors(zoom_region=[(97.67433, 118.449), (342.1023, 362.86)]):
    # binning colors in pcd, finding most common
    # round(2) reduces 350k to ~47k colors
    region = [(pt,color) for pt,color in zip(low_pts,low_colors) 
                if (pt[0]>zoom_region[0][0] and pt[0]<zoom_region[0][1]       
                    and pt[1]>zoom_region[1][0] 
                    and  pt[1]<zoom_region[1][1])]

    cols = [','.join([f'{round(y,1)}' for y in x[1]]) for x in region]
    d_cols, cnts = np.unique(cols, return_counts=True)
    print(len(d_cols))
    print((max(cnts),min(cnts)))
    # removing points of any color other than the most common
    most_common = d_cols[np.where(cnts)]
    most_common_rgb = [tuple((float(num) for num in col.split(','))) for col in most_common]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) in most_common_rgb ]
    color_limited = [(pt,tuple((round(y,1)for y in color))) for pt,color in region if tuple((round(y,1)for y in color)) != tuple((1.0,1.0,1.0)) ]
    limited_pcd = o3d.geometry.PointCloud()
    limited_pcd.points = o3d.utility.Vector3dVector(arr([x[0] for x in color_limited]))
    limited_pcd.colors = o3d.utility.Vector3dVector(arr([x[1] for x in color_limited]))
    draw(limited_pcd)


def color_compare(in_colors_one, in_colors_two,color_conds, cutoff=.01,elev=40, azim=110, roll=0, 
                space='none',min_s=.2,sat_correction=2,
                sc_func =lambda sc: sc + (1-sc)/2, in_rgb = False ):
    color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
    hsv_one = arr(rgb_to_hsv(in_colors_one))
    hsv_two = arr(rgb_to_hsv(in_colors_two))
    if in_rgb:
        hsv_one=in_colors_one
        hsv_two=in_colors_two

    rands = np.random.sample(len(in_colors_one))
    in_colors_one = arr(in_colors_one)[rands<cutoff]
    hsv_one = arr(hsv_one)[rands<cutoff]

    rands = np.random.sample(len(in_colors_two))
    in_colors_two = arr(in_colors_two)[rands<cutoff]
    hsv_two = arr(hsv_two)[rands<cutoff]
   
    # hsv_two[:,2] = hsv_two
    
    hco,sco,vco = zip(*hsv_one)
    hct,sct,vct = zip(*hsv_two)

    # breakpoint()
    fig = plt.figure(figsize=(12, 9))
    # row=row+1
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    # breakpoint()
    axis.scatter(hco, sco, vco, facecolors=in_colors_one, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(elev=elev, azim=azim, roll=roll)

    axis = fig.add_subplot(1, 2,2, projection="3d")
    axis.scatter(hct, sct, vct, facecolors=in_colors_two, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(elev=elev, azim=azim, roll=roll)
    plt.show()

def saturate_colors(pcd, min_s=1,sc_func =lambda sc: sc + (1-sc)/3):
    """
        Calls color distribution, which applies translates
        to hsv space, applies the sc_func to saturation and the
        converts back to rgb.
    """
    target = pcd
    orig_colors = arr(target.colors)
    log.info(f'Correcting colors')
    corrected_colors, sc = color_distribution(arr(target.colors),min_s=min_s,sc_func =sc_func)
    target.colors = o3d.utility.Vector3dVector(corrected_colors)
    return target, orig_colors

def segment_hues(pcd, seed, hues=['white','blues','pink','red_yellow', 'greens'],
                    draw_gif=False, down_sample=False,
                    draw_results=False, save_gif=False,
                    on_frames=25, off_frames=25, 
                    addnl_frame_duration=.01, point_size=5):
    log.info(f"Started 'segmenting_hues'")
    color_conds = {        'white' : lambda tup: tup[0]>.5 and tup[0]<5/6 and tup[2]>.5 ,'pink' : lambda tup:  tup[0]>=.7 and tup[2]>.3 ,'blues' : lambda tup:  tup[0]<.7 and tup[0]>.4 and tup[2]>.4 ,'subblue' : lambda tup:  tup[0]<.4 and tup[0]>.4 and tup[2]>.3 ,'greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.2 ,'light_greens' : lambda tup: tup[0]<=.5 and tup[0]>2/9 and tup[2]>.5 ,'red_yellow' : lambda tup:  tup[0]<=2/9 and tup[2]>.3}
    target,orig_colors = saturate_colors(pcd)
    # if draw_gif: center_and_rotate(target)
    # breakpoint()
    
    hue_pcds = [None]*(len(hues)+2)
    no_hue_pcds = [None]*(len(hues)+2)
    hue_idxs = [None]*(len(hues)+1)
    no_hue_pcds[0] = target
    hue_pcds[0] =target
    log.info(f'Segmenting')
    to_run = target
    greens_pcd= None
    for tup in enumerate(hues): 
        idh, hue = tup
        try:
            hue_pcds[idh+1],no_hue_pcds[idh+1],hue_idxs[idh] = get_color_by_hue(to_run, color_conds[hue])
            if len(hue_idxs[idh])>0:
                to_run = no_hue_pcds[idh+1]
            if hue =='greens': greens_pcd = hue_pcds[idh+1]
        except Exception as e:
            breakpoint()
            # idh, hue=[x for x in   enumerate(hues)]
            log.error(f'Error segmenting hues {e}')
    hue_pcds = [x for x in hue_pcds if x is not None]
    no_hue_pcds = [x for x in no_hue_pcds if x is not None]
    # if draw_gif:
    #     try:
    #         final = no_hue_pcds[len(no_hue_pcds)-1]
    #         gif_kwargs = { 'sub_dir':f'{seed}_segment_hues_final' ,'on_frames': on_frames, 'off_frames': off_frames, 'addnl_frame_duration':addnl_frame_duration, 'point_size':point_size, 'save':save_gif, 'rot_center':final.get_center()}
    #         target.colors = o3d.utility.Vector3dVector(orig_colors)
    #         # rotating_compare_gif(target,final,**gif_kwargs)
    #         # if greens_pcd is not None:
    #         #     gif_kwargs['sub_dir']=f'{seed}_segment_hues_greens' 
    #         #     rotating_compare_gif(greens_pcd,final,**gif_kwargs)
    #     except Exception as e:
    #         breakpoint()
    #         # idh, hue=[x for x in   enumerate(hues)]
    #         log.error(f'Error segmenting hues {e}')
    # sizes = [len(arr(x.points)) for x in hue_pcds[1:]]
    # hue_by_idhs = {hue: idhs for hue,idhs in zip(hues,hue_idxs)}
    return hue_pcds,no_hue_pcds

def get_color_by_hue(pcd,condition):
    target = pcd
    orig_colors = arr(target.colors)
    hsv = arr(rgb_to_hsv(arr(target.colors))) #tup[0]<.5 or tup[0]>5/6 or 
    hue_idxs,in_vals  = [x for x in zip(*[(idt,tup) for idt, tup in enumerate(hsv) if condition(tup) ])]
    target.colors = o3d.utility.Vector3dVector(orig_colors)
    hue = target.select_by_index(hue_idxs,invert=False)
    no_hue = target.select_by_index(hue_idxs,invert=True)
    # draw(hue)
    # draw(no_hue)
    return hue, no_hue, hue_idxs

def isolate_color(in_colors,icolor='white',get_all=True, std='hsv'):
    ################# NOT WORKING RN #############
    color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],'white': [[180, 18, 255], [0, 0, 231]],'red1': [[180, 255, 255], [159, 50, 70]],'red2': [[9, 255, 255], [0, 50, 70]],'green': [[89, 255, 255], [36, 50, 70]],'blue': [[128, 255, 255], [90, 50, 70]],'yellow': [[35, 255, 255], [25, 50, 70]],'purple': [[158, 255, 255], [129, 50, 70]],'orange': [[24, 255, 255], [10, 50, 70]],'gray': [[180, 18, 230], [0, 0, 40]]}        
    if std == 'hsv':
        hsv = arr(rgb_to_hsv(in_colors))
        color_dict_HSV ={'white':[[2/3,2/3,1],[0,0,0]], 'black': [[1/6,1,1],[0,.75,.75]],'red': [[.7,1,1],[0,.5,0.5]],
                          'green': [[.5,1,1],[1/6,.5,.5]],'blue': [[5/6,1,1],[.5,.5,.5]],}
        # color_dict_HSV= {col:arr(rgb_to_hsv(arr(vals)/255)) for col,vals in color_dict_HSV.items()}
        # color_dict_HSV ={'white': [[0,0,1],[0,0,0.75]] ,'silver': [[0,0,0.75],[0,0,0.5]] ,'gray': [[0,0,0.5],[0,0,0]] ,
                            # 'black': [[0,0,0],[0,1,1]] ,'red': [[0,1,1],[0,.5,0.5]] ,'maroon': [[0,1,0.5],[1/6,1,1]] ,'yellow': [[1/6,1,1],[1/6,1,0.5]] ,'olive': [[1/6,1,0.5],[1/3,1,1]] ,
        # 'lime': [[1/3,1,1],[1/3,1,0.5]] ,'green': [[1/3,1,0.5],[0.5,1,1]] ,
        # 'aqua': [[0.5,1,1],[0.5,1,0.5]] ,'teal': [[0.5,1,0.5],[2/3,1,1]] ,
        # 'blue': [[2/3,1,1],[2/3,1,0.5]] ,'navy': [[2/3,1,0.5],[5/6,1,1]] ,'fuchsia': [[5/6,1,1],[5/6,1,0.5]] ,'purple': [[5/6,1,0.5],[100,1,1]]}
        # color_dict_HSV = {col:[[vals[0][0],1,1],[vals[1][0],0,0]] for col,vals in color_dict_HSV.items()}
    
    else:
        hsv = arr(rgb_to_hsv(in_colors))
        hsv = in_colors*255

    if icolor not in [x for x in color_dict_HSV.keys()]: 
        raise ValueError(f'Color {icolor} not found in ranges, use one of {clist}')
    clist=[x for x in color_dict_HSV.keys()]
    ret={}
    in_range=  lambda tup,ub,lb: all([lbv<val<ubv for val,lbv,ubv in zip(tup,ub,lb)])
    test = []
    for color,(ub,lb) in color_dict_HSV.items(): #test.append({color:{'ids':idt,'cols':arr(tup)} for idt,tup in zip(*[(idt,tup) for idt, tup in enumerate(hsv) if in_range(tup,lb,ub)])})
        if get_all or color ==icolor: 
            in_hsv = [x for x in zip(*[(idt,tup) for idt, tup in enumerate(hsv) if in_range(tup,lb,ub)])]
            if len(in_hsv)>0:
                in_ids,in_vals = in_hsv
                color_details = {color: {'ids':in_ids,'cols':arr(in_vals)}}
                ret.update(color_details)
    for mydict in test: ret.update(mydict)
    # new_hsv.append(idc) 
    # new_hsv= hsv[np.logical_and(hsv<upper,hsv>lower)]
    # rgb = arr(hsv_to_rgb(new_hsv))
    log.info({col:len(x['ids']) for col, x in ret.items()})
    return ret

    non_white_idxs = remove_color(low_color)
    
    # center_and_rotate(clean_pcd)
    # gif_kwargs = {'on_frames': on_frames,'off_frames': off_frames, 'addnl_frame_duration':addnl_frame_duration,'point_size':point_size,'save':save_gif,'out_path':out_path}
    # rotating_compare_gif(skeleton,clean_pcd, **gif_kwargs)
    
    breakpoint()

def color_distribution(in_colors,oth_colors=None,cutoff=1,elev=40, azim=110, roll=0, 
                space='none',min_s=.2,sc_func =lambda sc: sc + (1-sc)/3):
    
    color_lists = [in_colors]
    if oth_colors is not None:
        color_lists.append(oth_colors)
    hsv_fulls = []
    hsvs = []
    hsv_news = []
    corrected_rgb_fulls = []
    for idc,color_list in enumerate(color_lists):
        hsv = arr(rgb_to_hsv(color_list))
        hsv_fulls.append(hsv)
        if idc ==0:
            rands = np.random.sample(len(color_list))
            color_list = arr(color_list)[rands<cutoff]
            hsv = arr(hsv)[rands<cutoff]
        data = []
        hc,sc,vc = zip(*hsv)
        sc = arr(sc)
        vc = arr(vc)
        low_saturation_idxs = np.where(sc<min_s)[0]
        # sc[sc<min_s] = sc[sc<min_s]*sat_correction
        ret_sc = [sc[i] if i not in low_saturation_idxs else sc_func(sc[i]) for i in range(len(sc))]
        # ret_sc = sc_func(sc)
        # vc =   sc_func(vc)
        # sc = sc*.6
        corrected_rgb_full = arr(hsv_to_rgb([x for x in zip(hc,ret_sc,vc)]))
        corrected_rgb_fulls.append(corrected_rgb_full)
        hsv_news.append([x for x in zip(hc,ret_sc,vc)])

        hsvs.append(hsv)
    
    # for idr, row_hsv in enumerate(hsv_fulls):
    #     nbins = 20 
    #     colors_h  = np.linspace(0,1,nbins)
    #     colors_hsv=zip(colors_h,[.5]*20,[.5]*20)
    #     rgb = hsv_to_rgb([x for x in colors_hsv])
    #     hc,sc,vc = zip(*row_hsv)
    #     res = plt.hist(hc, bins=nbins,facecolor=rgb)
    #     plt.show()
#    15     lower_blue = np.array([110,50,50])
#    16     upper_blue = np.array([130,255,255])
#    17 
#    18     # Threshold the HSV image to get only blue colors
#    19     mask = cv2.inRange(hsv, lower_blue, upper_blue)

    
    # for sids,series in enumerate(data):
    #     data[sids] = arr(series)[rands<cutoff]
    # corrected_rgb =  arr(corrected_rgb_full)[rands<cutoff]
    # osc = arr(osc)[rands<cutoff]
    ## RGB
    if space=='rgb':
        pixel_colors = in_colors
        r, g, b = zip(*in_colors)
        # r, g, b = cv2.split(pixel_colors)
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.scatter(r, g, b, facecolors=in_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()

    # HSV 
    if space=='hsv':
        # hsv = hsv[hsv[:,1]>min_s]
        import math
        # sc[sc<.5] = sc[sc<.5]*1.5
        fig = plt.figure(figsize=(12, 9))
        for idr, row_hsv in enumerate(hsvs):
            hc,sc,vc = zip(*row_hsv)
            # hcn,scn,vcn = zip(*hsv_news[idr])
            # row=row+1
            axis = fig.add_subplot(2, 1, idr+1, projection="3d")
            # breakpoint()
            axis.scatter(hc, sc, vc, facecolors=corrected_rgb_fulls[idr], marker=".")
            axis.set_xlabel("Hue")
            axis.set_ylabel("Saturation")
            axis.set_zlabel("Value")
            axis.view_init(elev=elev, azim=azim, roll=roll)

            # axis = fig.add_subplot(2, 2, idr+2, projection="3d")
            # axis.scatter(hcn, scn, vcn, facecolors=corrected_rgb_fulls[idr], marker=".")
            # axis.set_xlabel("Hue")
            # axis.set_ylabel("Saturation")
            # axis.set_zlabel("Value")
            # axis.view_init(elev=elev, azim=azim, roll=roll)
        plt.show()
    
    return corrected_rgb_full,hsv_fulls

def split_on_percentile(pcd,
                        val_list,
                        pctile,
                        comp=lambda x,y:x>y,
                        color_on_percentile=False):
    if len(pcd.points)!= len(val_list):
        msg = f'length of val list does not match size of pcd'
        log.error(f'length of val list does not match size of pcd')
        raise ValueError(msg)
    if color_on_percentile:
        color_continuous_map(pcd,val_list)
    val = np.percentile(val_list,pctile)
    highc_idxs = np.where(comp(val_list,val))[0]
    highc_pcd = pcd.select_by_index(highc_idxs,invert=False)
    lowc_pcd = pcd.select_by_index(highc_idxs,invert=True)
    return highc_idxs, highc_pcd,lowc_pcd
