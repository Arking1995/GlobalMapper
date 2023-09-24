import shapely.affinity as sa
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, MultiLineString
import numpy as np
import random
import networkx as nx
import copy
from sklearn.neighbors import NearestNeighbors


##########################################################################
def get_extend_line(a, b, block, isfront, is_extend_from_end = False):
    minx, miny, maxx, maxy = block.bounds
    if a.x == b.x:  # vertical line
        if a.y <= b.y:
            extended_line = LineString([a, (a.x, minx)])
        else:
            extended_line = LineString([a, (a.x, maxy)])
    elif a.y == b.y:  # horizonthal line
        if a.x <= b.x:
            extended_line = LineString([a, (minx, a.y)])
        else:
            extended_line = LineString([a, (maxx, a.y)])

    else:
        # linear equation: y = k*x + m
        k = (b.y - a.y) / (b.x - a.x)
        m = a.y - k * a.x
        if k >= 0:
            if b.x - a.x >= 0:
                y1 = k * minx + m
                x1 = (miny - m) / k
                points_on_boundary_lines = [Point(minx, y1), Point(x1, miny)]
            else:
                y1 = k * maxx + m
                x1 = (maxy - m) / k
                points_on_boundary_lines = [Point(maxx, y1), Point(x1, maxy)]        
        else:
            if b.x - a.x >= 0:
                y1 = k * minx + m
                x1 = (maxy - m) / k
                points_on_boundary_lines = [Point(minx, y1), Point(x1, maxy)]
            else:
                y1 = k * maxx + m
                x1 = (miny - m) / k
                points_on_boundary_lines = [Point(maxx, y1), Point(x1, miny)]

        # print('points on bound: ', points_on_boundary_lines[0].coords.xy, points_on_boundary_lines[1].coords.xy)
        
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=a.distance)
        extended_line = LineString([a, Point(points_sorted_by_distance[0])])
    

    min_dis = 9999999999.9
    intersect = block.boundary.intersection(extended_line)
    if intersect.geom_type == 'MultiPoint':
        for i in intersect:
            if i.distance(a) <= min_dis:
                nearest_points_on_contour = i
    elif intersect.geom_type == 'Point':
        nearest_points_on_contour = intersect
    # elif intersect.geom_type == 'LineString':
    #     if not is_extend_from_end:
    #         nearest_points_on_contour = a
    #     else:
    #         nearest_points_on_contour = b
    else:
        if not is_extend_from_end:
            nearest_points_on_contour = a
        else:
            nearest_points_on_contour = b
        print('intersect: ', intersect)
        print('unknow geom type on intersection: ', intersect.geom_type)

    if not is_extend_from_end:
        if isfront:
            line_to_contour = LineString([nearest_points_on_contour, a])
        else:
            line_to_contour = LineString([a, nearest_points_on_contour])
    else:
        if isfront:
            line_to_contour = LineString([nearest_points_on_contour, b])
        else:
            line_to_contour = LineString([b, nearest_points_on_contour])

    return line_to_contour

################################################################
def get_block_width_from_pt_on_midaxis(block, vector_midaxis, pt_on_midaxis):
    unit_v =  vector_midaxis / np.linalg.norm(vector_midaxis)
    left_v = np.array([-unit_v[1], unit_v[0]])
    right_v = np.array([unit_v[1], -unit_v[0]])
    
    dummy_left_pt = Point(pt_on_midaxis.x + left_v[0], pt_on_midaxis.y + left_v[1])
    dummy_right_pt = Point(pt_on_midaxis.x + right_v[0], pt_on_midaxis.y + right_v[1])

    left_line_to_contour = get_extend_line(dummy_left_pt, pt_on_midaxis, block, False, is_extend_from_end = True)
    right_line_to_contour = get_extend_line(dummy_right_pt, pt_on_midaxis, block, False, is_extend_from_end = True)
    
    return left_line_to_contour.length + right_line_to_contour.length

###############################################################
def get_block_aspect_ratio(block, midaxis):
    coords = np.array(midaxis.coords.xy)
    midaxis_length = midaxis.length

    relative_cutoff = [0.0]
    vector_midaxis = []
    block_width_list = []

    if midaxis.geom_type == 'GeometryCollection':
        for jj in list(midaxis.geoms):
            if jj.geom_type == 'LineString':
                midaxis = jj
                break
    coords = np.array(midaxis.coords.xy)

    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])

        if i < coords.shape[1] - 1:
            cur_width = get_block_width_from_pt_on_midaxis(block, coords[:, i] - coords[:, i-1], Point(coords[0, i], coords[1, i])) # each node on midaxis, except the last and the front.
            block_width_list.append(cur_width)

    if block_width_list == []:
        mean_block_width = block.bounds[3] - block.bounds[1]
    else:
        block_width_list = np.array(block_width_list)
        mean_block_width = np.mean(block_width_list)

    aspect_rto = np.double(mean_block_width) / np.double(midaxis_length)
    return aspect_rto



def norm_block_to_horizonal(bldg, azimuth, bbx):
    blk_offset_x = np.double(bbx.centroid.x)
    blk_offset_y = np.double(bbx.centroid.y)

    for i in range(len(bldg)):
        curr = sa.translate(bldg[i], -blk_offset_x, -blk_offset_y)
        bldg[i] = sa.rotate(curr, azimuth - 90, origin = (0.0, 0.0))

    return bldg

def get_block_parameters(block):
    bbx = block.minimum_rotated_rectangle
    azimuth = get_azimuth(bbx)
    return azimuth, bbx 


def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])

def get_azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])
    return az


def get_bbx(pos, size):
    bl = ( pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0)
    br = ( pos[0] + size[0] / 2.0, pos[1] - size[1] / 2.0)
    ur = ( pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0)
    ul = ( pos[0] - size[0] / 2.0, pos[1] + size[1] / 2.0)
    return bl, br, ur, ul




###############################################################
def get_insert_position(arr, K):
    # Traverse the array
    for i in range(arr.shape[0]):         
        # If K is found
        if arr[i] == K:
            return np.int16(i)
        # If arr[i] exceeds K
        elif arr[i] >= K:
            return np.int16(i)
    return np.int16(arr.shape[0])



########################### Input position and size from graph output, and block, and midaxis. Output original bldg with correct position/size/rotation ################################################################
def inverse_warp_bldg_by_midaxis(pos_sorted, size_sorted, midaxis, aspect_rto, rotate_bldg_by_midaxis = True, output_mode = False):
    org_size = np.zeros_like(pos_sorted)
    org_pos = np.zeros_like(pos_sorted)

    pos_sorted[:, 0] = (pos_sorted[:, 0] + 1.0) / 2.0 ############ normalize pos_x [-1, 1] back to [0, 1], pos_y keep [-1, 1] 
    pos_sorted[:, 1] = pos_sorted[:, 1] / 2.0
    size_sorted = size_sorted / 2.0

    midaxis_length = midaxis.length
    mean_block_width = aspect_rto * midaxis_length
    # print(midaxis_length, mean_block_width, aspect_rto)
    org_size[:, 0] = size_sorted[:, 0] * midaxis_length

    ###############################################################################   same as forward processing   ###################
    relative_cutoff = [0.0]
    vector_midaxis = []
    coords = np.array(midaxis.coords.xy)
    for i in range(1, coords.shape[1]):
        relative_cutoff.append(midaxis.project(Point(coords[0, i], coords[1, i]), normalized=True))
        vector_midaxis.append(coords[:, i] - coords[:, i-1])
    relative_cutoff = np.array(relative_cutoff)
    vector_midaxis = np.array(vector_midaxis)

    bldgnum = pos_sorted.shape[0]        
    corres_midaxis_vector_idx = []
    for i in range(bldgnum):
        cur_x = pos_sorted[i, 0]
        insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
        if insert_pos > vector_midaxis.shape[0]-1:
            print('\n out of index in vector_midaxis. \n')
            corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
            continue
        corres_midaxis_vector_idx.append(insert_pos)
    corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)
    ###############################################################################   same as forward processing   ###################

    ###############################################################################   get correct position and size   ###################
    for i in range(bldgnum):
        vertical_point_on_midaxis = midaxis.interpolate(pos_sorted[i, 0], normalized=True)
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        if pos_sorted[i, 1] <= 0:
            vec_from_midaxis_to_bldg = np.array([cur_vector_midaxis[1], -cur_vector_midaxis[0]])
        else:
            vec_from_midaxis_to_bldg = np.array([-cur_vector_midaxis[1], cur_vector_midaxis[0]])
        
        vec_from_midaxis_to_bldg = vec_from_midaxis_to_bldg / np.linalg.norm(vec_from_midaxis_to_bldg)

        cur_pos_x = vertical_point_on_midaxis.x + vec_from_midaxis_to_bldg[0] * np.abs(pos_sorted[i, 1]) * (mean_block_width)
        cur_pos_y = vertical_point_on_midaxis.y + vec_from_midaxis_to_bldg[1] * np.abs(pos_sorted[i, 1]) * (mean_block_width)


        org_pos[i, 0], org_pos[i, 1] = cur_pos_x, cur_pos_y
        org_size[i, 1] = size_sorted[i, 1] * mean_block_width   ##   changed from multiply by "line_from_midaxis_to_contour.length"
        # print(size_sorted[i, 1], org_size[i, 1])
    ###############################################################################   get correct position and size   ###################
    # if output_mode:
    #     org_pos, org_size = modify_pos_size_arr_overlap(org_pos, org_size)

    ###############################################################################   get original rotation  ###################
    org_bldg = []
    for i in range(org_pos.shape[0]):
        curr_bldg = Polygon(get_bbx(org_pos[i,:], org_size[i,:]))
        if rotate_bldg_by_midaxis:
            cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
            angle = np.arctan2(cur_vector_midaxis[1], cur_vector_midaxis[0]) * 180.0 / np.pi
            curr_bldg = sa.rotate(curr_bldg, angle, origin=(org_pos[i,0], org_pos[i,1]))
        org_bldg.append(curr_bldg)
    
    return org_bldg , org_pos, org_size




def get_node_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri




def graph_to_vector(g):
    output = {}
    num_nodes = g.number_of_nodes()

    asp_rto = g.graph['aspect_ratio']
    longside = g.graph['long_side']

    posx = get_node_attribute(g, 'posx', np.double)
    posy = get_node_attribute(g, 'posy', np.double)

    size_x = get_node_attribute(g, 'size_x', np.double)
    size_y = get_node_attribute(g, 'size_y', np.double)

    exist = get_node_attribute(g, 'exist', np.int_)

    node_pos = np.stack((posx, posy), 1)
    node_size = np.stack((size_x, size_y), 1)

    shape = get_node_attribute(g, 'shape', np.int16)
    iou = get_node_attribute(g, 'iou', np.double)


    output['n_size'] = node_size
    output['n_pos'] = node_pos
    output['n_exist'] = exist
    output['g_asp_rto'] = asp_rto
    output['g_longside'] = longside
    output['n_shape'] = shape
    output['n_iou'] = iou

    return output
