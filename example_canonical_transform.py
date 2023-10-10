import os, re, itertools
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, MultiLineString
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import shapely
from os import listdir
from os.path import isfile, join
import json
import shapely.geometry as sg
import shapely.affinity as sa
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.ops import nearest_points, linemerge
from skimage.morphology import medial_axis
import collections
import multiprocessing
import skgeom
import warnings
warnings.filterwarnings("ignore")
import random



class Vertex:
    def __init__(self, point, degree=0, edges=None):
        self.point = np.asarray(point)
        self.degree = degree
        self.edges = []
        self.visited = False
        if edges is not None:
            self.edges = edges

    def __str__(self):
        return str(self.point)


class Edge:

    def __init__(self, start, end=None, pixels=None):
        self.start = start
        self.end = end
        self.pixels = []
        if pixels is not None:
            self.pixels = pixels
        self.visited = False


def buildTree(img, start=None):
    # copy image since we set visited pixels to black
    img = img.copy()
    shape = img.shape
    nWhitePixels = np.sum(img)

    # neighbor offsets (8 nbors)
    nbPxOff = np.array([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1], [0, 1],
                        [1, -1], [1, 0], [1, 1]
                        ])

    queue = collections.deque()

    # a list of all graphs extracted from the skeleton
    graphs = []

    blackedPixels = 0
    # we build our graph as long as we have not blacked all white pixels!
    while nWhitePixels != blackedPixels:

        # if start not given: determine the first white pixel
        if start is None:
            it = np.nditer(img, flags=['multi_index'])
            while not it[0]:
                it.iternext()

            start = it.multi_index

        startV = Vertex(start)
        queue.append(startV)
        print("Start vertex: ", startV)

        # set start pixel to False (visited)
        img[startV.point[0], startV.point[1]] = False
        blackedPixels += 1

        # create a new graph
        G = nx.Graph()
        G.add_node(startV)

        # build graph in a breath-first manner by adding
        # new nodes to the right and popping handled nodes to the left in queue
        while len(queue):
            currV = queue[0];  # get current vertex
            # print("Current vertex: ", currV)

            # check all neigboor pixels
            for nbOff in nbPxOff:

                # pixel index
                pxIdx = currV.point + nbOff

                if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]):
                    continue;  # current neigbor pixel out of image

                if img[pxIdx[0], pxIdx[1]]:
                    # print( "nb: ", pxIdx, " white ")
                    # pixel is white
                    newV = Vertex([pxIdx[0], pxIdx[1]])

                    # add edge from currV <-> newV
                    G.add_edge(currV, newV, object=Edge(currV, newV))
                    # G.add_edge(newV,currV)

                    # add node newV
                    G.add_node(newV)

                    # push vertex to queue
                    queue.append(newV)

                    # set neighbor pixel to black
                    img[pxIdx[0], pxIdx[1]] = False
                    blackedPixels += 1

            # pop currV
            queue.popleft()
        # end while

        # empty queue
        # current graph is finished ->store it
        graphs.append(G)

        # reset start
        start = None

    # end while

    return graphs, img


def getEndNodes(g):
    # return [n for n in nx.nodes_iter(g) if nx.degree(g, n) == 1]
    return [n for n in list(g.nodes()) if nx.degree(g, n) == 1]



def mergeEdges(graph):
    # copy the graph
    g = graph.copy()

    # v0 -----edge 0--- v1 ----edge 1---- v2
    #        pxL0=[]       pxL1=[]           the pixel lists
    #
    # becomes:
    #
    # v0 -----edge 0--- v1 ----edge 1---- v2
    # |_________________________________|
    #               new edge
    #    pxL = pxL0 + [v.point]  + pxL1      the resulting pixel list on the edge
    #
    # an delete the middle one
    # result:
    #
    # v0 --------- new edge ------------ v2
    #
    # where new edge contains all pixels in between!

    # start not at degree 2 nodes
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]

    for v0 in startNodes:

        # start a line traversal from each neighbor
        startNNbs = nx.neighbors(g, v0)

        startNNbs = list(startNNbs)
        if not len(startNNbs):
            continue

        counter = 0
        v1 = startNNbs[counter]  # next nb of v0
        while True:

            if nx.degree(g, v1) == 2:
                # we have a node which has 2 edges = this is a line segement
                # make new edge from the two neighbors
                nbs = nx.neighbors(g, v1)
                nbs = list(nbs)
                # if the first neihbor is not n, make it so!
                if nbs[0] != v0:
                    nbs.reverse()

                pxL0 = g[v0][v1]["object"].pixels  # the pixel list of the edge 0
                pxL1 = g[v1][nbs[1]]["object"].pixels  # the pixel list of the edge 1

                # fuse the pixel list from right and left and add our pixel n.point
                g.add_edge(v0, nbs[1],
                           object=Edge(v0, nbs[1], pixels=pxL0 + [v1.point] + pxL1)
                           )

                # delete the node n
                g.remove_node(v1)

                # set v1 to new left node
                v1 = nbs[1]

            else:
                counter += 1
                if counter == len(startNNbs):
                    break
                v1 = startNNbs[counter]  # next nb of v0

    # weight the edges according to their number of pixels
    for u, v, o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)

    return g


def getLongestPath(graph, endNodes):
    """
        graph is a fully reachable graph = every node can be reached from every node
    """

    if len(endNodes) < 2:
        raise ValueError("endNodes need to contain at least 2 nodes!")

    # get all shortest paths from each endpoint to another endpoint
    allEndPointsComb = itertools.combinations(endNodes, 2)

    maxLength = 0
    maxPath = None

    for ePoints in allEndPointsComb:

        # get shortest path for these end points pairs
        try:
            sL = nx.dijkstra_path_length(graph,
                                        source=ePoints[0],
                                        target=ePoints[1])
        except:
            continue
        # dijkstra can throw if now path, but we are sure we have a path

        # store maximum
        if (sL > maxLength):
            maxPath = ePoints
            maxLength = sL

    if maxPath is None:
        raise ValueError("No path found!")

    return nx.dijkstra_path(graph,
                            source=maxPath[0],
                            target=maxPath[1]), maxLength

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

# ###############################################################
def is_on_contour(polygon, pt):
    for i in polygon.coords:
        if np.linalg.norm(i - np.array((pt.x(), pt.y()), dtype=np.float32)) < 0.01:
            return True
    return False

def skgeom_dist(p1, p2):
    return np.sqrt((float(p1.x()) - float(p2.x())) **2 + (float(p1.y()) - float(p2.y())) **2)


def get_polyskeleton_longest_path(skeleton, polygon):
    # skgeom.draw.draw(polygon)
    interior_skel_vertices = {}
    interior_skel_time = {}
    exterior_vertices = {}
    connect_dict = {}

    G = nx.Graph() 
    end_nodes = []

    for v in skeleton.vertices:
        if v.time > 0.001:
            interior_skel_vertices[v.id] = Point(v.point.x(), v.point.y())
            interior_skel_time[v.id] = v.time
        else:
            exterior_vertices[v.id] = Point(v.point.x(), v.point.y())
            end_nodes.append(v.id)
        G.add_node(v.id, posx = float(v.point.x()), posy = float(v.point.y()) )

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point
            # plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=2)
            if not (is_on_contour(polygon, p1) and is_on_contour(polygon, p2)):
                G.add_edge(h.vertex.id, h.opposite.vertex.id, weight = skgeom_dist(p1, p2))

    
    path, length = getLongestPath(G, end_nodes)
    longest_skel = []
    for i in path:
        longest_skel.append((G.nodes[i]['posx'], G.nodes[i]['posy']))
    longest_skel = LineString(longest_skel)


    return G, longest_skel

def get_modified_nodes(coords):
    default = 1
    for i in range(coords.shape[1] - 2):
        k1 = get_k_angle(Point(coords[:,i]), Point(coords[:,i+1]) )
        k2 = get_k_angle(Point(coords[:,i+1]), Point(coords[:,i+2]) )
        if np.abs(k1 - k2) > 2 * np.pi/36.0:  # 5 degree
            return i+1
    return default

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

def get_size_with_vector(mrr):
    mrr = mrr.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        return axis2, axis1, np.array(bbox[0])-np.array(bbox[1]), np.array(bbox[0])-np.array(bbox[3]), [0, 1], [0, 3]
    else:
        return axis1, axis2, np.array(bbox[0])-np.array(bbox[3]), np.array(bbox[0])-np.array(bbox[1]), [0, 3], [0, 1] # longside, shortside    

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
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=a.distance)
        extended_line = LineString([a, Point(points_sorted_by_distance[0])])
    

    min_dis = 9999999999.9
    intersect = block.boundary.intersection(extended_line)
    if intersect.geom_type == 'MultiPoint':
        for i in range(len(intersect.geoms)):
            if intersect.geoms[i].distance(a) <= min_dis:
                nearest_points_on_contour = intersect.geoms[i]
    elif intersect.geom_type == 'Point':
        nearest_points_on_contour = intersect
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

def modified_skel_to_medaxis(longest_skel, block):
    coords = np.array(longest_skel.coords.xy)
    if coords.shape[1] <=3:
        return longest_skel
    elif coords.shape[1] == 4:
        front_start = 1
        end_start = 1

    else:
        flip_coords = np.flip(coords,1)
        front_start = get_modified_nodes(coords)
        end_start = get_modified_nodes(flip_coords)

    p2 = Point(coords[:,front_start])
    p3 = Point(coords[:,front_start + 1])
    p_2 = Point(coords[:,-end_start -1])
    p_3 = Point(coords[:,-end_start -2])

    extended_line_front = get_extend_line(p2, p3, block, True)
    extended_line_back = get_extend_line(p_2, p_3, block, False)

    if len(longest_skel.coords[front_start:-end_start]) <=1:
        midaxis_miss_bothend = LineString(longest_skel.coords[1:-1])
        print('no enough point on medaxis')
    else:
        midaxis_miss_bothend = LineString(longest_skel.coords[front_start:-end_start])

    modfied_midaxis = linemerge([extended_line_front, midaxis_miss_bothend, extended_line_back])

    return modfied_midaxis

###############################################################
def warp_bldg_by_midaxis(bldg, block, midaxis):
    bldgnum = len(bldg)
    normalized_size = []
    midaxis_length = midaxis.length

    ################################################################
    relative_cutoff = [0.0]
    vector_midaxis = []
    block_width_list = []

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

    relative_cutoff = np.array(relative_cutoff)
    vector_midaxis = np.array(vector_midaxis)

    normalized_x = []
    corres_midaxis_vector_idx = []
    for i in range(bldgnum):
        cur_x = midaxis.project(bldg[i].centroid, normalized=True)
        normalized_x.append(cur_x)
        insert_pos = get_insert_position(relative_cutoff, cur_x) - 1
        if insert_pos > vector_midaxis.shape[0]-1:
            print('\n out of index in vector_midaxis. \n')
            corres_midaxis_vector_idx.append(vector_midaxis.shape[0]-1)
            continue
        corres_midaxis_vector_idx.append(insert_pos)
    corres_midaxis_vector_idx = np.array(corres_midaxis_vector_idx)

    normalized_y = []
    for i in range(bldgnum):
        vertical_point_on_midaxis = midaxis.interpolate(normalized_x[i], normalized=True)

        vector_midaxis_to_bldg = np.array( [bldg[i].centroid.x - vertical_point_on_midaxis.x, bldg[i].centroid.y - vertical_point_on_midaxis.y] )
        cur_vector_midaxis = vector_midaxis[corres_midaxis_vector_idx[i], :]
        cross_prod = np.cross(cur_vector_midaxis, vector_midaxis_to_bldg)

        dist_from_midaxis_to_bldg = np.sqrt(vector_midaxis_to_bldg[0] * vector_midaxis_to_bldg[0] + vector_midaxis_to_bldg[1] * vector_midaxis_to_bldg[1])
       
        relative_y = 2.0 * dist_from_midaxis_to_bldg / mean_block_width #  changed from "dist_from_midaxis_to_contour", without 2.0 multiple
        if cross_prod <= 0:
            normalized_y.append(-relative_y)
        else:
            normalized_y.append(relative_y)

        ###########################################################################################
        longside, shortside, long_vec, short_vec, _, _ = get_size_with_vector(bldg[i].minimum_rotated_rectangle)
        long_vec = long_vec / np.linalg.norm(long_vec)
        unit_cur_vector_midaxis =  cur_vector_midaxis / np.linalg.norm(cur_vector_midaxis)

        long_angle = np.arccos(np.dot(long_vec, unit_cur_vector_midaxis ))
        long_angle = np.min([np.pi - long_angle, long_angle])

        short_vec = short_vec / np.linalg.norm(short_vec)
        short_angle = np.arccos(np.dot(short_vec, unit_cur_vector_midaxis ))
        short_angle = np.min([np.pi - short_angle, short_angle])

        if short_angle < long_angle:
            currr = shortside
            shortside = longside
            longside = currr

        normalized_size_x = 2 * longside / midaxis_length  ############ because pos is [-1, 1], size can be 2 
        normalized_size_y = 2 * shortside / mean_block_width # changed from divded by "dist_from_midaxis_to_contour"
        normalized_size.append([normalized_size_x, normalized_size_y])
        
    normalized_x = np.array(normalized_x, dtype = np.double)
    normalized_x = 2.0 * normalized_x - 1.0    ############ normalize pos_x to [-1, 1], pos_y already been [-1, 1] 
    normalized_y = np.array(normalized_y, dtype = np.double)
    normalized_pos = np.stack((normalized_x, normalized_y), axis = 1)
    normalized_size = np.array(normalized_size, np.double)

    pos_sort = np.lexsort((normalized_pos[:,1],normalized_pos[:,0]))
    pos_sorted = normalized_pos[pos_sort]
    size_sorted = normalized_size[pos_sort]

    aspect_rto = np.double(mean_block_width) / np.double(midaxis_length)

    return pos_sorted, size_sorted, pos_sort, aspect_rto




if __name__ == '__main__':
    ### Normalize your block and building polygons to block center
    #### norm_blk_poly : city block shaply.Polygon
    #### norm_bldg_poly : list of building shaply.Polygon

    #### a random shaply.Polygon
    norm_blk_poly = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])

    #### a random list of shaply.Polygon
    norm_bldg_poly = []
    norm_bldg_poly.append(Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]))

    exterior_polyline = list(norm_blk_poly.exterior.coords)[:-1]
    exterior_polyline.reverse()
    poly_list = []
    for ix in range(len(exterior_polyline)):
        poly_list.append(exterior_polyline[ix])
    sk_norm_blk_poly = skgeom.Polygon(poly_list)

    #### get the skeleton of block
    skel = skgeom.skeleton.create_interior_straight_skeleton(sk_norm_blk_poly)
    G, longest_skel = get_polyskeleton_longest_path(skel, sk_norm_blk_poly)

    ### get the medial axis of block
    medaxis = modified_skel_to_medaxis(longest_skel, norm_blk_poly)

    #############   wrap all building locations and sizes ###############################################################
    pos_xsorted, size_xsorted, xsort_idx, aspect_rto = warp_bldg_by_midaxis(norm_bldg_poly, norm_blk_poly, medaxis)

    ### pos_xsorted : normalized building locations, size_xsorted : normalized building sizes, xsort_idx : sorted index of building locations in graph
    ### aspect_rto : aspect ratio of block
    ######################################################################################################################

    ### Then pos, size and aspect_rto are the used to generate graph by networkx.graph()

    ##### The position, and size attributes in provided graph is further normalized by minus mean and divided std of the entire dataset. But that is not necessary for performance.




