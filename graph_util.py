import networkx as nx
import matplotlib.pyplot as plt
import os, re
import pickle
import random
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import yaml

plt.subplots(figsize=(20, 4))


def visual_grid_graph(G, filepath, filename):
    pos = {}
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    for ee in range(G.number_of_nodes()):
        pos[ee] = (G.nodes[ee]['posx'], G.nodes[ee]['posy'])

    nx.draw_networkx(G, pos=pos,
            node_color='lightgreen',
            with_labels=True,
            node_size=600)
    plt.savefig(os.path.join(filepath,filename))
    plt.clf()



def sparse_generate_graph_from_ftsarray(height, width, x_pos, y_pos, h_out, w_out, exist, asp_rto, long_side, b_shape, b_iou, b_height = None):
    max_node = height * width
    g = nx.grid_2d_graph(height, width)
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')
    G.graph['aspect_ratio'] = asp_rto
    G.graph['long_side'] = long_side

    if b_height is None:
        b_height = np.zeros_like(b_iou)

    for i in range(height):
        for j in range(width):
            idx = i * width + j
            G.nodes[idx]['posy'] = y_pos[idx]
            G.nodes[idx]['posx'] = x_pos[idx]
            G.nodes[idx]['exist'] = exist[idx]
            G.nodes[idx]['merge'] = 0
            G.nodes[idx]['size_x'] = w_out[idx]
            G.nodes[idx]['size_y'] = h_out[idx]
            G.nodes[idx]['shape'] = b_shape[idx]
            G.nodes[idx]['iou'] = b_iou[idx]
            G.nodes[idx]['height'] = b_height[idx]

    return G




def recover_graph_features(g, existed, merge, posx, posy):
    G = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute = 'old_label')

    for i in range(g.number_of_nodes()):
        G.nodes[i]['posy'] = posy[i, 0]
        G.nodes[i]['posx'] = posx[i, 0]
        G.nodes[i]['merge'] = merge[i, 0]
        G.nodes[i]['exist'] = existed[i, 0]

    return G



def visual_block_graph(G, filepath, filename, draw_edge = False, draw_nonexist = False):
    pos = []
    size = []
    edge = []
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    if not draw_nonexist:
        for i in range(G.number_of_nodes()):
            if G.nodes[i]['exist'] == 0: # or abs(G.nodes[i]['size_x']) < 1e-2 or abs(G.nodes[i]['size_y']) < 1e-2 s
                G.remove_node(i)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    for i in range(G.number_of_nodes()):
        pos.append([G.nodes[i]['posx'], G.nodes[i]['posy']])
        size.append([G.nodes[i]['size_x'], G.nodes[i]['size_y']])

    for e in G.edges:
        edge.append(e)

    pos = np.array(pos, dtype = np.double)    
    size = np.array(size, dtype = np.double)
    edge = np.array(edge, dtype = np.int16)
    print(pos.shape, len(pos))

    if len(pos) > 0:
        plt.scatter(pos[:, 0], pos[:, 1], c = 'red', s=50)
        ax = plt.gca()
        for i in range(size.shape[0]):
            ax.add_patch(Rectangle((pos[i, 0] - size[i, 0] / 2.0, pos[i, 1] - size[i, 1] / 2.0), size[i, 0], size[i, 1], linewidth=2, edgecolor='r', facecolor='b', alpha=0.3)) 

        if draw_edge:
            for i in range(edge.shape[0]):
                l = mlines.Line2D([pos[edge[i, 0], 0], pos[edge[i, 1], 0]], [pos[edge[i, 0], 1], pos[edge[i, 1], 1]])
                ax.add_line(l)

    plt.savefig(os.path.join(filepath,filename + '.png'))
    plt.clf()



def visual_existence_template(G, filepath, filename, coord_scale = 1, template_width = 25, template_height = 2):
    unit_w = np.double(2 * coord_scale) / np.double(template_width)
    unit_h = np.double(2 * coord_scale) / np.double(template_height)

    w_anchor = np.arange(-coord_scale + unit_w / 2.0, coord_scale + 1e-6, unit_w)
    h_anchor = np.arange(-coord_scale + unit_h / 2.0, coord_scale + 1e-6, unit_h)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    exist = []

    for i in range(G.number_of_nodes()):
        exist.append(G.nodes[i]['exist'])
    
    for i in range(template_height):
        for j in range(template_width):
            if exist[template_width * i + j] == 0:
                plt.scatter(w_anchor[j], h_anchor[i], c = 'red', s=100)
            else:
                plt.scatter(w_anchor[j], h_anchor[i], c = 'green', s=100)


    plt.savefig(os.path.join(filepath,filename + '.png'))
    plt.clf()


def read_train_yaml(checkpoint_name, filename = "train.yaml"):
    with open(os.path.join(checkpoint_name, filename), "rb") as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return opt