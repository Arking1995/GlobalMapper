import torch, os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from urban_dataset import UrbanGraphDataset, graph_transform, get_transform, test_graph_transform
from model import *
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch
import numpy as np
import random
from torch.optim.lr_scheduler import MultiStepLR
from time import gmtime, strftime
import shutil
from graph_util import *
import shutil
from torchvision.utils import save_image


if __name__ == "__main__":
    root = os.getcwd()
    random.seed(42) # make sure every time has the same training and validation sets

    pth_name =  'latest'  
    epoch_name = 'your_epoch_name'
    dataset_path = 'your_dataset_folder'
    data_name = 'osm_cities'

    gpu_ids = 0
    batch_size = 1

    is_teaser = False
    teaser_note = 'continuous_chicago'
    if is_teaser:
        dataset_path = os.path.join('/opt/data/liuhe95/Teaser_set',teaser_note,data_name)


    template_height = 4 # opt['template_height']
    template_width = 30 # opt['template_width']
    N = template_width * template_height

    is_reconstruct = True # is reconstructed from validation dataset or from random normal distribution.

    test_yaml = os.path.join(root, 'epoch', epoch_name)
    opt = read_train_yaml(test_yaml, "train_save.yaml")

    root = os.getcwd()


    if is_teaser:
        output_num = 1e10 # number of needed test samples. if set to none, will run the number of validation set times.
    else:
        output_num = 1000

    draw_edge = True
    draw_nonexist = False


    save_pth = os.path.join(root,'test')
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    if is_teaser:
        save_pth = os.path.join('/opt/data/liuhe95/Teaser_set', teaser_note,'Rescale_results')
    else:
        save_pth = os.path.join(save_pth, 'test_' + epoch_name)
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)


    if is_reconstruct:
        dir = pth_name + '_reconstruct_' + data_name
        if is_teaser:
            dir = pth_name + '_reconstruct_continuous'
    else:
        dir = pth_name + '_var_gen'

    if not is_reconstruct:
        with open(os.path.join(save_pth, 'val_least_loss_geo_sample_stats'), 'rb') as f:
            [z_mean, z_std] = pickle.load(f)

    save_pth = os.path.join(save_pth, dir)
    res_path = os.path.join(save_pth, 'result')
    gt_path = os.path.join(save_pth, 'gt')

    res_graph_path = os.path.join(res_path, 'graph')
    res_visual_path = os.path.join(res_path, 'visual')

    res_block_img_path = os.path.join(res_path, 'block_img')
    res_final_img_path = os.path.join(res_path, 'final')


    gt_graph_path = os.path.join(gt_path, 'graph')
    gt_visual_path = os.path.join(gt_path, 'visual')

    ex_visual_path = os.path.join(res_path, 'exist')
    gt_ex_visual_path = os.path.join(gt_path, 'exist')


    if is_reconstruct:
        pathlist = [save_pth, res_path, gt_path, res_graph_path, res_visual_path, gt_graph_path, gt_visual_path, ex_visual_path, gt_ex_visual_path, res_block_img_path]
    else:
        pathlist = [save_pth, res_path, res_graph_path, res_visual_path, ex_visual_path, res_block_img_path]

    for i in pathlist:
        if not os.path.exists(i):
            os.mkdir(i)


    device = torch.device('cuda:' + str(gpu_ids))
    opt['device'] = device


    if opt['is_blockplanner']:
        model = NaiveBlockGenerator(opt, N = N)

    elif opt['is_conditional_block']:
        if opt['convlayer'] in opt['attten_net']:
            model = AttentionBlockGenerator(opt, N = N)
        else:
            model = BlockGenerator(opt, N = N)
    else:
        if opt['convlayer'] in opt['attten_net']:
            if opt['encode_cnn']:
                model = AttentionBlockGenerator_independent_cnn(opt, N = N) #, T = 1
            else:
                model = AttentionBlockGenerator_independent(opt, N = N)

    model.load_state_dict(torch.load(os.path.join(test_yaml, pth_name + ".pth"), map_location=device), strict=False)
    model.to(device)

    cnn_transform = get_transform(noise_range = 10.0, noise_type = 'gaussian', isaug = False, rescale_size = 64)
    dataset = UrbanGraphDataset(dataset_path,transform = test_graph_transform, cnn_transform = cnn_transform)
    num_data = len(dataset)

    ######################### This two lines only for teaser dataset
    if is_teaser:
        val_idx = np.arange(num_data)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print('Get {} graph for teaser testing.'.format(val_idx.shape[0]))


    ########################   uncomment for general testing on validation set
    else:
        val_num = int(num_data * opt['val_ratio'])
        val_idx = np.array(random.sample(range(num_data), val_num))
        print('Get {} graph for validation'.format(val_idx.shape[0]))
        val_dataset = dataset[val_idx]
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    fn_ct = 0
    z_sample_list = []

    for data in val_loader:

        if output_num != None:
            if fn_ct >= output_num:
                break
        print(fn_ct)
        data = data.to(device)
        edge_index = data.edge_index  # assume all edges are existed

        if opt['is_blockplanner']:
            if is_reconstruct:
                mu, log_var = model.encode(data)
                z_sample = model.reparameterize(mu, log_var)
            else:
                z_sample = torch.randn(batch_size, opt['latent_dim']).to(device)
            z_sample_list.append(z_sample.squeeze().detach().cpu().numpy())
            if opt['is_input_road']:
                block_condition = data.block_condition.view(batch_size, 2, 64, 64)
                block_condition = model.cnn_encode(block_condition)
                exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, block_condition, data.edge_index)
            else:
                exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, edge_index)

        else:
            if opt['encode_cnn']:
                if is_reconstruct:
                    mu, log_var = model.encode(data)
                    if opt['is_input_road']:
                        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
                        block_condition = model.cnn_encode(block_condition)
                    z_sample = model.reparameterize(mu, log_var)
                    z_sample_list.append(z_sample.squeeze().detach().cpu().numpy())
                else:
                    if opt['is_input_road']:
                        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
                        block_condition = model.cnn_encode(block_condition)
                    z_sample = torch.normal(torch.from_numpy(z_mean), torch.from_numpy(z_std)).unsqueeze(0).to(device)

                if opt['is_input_road']:
                    exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, block_condition, data.edge_index)
                else:
                    exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, data.edge_index)


            else:
                if is_reconstruct:
                    mu, log_var = model.encode(data)
                    z_sample = model.reparameterize(mu, log_var)
                    block_scale = model.enc_block_scale(data.block_scale_gt.unsqueeze(1))
                    block_shape = data.blockshape_latent_gt.view(-1, model.blockshape_latent_dim)
                    block_condition = torch.cat((block_shape, block_scale), 1)
                else:
                    z_sample = torch.randn(batch_size, opt['latent_dim']).to(device)
                    block_scale = model.enc_block_scale(data.block_scale_gt.unsqueeze(1))
                    block_shape = data.blockshape_latent_gt.view(-1, model.blockshape_latent_dim)
                    block_condition = torch.cat((block_shape, block_scale), 1)
                exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, block_condition, data.edge_index)

        asp_rto = torch.zeros_like(data.asp_rto_gt.unsqueeze(1))
        long_side = torch.zeros_like(data.long_side_gt.unsqueeze(1))


        exist_gt = data.x[:, 0].unsqueeze(1)
        pos_gt = data.org_node_pos
        size_gt = data.org_node_size

        exist = torch.ge(exist, 0.5).type(torch.uint8)
        correct_ext = (exist == data.x[:,0].unsqueeze(1)).sum() /  torch.numel(data.x[:,0])
        
        exist = exist.squeeze().detach().cpu().numpy()
        posx = posx.squeeze().detach().cpu().numpy()
        posy = posy.squeeze().detach().cpu().numpy()
        sizex = sizex.squeeze().detach().cpu().numpy()
        sizey = sizey.squeeze().detach().cpu().numpy()
        asp_rto = asp_rto.squeeze().detach().cpu().numpy()
        long_side = long_side.squeeze().detach().cpu().numpy()
        b_iou = b_iou.squeeze().detach().cpu().numpy()

        _, shape_pred = torch.max(b_shape, 1)
        shape_pred = shape_pred.detach().cpu().numpy()


        for i in range(batch_size):
            g_add = sparse_generate_graph_from_ftsarray(template_height, template_width, posx, posy, sizey, sizex, exist, asp_rto, long_side, shape_pred, b_iou)
            if is_reconstruct:
                filename = str(val_idx[fn_ct])
            else:
                filename = str(fn_ct)
            
    
            pickle.dump(g_add, open(os.path.join(res_graph_path, filename + ".gpickle"), 'wb')) 
            # visual_block_graph(g_add, res_visual_path, filename, draw_edge, draw_nonexist)
            # visual_existence_template(g_add, ex_visual_path, filename, coord_scale = 1, template_width = template_width, template_height = template_height)
            # save_image(target_image, os.path.join(res_block_img_path,filename+'.png') )

            if is_reconstruct:
                rst = os.path.join(dataset_path, 'processed', filename + ".gpickle")
                dst = os.path.join(gt_graph_path, filename + '.gpickle')
                g = nx.read_gpickle(rst)
                shutil.copyfile(rst, dst)
                # visual_block_graph(g, gt_visual_path, filename, draw_edge, draw_nonexist)
                # visual_existence_template(g, gt_ex_visual_path, filename, coord_scale = 1, template_width = template_width, template_height = template_height)

            fn_ct += 1


    if is_reconstruct:
        z_sample_array = np.array(z_sample_list)
        z_mean = np.mean(z_sample_array, axis = 0)
        z_std = np.std(z_sample_array, axis = 0)

        xpoints = range(opt['latent_dim'])
        ypoints = z_mean.flatten()
        plt.plot(xpoints, ypoints)
        plt.savefig(os.path.join(save_pth,'mean.png'))
        plt.clf()

        ypoints = z_std.flatten()
        plt.plot(xpoints, ypoints)
        plt.savefig(os.path.join(save_pth,'std.png'))

        with open(os.path.join(save_pth, 'sample_stats_'+ str(fn_ct)), 'wb') as f:
            pickle.dump([z_mean, z_std], f)

        with open(os.path.join(save_pth, 'z_sample_'+ str(fn_ct)), 'wb') as f:
            pickle.dump([z_sample_array], f)

    print('Finish')









