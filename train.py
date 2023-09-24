import torch, os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from urban_dataset import UrbanGraphDataset, graph_transform, get_transform
from model import *
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch
import numpy as np
import random
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import MultiStepLR
from time import gmtime, strftime
import shutil
import logging
from graph_util import read_train_yaml
from graph_trainer import train, validation
import yaml
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    random.seed(42) # make sure every time has the same training and validation sets


    root = os.getcwd()

    dataset_path = os.path.join(root, 'dataset')  ### you may set up your own dataset

    train_opt = read_train_yaml(root, filename = "train_gnn.yaml")
    print(train_opt)
    is_resmue = train_opt['resume']
    gpu_ids = train_opt['gpu_ids']

    if is_resmue:
        resume_epoch = train_opt['resume_epoch']
        resume_dir = train_opt['resume_dir']
        import_name = train_opt['import_name']
        opt = read_train_yaml(os.path.join(root,'epoch', resume_dir), filename = "train_save.yaml")
    else:
        opt = train_opt

    notes = 'GlobalMapper'
        
    maxlen = opt['template_width']
    N = maxlen * opt['template_height']
    min_bldg = 0   # >
    max_bldg = N   # <=
    opt['N'] = int(N)

    fname = opt['convlayer'] + '_' + opt['aggr'] + '_dim' + str(opt['n_ft_dim'])
    data_name = 'osm_cities' #Structure + '_Bldg'+str(min_bldg)+'-'+str(max_bldg)+'_ML'+str(maxlen) +'_N' + str(N)

    opt['data_name'] = data_name 
    print(data_name)
    device = torch.device('cuda:{}'.format(gpu_ids[0]))
    print(device)
    opt['device'] = str(device)
    start_epoch = opt['start_epoch']


    loss_dict = {}
    loss_dict['Posloss'] = nn.MSELoss(reduction='sum')
    loss_dict['ShapeCEloss'] = nn.CrossEntropyLoss(reduction='sum')
    loss_dict['Iouloss'] = nn.MSELoss(reduction='sum')
    loss_dict['ExistBCEloss'] = nn.BCEWithLogitsLoss(reduction='sum')
    loss_dict['CELoss'] = nn.CrossEntropyLoss(reduction='none')
    loss_dict['Sizeloss'] = nn.MSELoss(reduction='sum')  # nn.SmoothL1Loss
    loss_dict['ExtSumloss'] = nn.MSELoss(reduction='sum')  # nn.SmoothL1Loss


    save_pth = os.path.join(root,'epoch')
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    log_pth = os.path.join(root,'tensorboard')
    if not os.path.exists(log_pth):
        os.mkdir(log_pth)

    logs_pth = os.path.join(root,'logs')
    if not os.path.exists(logs_pth):
        os.mkdir(logs_pth)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    if is_resmue:
        save_name = notes + fname + '_lr{}_epochs{}_batch{}'.format(opt['lr'], opt['total_epochs'], opt['batch_size']) 
        save_pth = os.path.join(root,'epoch', resume_dir)
        log_file = os.path.join(root,'logs', resume_dir + '.log')
        tb_path = os.path.join(root,'tensorboard', resume_dir)
    else:
        save_name = notes + fname + '_lr{}_epochs{}_batch{}_'.format(opt['lr'], opt['total_epochs'], opt['batch_size']) 
        save_pth = os.path.join(root,'epoch', save_name + time)
        log_file = os.path.join(root,'logs', save_name + time + '.log')
        tb_path = os.path.join(root,'tensorboard', save_name + time)


    if opt['save_record']:
        if not os.path.exists(save_pth):
            os.mkdir(save_pth)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO, 
            format="[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ", 
            filename = log_file)
        opt['save_path'] = save_pth
        opt['log_path'] = log_file
        opt['tensorboard_path'] = tb_path
        if is_resmue:
            yaml_fn = 'resume_train_save.yaml'
            opt['save_notes'] = save_name
        else:
            yaml_fn = 'train_save.yaml'
        with open(os.path.join(save_pth, yaml_fn), 'w') as outfile:
            yaml.dump(opt, outfile, default_flow_style=False)

        configure(tb_path, flush_secs=5)



    torch.autograd.set_detect_anomaly(True)

    cnn_transform = get_transform(noise_range = 10.0, noise_type = 'gaussian', isaug = False, rescale_size = 64)
    dataset = UrbanGraphDataset(dataset_path,transform = graph_transform, cnn_transform = cnn_transform)
    num_data = len(dataset)

    opt['num_data'] = int(num_data)
    print(num_data)

    ### get the validation data number
    val_num = int(num_data * opt['val_ratio']) 
    ### sample the validation data index
    val_idx = np.array(random.sample(range(num_data), val_num))
    ### remove the validation data index for training
    train_idx = np.delete(np.arange(num_data), val_idx)


    print('Get {} graph for training'.format(train_idx.shape[0]))
    print('Get {} graph for validation'.format(val_idx.shape[0]))


    val_dataset = dataset[val_idx]
    train_dataset = dataset[train_idx]
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=8)
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=8)

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
                print('attention net')
                model = AttentionBlockGenerator_independent_cnn(opt, N = N) #, T = 1
            else:
                model = AttentionBlockGenerator_independent(opt, N = N)

    if is_resmue:
        start_epoch = resume_epoch
        print('import from {}'.format(os.path.join(root,'epoch',resume_dir, import_name+'.pth')))
        model.load_state_dict(torch.load(os.path.join(root,'epoch',resume_dir, import_name+'.pth'), map_location=device))


    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= float(opt['lr']), weight_decay=1e-6)
    scheduler = MultiStepLR(optimizer, milestones=[(opt['total_epochs'] - start_epoch) * 0.6, (opt['total_epochs']-start_epoch) * 0.8], gamma=0.3)


    best_val_acc = None
    best_train_acc = None
    best_train_loss = None
    best_val_loss = None
    best_val_geo_loss = None

    print('Start Training...')
    logging.info('Start Training...' )

    for epoch in range(start_epoch, opt['total_epochs']):

        t_acc, t_loss = train(model, epoch, train_loader, device, opt, loss_dict, optimizer, scheduler)
        v_acc, v_loss, v_loss_geo = validation(model, epoch, val_loader, device, opt, loss_dict, scheduler)

        if opt['save_record']:
            if best_train_acc is None or t_acc >= best_train_acc:
                best_train_acc = t_acc

            if best_train_loss is None or t_loss <= best_train_loss:
                best_train_loss = t_loss

            if best_val_acc is None or v_acc >= best_val_acc:
                best_val_acc = v_acc
                filn = os.path.join(save_pth, "val_best_extacc.pth")
                torch.save(model.state_dict(), filn)

            if best_val_loss is None or v_loss <= best_val_loss:
                best_val_loss = v_loss
                filn = os.path.join(save_pth, "val_least_loss_all.pth")
                torch.save(model.state_dict(), filn)

            if best_val_geo_loss is None or v_loss_geo <= best_val_geo_loss:
                best_val_geo_loss = v_loss_geo
                filn = os.path.join(save_pth, "val_least_loss_geo.pth")
                torch.save(model.state_dict(), filn)

            if epoch % opt['save_epoch'] == 0:
                filn = os.path.join(save_pth, str(epoch) + "_save.pth")
                torch.save(model.state_dict(), filn)            
            logging.info('Epoch: {:03d}, Train Loss: {:.7f}, Train exist accuracy: {:.7f}, Valid Loss: {:.7f}, Valid exist accuracy: {:.7f}, valid geo loss {:.7f}'.format(epoch, t_loss, t_acc, v_loss, v_acc, v_loss_geo) )
            print('Epoch: {:03d}, Train Loss: {:.7f}, Train exist accuracy: {:.7f}, Valid Loss: {:.7f}, Valid exist accuracy: {:.7f}, valid geo loss {:.7f}'.format(epoch, t_loss, t_acc, v_loss, v_acc, v_loss_geo) )

            filn = os.path.join(save_pth, "latest.pth")
            torch.save(model.state_dict(), filn)  

    if opt['save_record']:
        logging.info('Least Train Loss: {:.7f}, Best Train exist accuracy: {:.7f}, Least Valid Loss: {:.7f}, Best Valid exist accuracy: {:.7f}, best valid geo loss {:.7f}'.format(best_train_loss, best_train_acc, best_val_loss, best_val_acc, best_val_geo_loss))
        print('Least Train Loss: {:.7f}, Best Train exist accuracy: {:.7f}, Least Valid Loss: {:.7f}, Best Valid exist accuracy: {:.7f}, best valid geo loss {:.7f}'.format(best_train_loss, best_train_acc, best_val_loss, best_val_acc, best_val_geo_loss))
