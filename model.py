import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
from torch_geometric.utils import add_self_loops, degree



class NaiveMsgPass(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = (deg_inv_sqrt[row] + deg_inv_sqrt[col]).pow(-1.0)

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_j has shape [E, out_channels]
        tmp = torch.cat([x_i, x_j], dim=1)
        tmp = self.lin(tmp)

        # Step 4: Normalize node features.
        return tmp



class BlockGenerator(torch.nn.Module):
    def __init__(self, opt, N = 80, T=3, frequency_num = 32, f_act = 'relu'):    
        super(BlockGenerator, self).__init__()
        self.N = N
        self.device = opt['device']
        self.latent_dim = opt['latent_dim']
        self.latent_ch = opt['n_ft_dim']
        self.T = T
        self.blockshape_latent_dim = opt['block_latent_dim']
        

        if opt['aggr'] == 'Mean':
            self.global_pool = torch_geometric.nn.global_mean_pool
        elif opt['aggr'] == 'Max':
            self.global_pool = torch_geometric.nn.global_max_pool
        elif opt['aggr'] == 'Add':
            self.global_pool = torch_geometric.nn.global_add_pool
        elif opt['aggr'] == 'global_sort_pool':
            self.global_pool = torch_geometric.nn.global_sort_pool     
        elif opt['aggr'] == 'GlobalAttention':
            self.global_pool = torch_geometric.nn.GlobalAttention
        elif opt['aggr'] == 'Set2Set':
            self.global_pool = torch_geometric.nn.Set2Set
        elif opt['aggr'] == 'GraphMultisetTransformer':
            self.global_pool = torch_geometric.nn.GraphMultisetTransformer


        if opt['convlayer'] == 'GCNConv':
            self.convlayer = torch_geometric.nn.GCNConv
        elif opt['convlayer'] == 'NaiveMsgPass':
            self.convlayer = NaiveMsgPass
        else:
            self.convlayer = torch_geometric.nn.GCNConv


        self.ex_init = nn.Linear(2, int(self.latent_ch/4))
        self.ft_init = nn.Linear(int(self.latent_ch/4) + N, int(self.latent_ch/2))
        
        self.pos_init = nn.Linear(2, int(self.latent_ch/2))
        self.size_init = nn.Linear(2, int(self.latent_ch/2))


        self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch)
        self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        self.d_conv1 = self.convlayer(self.latent_ch + N, self.latent_ch)
        self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        self.d_ft_init = nn.Linear(self.latent_dim, self.latent_ch * N)


        self.aggregate = nn.Linear(int(self.latent_ch*(2.5 + self.T)), self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        self.d_exist_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_exist_1 = nn.Linear(self.latent_ch, 1)

        self.d_posx_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_posx_1 = nn.Linear(self.latent_ch, 1)

        self.d_posy_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_posy_1 = nn.Linear(self.latent_ch, 1)

        self.d_sizex_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_sizex_1 = nn.Linear(self.latent_ch, 1)

        self.d_sizey_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_sizey_1 = nn.Linear(self.latent_ch, 1)

        self.enc_shape = nn.Linear(6, int(self.latent_ch/4))
        self.enc_iou = nn.Linear(1, int(self.latent_ch/4))
        self.d_shape_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_shape_1 = nn.Linear(self.latent_ch, 6)
        self.d_iou_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_iou_1 = nn.Linear(self.latent_ch, 1)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))



    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def encode(self, data):
        ####### node level attributes and edge index
        x, edge_index, pos, size, pos_org, size_org = data.x, data.edge_index, data.node_pos, data.node_size, data.org_node_pos, data.org_node_size
        b_shape, b_iou = data.b_shape, data.b_iou
        b_shape = self.enc_shape(b_shape)
        b_iou = self.enc_iou(b_iou)
        shape_feature = torch.cat((b_shape, b_iou), 1)
        
        ###### graph level attributes
        org_asp_rto, org_long_side = data.asp_rto_gt.unsqueeze(1), data.long_side_gt.unsqueeze(1)            
        org_asp_rto = self.enc_asp_rto(org_asp_rto)
        org_long_side = self.enc_long_side(org_long_side)

        org_block_shape, org_block_scale = data.blockshape_latent_gt.view(-1, self.blockshape_latent_dim), data.block_scale_gt.unsqueeze(1)  
        org_block_shape = F.relu(self.enc_block_shape_0(org_block_shape))
        org_block_shape = self.enc_block_shape_1(org_block_shape)

        org_block_scale = self.enc_block_scale(org_block_scale)
        org_graph_feature = torch.cat((org_block_shape, org_block_scale), 1)


        batch_size = data.ptr.shape[0] - 1
        dat_size = x.shape[0]
        
        pos = F.relu(self.pos_init(pos_org))
        size = F.relu(self.size_init(size_org))

        x = self.ex_init(x)

        one_hot = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        x = torch.cat([x, one_hot], 1)

        
        ft = F.relu(self.ft_init(x)) # encoding existence

        n_embd_0 = torch.cat((shape_feature, size, pos, ft), 1)            

        n_embd_1 = F.relu(self.e_conv1(n_embd_0, edge_index))
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))
        
        g_embd_0 = self.global_pool(n_embd_0, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)


        g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3, org_graph_feature), 1)

        latent = self.aggregate(g_embd)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return [mu, log_var]



    def decode(self, z, edge_index):
        batch_size = z.shape[0]

        z = self.d_ft_init(z).view(z.shape[0] * self.N, -1)
    
        one_hot = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        z = torch.cat([z, one_hot], 1)

        d_embd_0 = F.relu(z)
        d_embd_1 = F.relu(self.d_conv1(d_embd_0, edge_index))
        d_embd_2 = F.relu(self.d_conv2(d_embd_1, edge_index))
        d_embd_3 = F.relu(self.d_conv3(d_embd_2, edge_index))
        
        exist = self.d_exist_1(d_embd_3)
    
        posx = F.relu(self.d_posx_0(d_embd_3))
        posx = self.d_posx_1(posx)

        posy = F.relu(self.d_posy_0(d_embd_3))
        posy = self.d_posy_1(posy)

        sizex = F.relu(self.d_sizex_0(d_embd_3))
        sizex = self.d_sizex_1(sizex)

        sizey = F.relu(self.d_sizey_0(d_embd_3))
        sizey = self.d_sizey_1(sizey)

        b_shape = F.relu(self.d_shape_0(d_embd_3))
        b_shape = self.d_shape_1(b_shape)

        b_iou = F.relu(self.d_iou_0(d_embd_3))
        b_iou = self.d_iou_1(b_iou)

        return exist, posx, posy, sizex, sizey, b_shape, b_iou



    def forward(self, data):
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        exist, posx, posy, sizex, sizey, b_shape, b_iou, = self.decode(z, data.edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)

        return exist, pos, size, mu, log_var, b_shape, b_iou





#######################################################################################################################
class NaiveBlockGenerator(BlockGenerator):
    def __init__(self, opt, N = 80, T=3, frequency_num = 32, f_act = 'relu'):    
        opt['enc_ex'] = False
        opt['pe'] = False
        opt['is_graphft_both'] = False
        opt['is_graphft_bottleneck'] = False

        super(NaiveBlockGenerator, self).__init__(opt, N, T, frequency_num, f_act)   

        self.aggregate = nn.Linear(int(self.latent_ch*(2.0 + self.T)), self.latent_dim)
        self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch)
    
    def encode(self, data):
        ####### node level attributes and edge index
        batch_size = data.ptr.shape[0] - 1
        x, edge_index, pos, size, pos_org, size_org = data.x, data.edge_index, data.node_pos, data.node_size, data.org_node_pos, data.org_node_size
        b_shape, b_iou = data.b_shape, data.b_iou
        b_shape = self.enc_shape(b_shape)
        b_iou = self.enc_iou(b_iou)
        shape_feature = torch.cat((b_shape, b_iou), 1)

        pos = pos_org
        size = size_org        

        pos = F.relu(self.pos_init(pos))
        size = F.relu(self.size_init(size))
        one_hot = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        x = torch.cat([x, one_hot], 1)
        ft = F.relu(self.ft_init(x)) # encoding existence

        n_embd_0 = torch.cat((shape_feature, size, pos, ft), 1) 

        n_embd_1 = F.relu(self.e_conv1(n_embd_0, edge_index))
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))
        
        g_embd_0 = self.global_pool(n_embd_0, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)

        g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3), 1)

        latent = self.aggregate(g_embd)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return [mu, log_var]


#######################################################################################################################################################
class AttentionBlockGenerator(BlockGenerator):
    def __init__(self, opt, N = 120, T=3, frequency_num = 32, f_act = 'relu'):
        super(AttentionBlockGenerator, self).__init__(opt, N, T, frequency_num, f_act)

        self.head = opt['head_num']
        self.blockshape_latent_dim = opt['block_latent_dim']

        use_head = ['GATConv', 'TransformerConv', 'GPSConv', 'GATv2Conv', 'SuperGATConv']

        if opt['convlayer'] == 'ChebConv':
            self.convlayer = torch_geometric.nn.ChebConv            
        elif opt['convlayer'] == 'SAGEConv':
            self.convlayer = torch_geometric.nn.SAGEConv
        elif opt['convlayer'] == 'GraphConv':
            self.convlayer = torch_geometric.nn.GraphConv
        elif opt['convlayer'] == 'GravNetConv':
            self.convlayer = torch_geometric.nn.GravNetConv
        elif opt['convlayer'] == 'GatedGraphConv':
            self.convlayer = torch_geometric.nn.GatedGraphConv
        elif opt['convlayer'] == 'ResGatedGraphConv':
            self.convlayer = torch_geometric.nn.ResGatedGraphConv     
        elif opt['convlayer'] == 'GATConv':
            self.convlayer = torch_geometric.nn.GATConv
        elif opt['convlayer'] == 'GATv2Conv':
            self.convlayer = torch_geometric.nn.GATv2Conv
        elif opt['convlayer'] == 'TransformerConv':
            self.convlayer = torch_geometric.nn.TransformerConv
        elif opt['convlayer'] == 'GPSConv':
            self.convlayer = torch_geometric.nn.GPSConv
        elif opt['convlayer'] == 'SuperGATConv':
            self.convlayer = torch_geometric.nn.SuperGATConv
            

        self.ft_init = nn.Linear(int(self.latent_ch/4) + N, int(self.latent_ch/2))

        if opt['convlayer'] in use_head:
            self.d_conv1 = self.convlayer((-1, self.latent_ch + N), self.latent_ch, heads = self.head)
        else:
            self.d_conv1 = self.convlayer((-1, self.latent_ch + N), self.latent_ch)
        

        if opt['convlayer'] in use_head:
            self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch, heads = self.head)
            self.e_conv2 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            self.e_conv3 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            self.d_conv2 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            self.d_conv3 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            
        else:
            self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch)
            self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
            self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)
            self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
            self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

            self.aggregate = nn.Linear(int(self.latent_ch* (2.5+self.head*self.T)), self.latent_dim)        



        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)


        if opt['convlayer'] in use_head:
            self.d_exist_1 = nn.Linear(self.latent_ch * self.head, 1)

            self.d_posx_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_posx_1 = nn.Linear(self.latent_ch, 1)

            self.d_posy_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_posy_1 = nn.Linear(self.latent_ch, 1)

            self.d_sizex_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_sizex_1 = nn.Linear(self.latent_ch, 1)

            self.d_sizey_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_sizey_1 = nn.Linear(self.latent_ch, 1)

            self.d_shape_0 = nn.Linear(self.latent_ch* self.head, self.latent_ch)
            self.d_shape_1 = nn.Linear(self.latent_ch, 6)
            self.d_iou_0 = nn.Linear(self.latent_ch* self.head, self.latent_ch)
            self.d_iou_1 = nn.Linear(self.latent_ch, 1)



        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))





#######################################################################################################################################################
class AttentionBlockGenerator_independent(AttentionBlockGenerator):
    def __init__(self, opt, N = 80, T=3, frequency_num = 32, f_act = 'relu'):
        super(AttentionBlockGenerator_independent, self).__init__(opt, N, T, frequency_num, f_act)
        
        use_head = ['GATConv', 'TransformerConv', 'GPSConv', 'GATv2Conv', 'SuperGATConv']
        
        if opt['convlayer'] in use_head:
            self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch, heads = self.head)
            self.aggregate = nn.Linear(int(self.latent_ch* (2.0+self.head*self.T)), self.latent_dim) 
        else:
            self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch)
            self.aggregate = nn.Linear(int(self.latent_ch* (2.0+self.T)), self.latent_dim) 
 
        self.d_ft_init = nn.Linear(self.latent_dim + 40, self.latent_ch * N)
        self.enc_block_scale = nn.Linear(1, 20)



    def encode(self, data):  
        ####### node level attributes and edge index
        x, edge_index, pos, size, pos_org, size_org = data.x, data.edge_index, data.node_pos, data.node_size, data.org_node_pos, data.org_node_size

        b_shape, b_iou = data.b_shape, data.b_iou
        b_shape = self.enc_shape(b_shape)
        b_iou = self.enc_iou(b_iou)
        shape_feature = torch.cat((b_shape, b_iou), 1)


        batch_size = data.ptr.shape[0] - 1
        dat_size = x.shape[0]

        pos = pos_org
        size = size_org
        
        pos = F.relu(self.pos_init(pos))
        size = F.relu(self.size_init(size))

        x = self.ex_init(x)

        one_hot = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        x = torch.cat([x, one_hot], 1)
        
        ft = F.relu(self.ft_init(x)) # encoding existence
        n_embd_0 = torch.cat((shape_feature, size, pos, ft), 1)            
        n_embd_1 = F.relu(self.e_conv1(n_embd_0, edge_index))
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))
        
        g_embd_0 = self.global_pool(n_embd_0, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)


        g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3), 1)

        latent = self.aggregate(g_embd)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return [mu, log_var]
    


    def decode(self, z, block_condition, edge_index):

        batch_size = z.shape[0]

        z = torch.cat((z, block_condition), 1)

        z = self.d_ft_init(z).view(z.shape[0] * self.N, -1)

        one_hot = torch.eye(self.N, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        z = torch.cat([z, one_hot], 1)
        
        d_embd_0 = F.relu(z)
        d_embd_1 = F.relu(self.d_conv1(d_embd_0, edge_index))
        d_embd_2 = F.relu(self.d_conv2(d_embd_1, edge_index))
        d_embd_3 = F.relu(self.d_conv3(d_embd_2, edge_index))

        exist = self.d_exist_1(d_embd_3)
    
        posx = F.relu(self.d_posx_0(d_embd_3))
        posx = self.d_posx_1(posx)

        posy = F.relu(self.d_posy_0(d_embd_3))
        posy = self.d_posy_1(posy)

        sizex = F.relu(self.d_sizex_0(d_embd_3))
        sizex = self.d_sizex_1(sizex)

        sizey = F.relu(self.d_sizey_0(d_embd_3))
        sizey = self.d_sizey_1(sizey)

        b_shape = F.relu(self.d_shape_0(d_embd_3))
        b_shape = self.d_shape_1(b_shape)

        b_iou = F.relu(self.d_iou_0(d_embd_3))
        b_iou = self.d_iou_1(b_iou)

        return exist, posx, posy, sizex, sizey, b_shape, b_iou



    def forward(self, data):
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        block_scale = self.enc_block_scale(data.block_scale_gt.unsqueeze(1))
        block_shape = data.blockshape_latent_gt.view(-1, self.blockshape_latent_dim)
        block_condition = torch.cat((block_shape, block_scale), 1)
        exist, posx, posy, sizex, sizey, b_shape, b_iou = self.decode(z, block_condition, data.edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)

        return exist, pos, size, mu, log_var, b_shape, b_iou




#######################################################################################################################################################
class AttentionBlockGenerator_independent_cnn(AttentionBlockGenerator_independent):
    def __init__(self, opt, N = 80, T=3, frequency_num = 32, f_act = 'relu', bottleneck = 128, image_size = 64, inner_channel = 80):
        super(AttentionBlockGenerator_independent_cnn, self).__init__(opt, N, T, frequency_num, f_act)

        channel_num = int((image_size / 2**4)**2 * inner_channel)
        self.inner_channel = 80
        self.image_size = 64
        self.linear1 = nn.Linear(channel_num, bottleneck)
        self.bottleneck = int(bottleneck)

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(2, int(self.inner_channel / 8), 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(int(self.inner_channel / 8), int(self.inner_channel / 4), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(int(self.inner_channel / 4), int(self.inner_channel / 2), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(int(self.inner_channel / 2), int(self.inner_channel), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
        )

        # self.aggregate = nn.Linear(int(self.latent_ch* (2.0+self.head*self.T)), self.latent_dim)  
        self.d_ft_init = nn.Linear(self.latent_dim + bottleneck, self.latent_ch * N)



    def cnn_encode(self, x):
        x = self.cnn_encoder(x)
        x = torch.flatten(x, 1)    
        x = self.linear1(x)
        return x


    def forward(self, data):
        batch_size = data.ptr.shape[0] - 1
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
        block_condition = self.cnn_encode(block_condition)
        exist, posx, posy, sizex, sizey, b_shape, b_iou = self.decode(z, block_condition, data.edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)

        return exist, pos, size, mu, log_var, b_shape, b_iou
