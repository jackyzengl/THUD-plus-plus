import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.nn import functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
# import yaml

'''MLP model'''


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class PECNet(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, non_local_theta_size,
                 non_local_phi_size, non_local_g_size, fdim, zdim, nonlocal_pools, non_local_dim, sigma, past_length,
                 future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()

        self.zdim = zdim
        self.nonlocal_pools = nonlocal_pools
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim=past_length * 2, output_dim=fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim=2, output_dim=fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim=2 * fdim, output_dim=2 * zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim=fdim + zdim, output_dim=2, hidden_size=dec_size)

        self.non_local_theta = MLP(input_dim=2 * fdim + 2, output_dim=non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim=2 * fdim + 2, output_dim=non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim=2 * fdim + 2, output_dim=2 * fdim + 2, hidden_size=non_local_g_size)

        self.predictor = MLP(input_dim=2 * fdim + 2, output_dim=2 * (future_length - 1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))

            print("Non Local Theta architecture : {}".format(architecture(self.non_local_theta)))
            print("Non Local Phi architecture : {}".format(architecture(self.non_local_phi)))
            print("Non Local g architecture : {}".format(architecture(self.non_local_g)))

    def non_local_social_pooling(self, feat, mask):
        """
        args:
            feat: [bs, 2*fdim+2]
            mask: [bs, bs]
        return:
            pooled_f + feat: [bs, 2*dim+2]
        """

        # N,C
        theta_x = self.non_local_theta(feat)  # [bs, non_local_dim]

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1, 0)  # [non_local_dim, bs]

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim=-1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x, initial_pos, dest=None, mask=None, device=torch.device('cpu')):
        """
        args:
            x:[batchsize, 8*2] ; it has been multiply by the "data_scale"
            initial_pos: [batchsize, 2]; it has not been multiply by the "data_scale" but by the "1/1000"
            dest: [batchsize, 2] ; it has been multiply by the "data_scale"
            mask: [batchsize, batchsize] ;
        return:
            train:
                generated_dest: [bs, 2]
                mu: [bs, 2*zdim], CVAE均值
                logvar: [bs, 2*zdim], CVAE方差
                pred_future: [bs, 2 * (future_length - 1)]
            val/test:
                generated_dest: [bs, 2]
        """

        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None)
        assert self.training ^ (mask is None)

        # encode
        ftraj = self.encoder_past(x)

        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)  # [batchsize, zdim]
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features), dim=1)
            latent = self.encoder_latent(features)  # [bs, 2*zdim]

            mu = latent[:, 0:self.zdim]  # 2-d array
            logvar = latent[:, self.zdim:]  # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim=1)
        generated_dest = self.decoder(decoder_input)  # [bs, 2]

        if self.training:
            # prediction in training, no best selection
            generated_dest_features = self.encoder_dest(generated_dest)

            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim=1)  # [bs, 2*fdim+2]

            for i in range(self.nonlocal_pools):
                # non local social pooling
                prediction_features = self.non_local_social_pooling(prediction_features, mask)

            pred_future = self.predictor(prediction_features)  # [bs, 2 * (future_length - 1)]
            return generated_dest, mu, logvar, pred_future

        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest, mask, initial_pos):
        """
        args:
            generated_dest: [bs, 2], 是多次采样后筛选的每个人的最优destination
        return:
            interpolated_future: [bs, 2 * (future_length - 1)]
        """
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim=1)

        for i in range(self.nonlocal_pools):
            # non local social pooling
            prediction_features = self.non_local_social_pooling(prediction_features, mask)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future


class PECNet_FLA(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, non_local_theta_size,
                 non_local_phi_size, non_local_g_size, fdim, zdim, nonlocal_pools, non_local_dim, sigma, past_length,
                 future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet_FLA, self).__init__()

        self.zdim = zdim
        self.nonlocal_pools = nonlocal_pools
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim=past_length * 2, output_dim=fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim=2, output_dim=fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim=2 * fdim, output_dim=2 * zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim=fdim + zdim, output_dim=2, hidden_size=dec_size)

        self.non_local_theta = MLP(input_dim=2 * fdim + 2, output_dim=non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim=2 * fdim + 2, output_dim=non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim=2 * fdim + 2, output_dim=2 * fdim + 2, hidden_size=non_local_g_size)

        self.predictor = MLP(input_dim=2 * fdim + 2, output_dim=2 * (future_length - 1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))

            print("Non Local Theta architecture : {}".format(architecture(self.non_local_theta)))
            print("Non Local Phi architecture : {}".format(architecture(self.non_local_phi)))
            print("Non Local g architecture : {}".format(architecture(self.non_local_g)))

    def non_local_social_pooling(self, feat, mask):
        """
        args:
            feat: [bs, 2*fdim+2]
            mask: [bs, bs]
        return:
            pooled_f + feat: [bs, 2*dim+2]
        """

        # N,C
        theta_x = self.non_local_theta(feat)  # [bs, non_local_dim]

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1, 0)  # [non_local_dim, bs]

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim=-1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x, initial_pos, dest=None, mask=None, device=torch.device('cpu')):
        """
        args:
            x:[batchsize, 8*2] ; it has been multiply by the "data_scale"
            initial_pos: [batchsize, 2]; it has not been multiply by the "data_scale" but by the "1/1000"
            dest: [batchsize, 2] ; it has been multiply by the "data_scale"
            mask: [batchsize, batchsize] ;
        return:
            train:
                generated_dest: [bs, 2]
                mu: [bs, 2*zdim], CVAE均值
                logvar: [bs, 2*zdim], CVAE方差
                pred_future: [bs, 2 * (future_length - 1)]
            val/test:
                generated_dest: [bs, 2]
        """

        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None)
        assert self.training ^ (mask is None)

        # encode
        ftraj = self.encoder_past(x)

        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)  # [batchsize, zdim]
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features = self.encoder_dest(dest)
            features = torch.cat((ftraj, dest_features), dim=1)
            # prediction_encoding = features.clone()
            latent = self.encoder_latent(features)  # [bs, 2*zdim]

            mu = latent[:, 0:self.zdim]  # 2-d array
            logvar = latent[:, self.zdim:]  # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim=1)
        # prediction_encoding = decoder_input.clone()
        generated_dest = self.decoder(decoder_input)  # [bs, 2]

        if self.training:
            # prediction in training, no best selection
            generated_dest_features = self.encoder_dest(generated_dest)

            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim=1)  # [bs, 2*fdim+2]
            # prediction_encoding = prediction_features.clone()

            for i in range(self.nonlocal_pools):
                # non local social pooling
                prediction_features = self.non_local_social_pooling(prediction_features, mask)

            prediction_encoding = prediction_features.clone()

            pred_future = self.predictor(prediction_features)  # [bs, 2 * (future_length - 1)]
            return generated_dest, mu, logvar, pred_future, prediction_encoding

        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest, mask, initial_pos):
        """
        args:
            generated_dest: [bs, 2], 是多次采样后筛选的每个人的最优destination
        return:
            interpolated_future: [bs, 2 * (future_length - 1)]
        """
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim=1)

        for i in range(self.nonlocal_pools):
            # non local social pooling
            prediction_features = self.non_local_social_pooling(prediction_features, mask)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future


class Discriminator_inlay(nn.Module):
    def __init__(self,
                 input_dim=34,
                 mlp_dim=68,
                 ):
        super(Discriminator_inlay, self).__init__()

        # self.attn_layer = SelfAttn_module()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self,
                feature_in,  # fake or real [bs, 2*fdim+2]
                ):
        feature_out = self.mlp_layers(feature_in)  # [bs, 1]
        return feature_out  # fake_score, [bs, 1]