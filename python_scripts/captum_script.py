import numpy as np
import yaml
import torch
import random
from captum.attr import IntegratedGradients
import pandas as pd
import sys
from move.models.base import reload_vae



class VAE_Latent(torch.nn.Module):
    def __init__(self, vae_model):
        super(VAE_Latent, self).__init__()
        self.encoder = vae_model.encoder
        self.model = vae_model

    def forward(self, x):
        mu, _ = self.model.encode(x)
        return mu


class VAE_Decoder(torch.nn.Module):
    def __init__(self, vae_model):
        super(VAE_Decoder, self).__init__()
        self.decoder = vae_model.decoder 
        self.model = vae_model  
    
    def forward(self, z):
        x_recon, *_ = self.model.decode(z)
        return x_recon


def compute_captum(config_path, model_path, sample_size):

    random.seed(6)
    np.random.seed(6)
    torch.manual_seed(6)

    # extract continuous & categorical feature names
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print('loaded config', flush=True)

    cont, cat = [], []
    for item in config['continuous_inputs']:
        cont = cont+[item['name']]
    for item in config['categorical_inputs']:
        cat = cat+[item['name']]

    # reload pretrained model and print model info
    model = reload_vae(model_path)
    model.cuda()
    model.eval()

    latent_model = VAE_Latent(model).cuda()
    latent_decoder = VAE_Decoder(model).cuda()
    
    # Use Integrated Gradients for interpretability
    ig_e, ig_d = IntegratedGradients(latent_model), IntegratedGradients(latent_decoder)

    # Load all data
    all_names = []
    big_tensor = None
    interim_data_path = config['interim_data_path']
    for name in cat+cont:
        dataset = torch.load(f'{interim_data_path}/{name}.pt', weights_only=False)

        if len(dataset['tensor'].shape)<3:
            all_names = all_names + dataset['feature_names']

        elif len(dataset['tensor'].shape)>2:
            col_new = [f'{i}_{j}' for i in dataset['feature_names'] for j in range(0,dataset['tensor'].shape[-1])]
            all_names = all_names + col_new


        if big_tensor is not None:
            if len(dataset['tensor'].shape)<3 :
                big_tensor = torch.cat((big_tensor, dataset['tensor']), 1)  
            elif len(dataset['tensor'].shape)>2:
                big_tensor = torch.cat((big_tensor, dataset['tensor'].reshape(dataset['tensor'].size()[0], -1)), 1)
            else:
                print('COULD NOT CONCATENATE')
        else: 
            if len(dataset['tensor'].shape)<3 :
                big_tensor = dataset['tensor']
            elif len(dataset['tensor'].shape)>2:
                big_tensor = dataset['tensor'].reshape(dataset['tensor'].size()[0], -1)
            

    print('dataset loaded', flush=True)
    print(f'dataset size: {big_tensor.shape}', flush=True)
    

    ## Compute and store (or load) correlation matrix of all input features
    # Convert tensor to numpy array
    data_np = pd.DataFrame(data=big_tensor[:,:], columns=all_names[:])
    if len(all_names)<1000:
        print('starting input correlation...', flush=True)
        results_path = config['results_path']
        correlation_matrix = data_np.corr()
        correlation_matrix.to_csv(f"{results_path}/input_correlation_bash.csv")
        print('Input correlation computed', flush=True)


    input_data = big_tensor
    
    # Randomly downsample patients
    if sample_size is not None:
        indices = torch.randperm(big_tensor.shape[0])[:sample_size]
        input_data = big_tensor[indices]
    
    input_data.to('cuda')
    
    # get latent representation
    z_loc, z_scale = model.encode(input_data)
    z = model.reparameterize(z_loc, z_scale)
    z = z.to('cuda')

    # Load or compute attributions, compute and plot correlations of encoder and decoder attributions (over input / output features)
    print('starting encoder attributions...', flush=True)

    attr_matrix_e = None
    for n in range(128):
        print(n, flush=True)
        attr = ig_e.attribute(input_data, target=n)
        attr = attr.cpu().detach().numpy()
        if attr_matrix_e is not None:
            attr_matrix_e = np.vstack((attr_matrix_e, np.mean(abs(attr), axis=0)))
            #attr_matrix_e = np.vstack((attr_matrix_e, np.mean(attr, axis=0)))
        else:
            attr_matrix_e = np.mean(abs(attr), axis=0)
            #attr_matrix_e = np.mean(attr, axis=0)
    ame_df = pd.DataFrame(attr_matrix_e)
    ame_df.to_csv(f"{results_path}/attributes_e_a_{sample_size}.csv", index=False, header=False) 
    attr_matrix_e_df = pd.read_csv(f"{results_path}/attributes_e_a_{sample_size}.csv", header=None, names=all_names)
    attr_matrix_e= attr_matrix_e_df.to_numpy()
    print('encoder attributions computed, starting decoder...', flush=True)


    attr_matrix_d = None
    print('check if correct input data: ', input_data.shape[1], flush=True)
    for j in range(input_data.shape[1]):
        print(j, flush=True)
        attr = ig_d.attribute((z), target=j)
        attr = attr.cpu().detach().numpy()
        if attr_matrix_d is not None:
            attr_matrix_d = np.vstack((attr_matrix_d, np.mean(abs(attr), axis=0)))
            #attr_matrix_d = np.vstack((attr_matrix_d, np.mean(attr, axis=0)))
        else:
            attr_matrix_d = np.mean(abs(attr), axis=0)
            #attr_matrix_d = np.mean(attr, axis=0)
    try:
        attr_matrix_d = attr_matrix_d.detach().numpy()
    except:
        pass
    amd_df = pd.DataFrame(attr_matrix_d)
    amd_df.to_csv(f"{results_path}/attributes_d_a_{sample_size}.csv", index=False, header=False) 
    print('decoder attr. stored as csv', flush=True) 
    attr_matrix_d_df = pd.read_csv(f"{results_path}/attributes_d_a_{sample_size}.csv", header=None)
    attr_matrix_d_df['names'] = all_names
    attr_matrix_d_df = attr_matrix_d_df.set_index('names')
    attr_matrix_d= attr_matrix_d_df.to_numpy()
    
    print('decoder attributions computed, starting correlations...', flush=True)

    correlation_e = attr_matrix_e_df.corr()
    correlation_e.to_csv(f"{results_path}/encoder_correlation_a_{sample_size}.csv")

    correlation_d = attr_matrix_d_df.transpose().corr()
    correlation_d.to_csv(f"{results_path}/decoder_correlation_a_{sample_size}.csv")
    print('correlations done.', flush=True)


if __name__ == "__main__":
    print('in main', flush=True)
    config_path = sys.argv[1]
    model_path = sys.argv[2]
    try:
        sample_size = int(sys.argv[3])
    except:
        sample_size = None
    compute_captum(config_path, model_path, sample_size)
