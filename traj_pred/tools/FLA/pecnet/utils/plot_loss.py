import torch
import numpy as np
import matplotlib.pyplot as plt


# dada_datasets = ['A2D', 'A2E', 'C2B', 'C2E', 'D2B', 'D2E', 'E2B']
# dada_datasets = ['A2B', 'A2C', 'A2D', 'A2E', 'B2A', 'B2C', 'B2D', 'B2E', 'C2A', 'C2B']
dada_datasets = [
    'A2B', 'A2C', 'A2D', 'A2E',
    'B2A', 'B2C', 'B2D', 'B2E',
    'C2A', 'C2B', 'C2D', 'C2E',
    'D2A', 'D2B', 'D2C', 'D2E',
    'E2A', 'E2B', 'E2C', 'E2D']

for subset in dada_datasets:
    root = '../checkpoint_DADA/after_SocialPooling/'
    load_file = f"PECNET_social_model_{subset}_DADA.pt"
    checkpoint = torch.load(root+load_file, map_location=torch.device('cpu'))
    metrics = checkpoint['metrics']


    train_loss = np.array(metrics['train_loss'])
    print(len(train_loss))
    d_loss = torch.tensor(metrics['d_loss']).detach().numpy()
    g_loss = torch.tensor(metrics['g_loss']).detach().numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(train_loss)
    ax[1].plot(d_loss)
    ax[1].plot(g_loss)

    ax[0].legend(['train_loss'])
    ax[1].legend(['d_loss', 'g_loss'])
    plt.savefig(root+f'{subset}_loss.png')
    # print(train_loss.shape, d_loss.shape)

# for subset in dada_datasets:
#     root = '../checkpoint_DLA/'
#     load_file = f'{subset}_loss.npy'
#     train_loss = np.load(root+load_file)
#
#     fig, ax = plt.subplots()
#     ax.plot(train_loss)
#
#     ax.legend(['train_loss'])
#     plt.savefig(root+f'{subset}_loss.png')
#     # print(train_loss.shape, d_loss.shape)

