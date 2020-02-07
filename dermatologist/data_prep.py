# %% Imports
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from utils import *

# %% Load data
%time data = load_data()

# %% Histograms
data.dx_type.value_counts().plot(kind='bar')
plt.show()
data.sex.value_counts().plot(kind='bar')
plt.show()
data.localization.value_counts().plot(kind='bar')
plt.show()
data.classification.value_counts().plot(kind='bar')
plt.show()

# %% Range of image sizes - seems to be a mono-sized set.
data['image_meta'].value_counts()

# %% Show figure fom each category - source Kaggle.
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(
    m_axs, data.sort_values(['classification']).groupby('classification')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(
        n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)


# %% Colour information - source Kaggle.
rgb_info_df = data.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v in
                                  zip(['Red', 'Green', 'Blue'],
                                      np.mean(x['image'], (0, 1)))}),1)
gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1)
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
rgb_info_df['Gray_mean'] = gray_col_vec
rgb_info_df.sample(3)

#
for c_col in rgb_info_df.columns:
    data[c_col] = rgb_info_df[c_col].values # we cant afford a copy

#
sns.pairplot(data[['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean', 'classification']],
             hue='classification', plot_kws = {'alpha': 0.5})

# %%
from skimage.util import montage
rgb_stack = np.stack(data.\
                     sort_values(['classification',])['image'].\
                     map(lambda x: np.array(x)[::5, ::5]).values, 0)
rgb_montage = np.stack([montage(rgb_stack[:, :, :, i]) for i in range(rgb_stack.shape[3])], -1)
print(rgb_montage.shape)

# %%
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20), dpi=300)
ax1.imshow(rgb_montage)
fig.savefig('nice_montage.png')

# %%
n_samples = 5
for sample_col in ['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean']:
    fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
    def take_n_space(in_rows, val_col, n):
        s_rows = in_rows.sort_values([val_col])
        s_idx = np.linspace(0, s_rows.shape[0]-1, n, dtype=int)
        return s_rows.iloc[s_idx]
    for n_axs, (type_name, type_rows) in zip(m_axs,
                                             data.sort_values(['classification']).groupby('classification')):

        for c_ax, (_, c_row) in zip(n_axs,
                                    take_n_space(type_rows,
                                                 sample_col,
                                                 n_samples).iterrows()):
            c_ax.imshow(c_row['image'])
            c_ax.axis('off')
            c_ax.set_title('{:2.2f}'.format(c_row[sample_col]))
        n_axs[0].set_title(type_name)
    fig.savefig('{}_samples.png'.format(sample_col), dpi=300)

# %% Dump data
dump_data(data)
