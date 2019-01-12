import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# transefer-learning vs fine-tuning

df_tf = pd.read_csv('../hists/transfered/512-avg-0.5vs1-v2.log.csv')
df_ft = pd.read_csv('../hists/fine-tuned/512-avg-0.5vs1-15.log.csv')
df_rd = pd.read_csv('../hists/fine-tuned/512-avg-0.5vs1-random.log.csv')

# %%
# output2

tf_acc = df_tf.loc[:, ['epoch', 'output2_acc', 'val_output2_acc']]
ft_acc = df_ft.loc[:, ['epoch', 'output2_acc', 'val_output2_acc']]
rd_acc = df_rd.loc[:, ['epoch', 'output2_acc', 'val_output2_acc']]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(17, 10), tight_layout=False)
axes.hlines(y=0.9074, xmin=0, xmax=100, colors='pink', label='SVM train')
axes.hlines(y=0.5288, xmin=0, xmax=100, colors='cyan', label='SVM valid')


tf_acc.plot(x='epoch', y='output2_acc', ax=axes, label='transfered train')
ft_acc.plot(x='epoch', y='output2_acc', ax=axes, label='fine-tuned train')
rd_acc.plot(x='epoch', y='output2_acc', ax=axes, label='randomed train')

tf_acc.plot(x='epoch', y='val_output2_acc', ax=axes, label='transfered valid')
ft_acc.plot(x='epoch', y='val_output2_acc', ax=axes, label='fine-tuned valid')
rd_acc.plot(x='epoch', y='val_output2_acc', ax=axes, label='randomed valid')

plt.xlabel('epoch', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.legend(fontsize=20, ncol=2)
plt.title('VGG16 Architecture - accuracy for output 2 (city classification)', fontsize=20)
plt.tight_layout()
plt.savefig('./transfer-learning_vs._fine-tuning_output2.png')
plt.show()

# %%
# output1

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(17, 10), tight_layout=False)
tf_acc = df_tf.loc[:, ['epoch', 'output1_acc', 'val_output1_acc']]
ft_acc = df_ft.loc[:, ['epoch', 'output1_acc', 'val_output1_acc']]
rd_acc = df_rd.loc[:, ['epoch', 'output1_acc', 'val_output1_acc']]

tf_acc.plot(x='epoch', y='output1_acc', ax=axes, label='transfered train')
ft_acc.plot(x='epoch', y='output1_acc', ax=axes, label='fine-tuned train')
rd_acc.plot(x='epoch', y='output1_acc', ax=axes, label='randomed train')
axes.hlines(y=0.9599, xmin=0, xmax=100, colors='pink', label='SVM train')

tf_acc.plot(x='epoch', y='val_output1_acc', ax=axes, label='transfered valid')
ft_acc.plot(x='epoch', y='val_output1_acc', ax=axes, label='fine-tuned valid')
rd_acc.plot(x='epoch', y='val_output1_acc', ax=axes, label='randomed valid')
axes.hlines(y=0.8188, xmin=0, xmax=100, colors='cyan', label='SVM valid')

plt.xlabel('epoch', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.legend(fontsize=20, ncol=2)
plt.title('VGG16 Architecture - accuracy for output 1 (Eastern vs. Western classification)', fontsize=20)
plt.tight_layout()
plt.savefig('./transfer-learning_vs._fine-tuning_output1.png')
plt.show()

# %%
# Places vs. ImageNet

df_im = pd.read_csv('../hists/fine-tuned/512-avg-0.5vs1-imagenet-15.log.csv')

# %%
# output2

ft_acc = df_ft.loc[:, ['epoch', 'output2_acc', 'val_output2_acc']]
im_acc = df_im.loc[:, ['epoch', 'output2_acc', 'val_output2_acc']]
rd_acc = df_rd.loc[:, ['epoch', 'output2_acc', 'val_output2_acc']]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(17, 10), tight_layout=False)

ft_acc.plot(x='epoch', y='output2_acc', ax=axes, label='Places train')
im_acc.plot(x='epoch', y='output2_acc', ax=axes, label='ImageNet train')
rd_acc.plot(x='epoch', y='output2_acc', ax=axes, label='randomed train')

ft_acc.plot(x='epoch', y='val_output2_acc', ax=axes, label='Places valid')
im_acc.plot(x='epoch', y='val_output2_acc', ax=axes, label='ImageNet valid')
rd_acc.plot(x='epoch', y='val_output2_acc', ax=axes, label='randomed valid')

plt.xlabel('epoch', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.legend(fontsize=20, ncol=2)
plt.title('VGG16 Architecture - accuracy for output 2 (city classification)', fontsize=20)
plt.tight_layout()
plt.savefig('./Places_vs._ImageNet_output2.png')
plt.show()


# %%
# output1

ft_acc = df_ft.loc[:, ['epoch', 'output1_acc', 'val_output1_acc']]
im_acc = df_im.loc[:, ['epoch', 'output1_acc', 'val_output1_acc']]
rd_acc = df_rd.loc[:, ['epoch', 'output1_acc', 'val_output1_acc']]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(17, 10), tight_layout=False)

ft_acc.plot(x='epoch', y='output1_acc', ax=axes, label='Places train')
im_acc.plot(x='epoch', y='output1_acc', ax=axes, label='ImageNet train')
rd_acc.plot(x='epoch', y='output1_acc', ax=axes, label='randomed train')

ft_acc.plot(x='epoch', y='val_output1_acc', ax=axes, label='Places valid')
im_acc.plot(x='epoch', y='val_output1_acc', ax=axes, label='ImageNet valid')
rd_acc.plot(x='epoch', y='val_output1_acc', ax=axes, label='randomed valid')

plt.xlabel('epoch', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.legend(fontsize=20, ncol=2)
plt.title('VGG16 Architecture - accuracy for output 1 (Eastern vs. Western classification)', fontsize=20)
plt.tight_layout()
plt.savefig('./Places_vs._ImageNet_output1.png')
plt.show()
