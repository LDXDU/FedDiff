import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as dataf
from scipy import io

""" Training dataset"""


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class HSIDataLoader(object):
    def __init__(self, param={}) -> None:
        self.data_param = param.get('data', {})
        self.data = None  # 原始读入X数据 shape=(h,w,c)
        self.labels = None  # 原始读入Y数据 shape=(h,w,1)

        # 参数设置
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.patch_size = self.data_param.get('patch_size', 32)  # n * n
        self.padding = self.data_param.get('padding', True)  # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', False)
        self.batch_size = self.data_param.get('batch_size', 256)
        self.select_spectral = self.data_param.get('select_spectral', [])  # [] all spectral selected

        self.squzze = True

        self.split_row = 0
        self.split_col = 0

        self.light_split_ori_shape = None
        self.light_split_map = []

    def load_data(self):
        data, labels = None, None
        if self.data_sign == "Indian":
            data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
            labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_sign == "Pavia":
            data = sio.loadmat('../data/PaviaU.mat')['paviaU']
            labels = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']
        elif self.data_sign == "Houston":
            data = sio.loadmat('../data/Houston.mat')['img']
            labels = sio.loadmat('../data/Houston_gt.mat')['Houston_gt']
        else:
            pass
        print("ori data load shape is", data.shape, labels.shape)
        if len(self.select_spectral) > 0:  # user choose spectral himself
            data = data[:, :, self.select_spectral]
        return data, labels

    def get_ori_data(self):
        return np.transpose(self.data, (2, 0, 1)), self.labels

    def _padding(self, X, margin=2):
        # pading with zeros
        w, h, c = X.shape
        new_x, new_h, new_c = w + margin * 2, h + margin * 2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x + w, start_y:start_y + h, :] = X
        return returnX

    def get_patches_by_light_split(self, X, Y, patch_size=1):
        h, w, c = X.shape
        row = h // patch_size
        if h % patch_size != 0:
            row += 1
        col = w // patch_size
        if w % patch_size != 0:
            col += 1
        res = np.zeros((row * col, patch_size, patch_size, c))
        self.light_split_ori_shape = X.shape
        resY = np.zeros((row * col))
        index = 0
        for i in range(row):
            for j in range(col):
                start_row = i * patch_size
                if start_row + patch_size > h:
                    start_row = h - patch_size
                start_col = j * patch_size
                if start_col + patch_size > w:
                    start_col = w - patch_size

                res[index, :, :, :] = X[start_row:start_row + patch_size, start_col:start_col + patch_size, :]
                self.light_split_map.append(
                    [index, start_row, start_row + patch_size, start_col, start_col + patch_size])
                index += 1
        return res, resY

    def reconstruct_image_by_light_split(self, inputX, pathch_size=1):
        '''
        input shape is (batch, h, w, c)
        '''
        assert self.light_split_ori_shape is not None
        ori_h, ori_w, ori_c = self.light_split_ori_shape
        batch, h, w, c = inputX.shape
        assert batch == len(self.light_split_map)  # light_split_map必须与batch值相同
        X = np.zeros((ori_h, ori_w, c))
        for tup in self.light_split_map:
            index, start_row, end_row, start_col, end_col = tup
            X[start_row:end_row, start_col:end_col, :] = inputX[index, :, :, :]
        return X

    def get_patches_by_split(self, X, Y, patch_size=1):
        h, w, c = X.shape
        row = h // patch_size
        col = w // patch_size
        newX = X
        res = np.zeros((row * col, patch_size, patch_size, c))
        resY = np.zeros((row * col))
        index = 0
        for i in range(row):
            for j in range(col):
                res[index, :, :, :] = newX[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                index += 1
        self.split_row = row
        self.split_col = col
        return res, resY

    def split_to_big_image(self, splitX):
        '''
        input splitX shape (batch, 1, spe, h, w)
        return newX shape (spe, bigh, bigw)
        '''
        patch_size = self.patch_size
        batch, channel, spe, h, w = splitX.shape
        assert self.split_row * self.split_col == batch
        newX = np.zeros((spe, self.split_row * patch_size, self.split_col * patch_size))
        index = 0
        for i in range(self.split_row):
            for j in range(self.split_col):
                index = i * self.split_col + j
                newX[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = splitX[index, 0, :,
                                                                                                    :, :]
        return newX

    def re_build_split(self, X_patches, patch_size):
        '''
        X_pathes shape is (batch, channel=1, spectral, height, with)
        return shape is (height, width, spectral)
        '''
        h, w, c = self.data.shape
        row = h // patch_size
        if h % patch_size > 0:
            row += 1
        col = w // patch_size
        if w % patch_size > 0:
            col += 1
        newX = np.zeros((c, row * patch_size, col * patch_size))
        for i in range(row):
            for j in range(col):
                newX[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = X_patches[
                                                                                                    i * col + j, 0, :,
                                                                                                    :, :]
        return np.transpose(newX, (1, 2, 0))

    def get_patches(self, X, Y, patch_size=1, remove_zero=False):
        w, h, c = X.shape
        # 1. padding
        margin = (patch_size - 1) // 2
        if self.padding:
            X_padding = self._padding(X, margin=margin)
        else:
            X_padding = X

        # 2. zero patchs
        temp_w, temp_h, temp_c = X_padding.shape
        row = temp_w - patch_size + 1
        col = temp_h - patch_size + 1
        X_patchs = np.zeros((row * col, patch_size, patch_size, c))  # one pixel one patch with padding
        Y_patchs = np.zeros((row * col))
        patch_index = 0
        for r in range(0, row):
            for c in range(0, col):
                temp_patch = X_padding[r:r + patch_size, c:c + patch_size, :]
                X_patchs[patch_index, :, :, :] = temp_patch
                patch_index += 1

        if remove_zero:
            X_patchs = X_patchs[Y_patchs > 0, :, :, :]
            Y_patchs = Y_patchs[Y_patchs > 0]
            Y_patchs -= 1
        return X_patchs, Y_patchs  # (batch, w, h, c), (batch)

    def custom_process(self, data):
        '''
        pavia数据集 增加一个光谱维度 从103->104 其中第104维为103的镜像维度
        data shape is [h, w, spe]
        '''

        if self.data_sign == "Pavia":
            h, w, spe = data.shape
            new_data = np.zeros((h, w, spe + 1))
            new_data[:, :, :spe] = data
            new_data[:, :, spe] = data[:, :, spe - 1]
            return new_data
        return data

    def generate_torch_dataset(self, split=False, light_split=False):
        # 1. 根据data_sign load data
        self.data, self.labels = self.load_data()

        # 1.1 norm化
        norm_data = np.zeros(self.data.shape)
        for i in range(self.data.shape[2]):
            input_max = np.max(self.data[:, :, i])
            input_min = np.min(self.data[:, :, i])
            norm_data[:, :, i] = (self.data[:, :, i] - input_min) / (input_max - input_min) * 2 - 1  # [-1,1]

        print('[data] load data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))
        self.data = norm_data

        # 1.2 专门针对特殊的数据集补充或删减一些光谱维度
        norm_data = self.custom_process(norm_data)

        # 2. 获取patchs
        if not split and not light_split:
            X_patchs, Y_patchs = self.get_patches(norm_data, self.labels, patch_size=self.patch_size, remove_zero=False)
            print('[data not split] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))
        elif split:
            X_patchs, Y_patchs = self.get_patches_by_split(norm_data, self.labels, patch_size=self.patch_size)
            print('[data split] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))
        elif light_split:
            X_patchs, Y_patchs = self.get_patches_by_light_split(norm_data, self.labels, patch_size=self.patch_size)
            print(
                '[data light split] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))

        # 4. 调整shape来满足torch使用
        X_all = X_patchs.transpose((0, 3, 1, 2))
        X_all = np.expand_dims(X_all, axis=1)
        print('------[data] after transpose train, test------')
        print("X.shape=", X_all.shape)
        print("Y.shape=", Y_patchs.shape)

        trainset = TrainDS(X_all, Y_patchs)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   )
        return train_loader, X_all, Y_patchs


def MFTDataset(data_path, datasetName, split="train"):
    data1Name = datasetName
    data2Name = "LiDAR"
    HSI = io.loadmat(data_path + "/" + data1Name + '11x11/HSI_Tr.mat')
    TrainPatch = HSI['Data']
    TrainPatch = TrainPatch.astype(np.float32)
    NC = TrainPatch.shape[3]  # NC is number of bands

    LiDAR = io.loadmat(data_path + '/' + data1Name + '11x11/' + data2Name + '_Tr.mat')
    TrainPatch2 = LiDAR['Data']
    TrainPatch2 = TrainPatch2.astype(np.float32)
    NCLiDAR = TrainPatch2.shape[3]  # NC is number of bands

    label = io.loadmat(data_path + '/' + data1Name + '11x11/TrLabel.mat')
    TrLabel = label['Data']

    HSI = io.loadmat(data_path + '/' + data1Name + '11x11/HSI_Te.mat')
    TestPatch = HSI['Data']
    TestPatch = TestPatch.astype(np.float32)

    LiDAR = io.loadmat(data_path + '/' + data1Name + '11x11/' + data2Name + '_Te.mat')
    TestPatch2 = LiDAR['Data']
    TestPatch2 = TestPatch2.astype(np.float32)

    label = io.loadmat(data_path + '/' + data1Name + '11x11/TeLabel.mat')
    TsLabel = label['Data']

    TrainPatch1 = torch.from_numpy(TrainPatch).to(torch.float32)
    # TrainPatch1 = (TrainPatch1 - torch.min(TrainPatch1)) / (torch.max(TrainPatch1) - torch.min(TrainPatch1))
    # TrainPatch1 = (TrainPatch1 - 0.5) / 0.5
    # print(TrainPatch1.shape)
    TrainPatch1 = TrainPatch1.permute(0, 3, 1, 2)
    # TrainPatch1 = TrainPatch1.reshape(TrainPatch1.shape[0],TrainPatch1.shape[1],-1).to(torch.float32)
    TrainPatch2 = torch.from_numpy(TrainPatch2).to(torch.float32)
    TrainPatch2 = TrainPatch2.permute(0, 3, 1, 2)
    # TrainPatch2 = (TrainPatch2 - torch.min(TrainPatch2)) / (torch.max(TrainPatch2) - torch.min(TrainPatch2))
    # TrainPatch2 = (TrainPatch2 - 0.5) / 0.5
    # TrainPatch2 = TrainPatch2.reshape(TrainPatch2.shape[0],TrainPatch2.shape[1],-1).to(torch.float32)
    TrainLabel1 = torch.from_numpy(TrLabel[0]) - 1
    TrainLabel1 = TrainLabel1.long()
    # TrainLabel1 = TrainLabel1.reshape(-1)

    TestPatch1 = torch.from_numpy(TestPatch).to(torch.float32)
    TestPatch1 = TestPatch1.permute(0, 3, 1, 2)
    # TestPatch1 = (TestPatch1 - torch.min(TestPatch1)) / (torch.max(TestPatch1) - torch.min(TestPatch1))
    # TestPatch1 = (TestPatch1 - 0.5) / 0.5
    # TestPatch1 = TestPatch1.reshape(TestPatch1.shape[0],TestPatch1.shape[1],-1).to(torch.float32)
    TestPatch2 = torch.from_numpy(TestPatch2).to(torch.float32)
    TestPatch2 = TestPatch2.permute(0, 3, 1, 2)
    # TestPatch2 = (TestPatch2 - torch.min(TestPatch2)) / (torch.max(TestPatch2) - torch.min(TestPatch2))
    # TestPatch2 = (TestPatch2 - 0.5) / 0.5
    # TestPatch2 = TestPatch2.reshape(TestPatch2.shape[0],TestPatch2.shape[1],-1).to(torch.float32)
    TestLabel1 = torch.from_numpy(TsLabel[0]) - 1
    TestLabel1 = TestLabel1.long()
    # TestLabel1 = TestLabel1.reshape(-1)
    print(TrainPatch1.shape)
    print(TestPatch1.shape)

    Classes = len(np.unique(TrainLabel1))
    if split == "train":
        # dataset = dataf.TensorDataset(TestPatch1, TestPatch2, TestLabel1)
        dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel1)
    else:
        dataset = dataf.TensorDataset(TestPatch1, TestPatch2, TestLabel1)
        # dataset = dataf.TensorDataset(TrainPatch1,TrainPatch2, TrainLabel1)
    return dataset


def MFTDataset_77(data_path, datasetName, split="train"):
    data1Name = datasetName
    data2Name = "LiDAR"
    HSI = io.loadmat(data_path + "/" + data1Name + '/HSI_TrSet.mat')
    TrainPatch = HSI['HSI_TrSet']
    TrainPatch = TrainPatch.astype(np.float32).reshape((-1, 7, 7, 144))
    NC = TrainPatch.shape[3]  # NC is number of bands

    LiDAR = io.loadmat(data_path + '/' + data1Name + '/' + data2Name + '_TrSet.mat')
    TrainPatch2 = LiDAR['LiDAR_TrSet']
    TrainPatch2 = TrainPatch2.astype(np.float32).reshape((-1, 7, 7, 21))
    print(TrainPatch2.shape)
    NCLiDAR = TrainPatch2.shape[3]  # NC is number of bands

    label = io.loadmat(data_path + '/' + data1Name + '/TrLabel.mat')
    TrLabel = label['TrLabel'][:, 0] - 1
    TrLabel = np.eye(15)[TrLabel]

    HSI = io.loadmat(data_path + '/' + data1Name + '/HSI_TeSet.mat')
    TestPatch = HSI['HSI_TeSet']
    TestPatch = TestPatch.astype(np.float32).reshape((-1, 7, 7, 144))

    LiDAR = io.loadmat(data_path + '/' + data1Name + '/' + data2Name + '_TeSet.mat')
    TestPatch2 = LiDAR['LiDAR_TeSet']
    TestPatch2 = TestPatch2.astype(np.float32).reshape((-1, 7, 7, 21))

    label = io.loadmat(data_path + '/' + data1Name + '/TeLabel.mat')
    TsLabel = label['TeLabel'][:, 0] - 1
    TsLabel = np.eye(15)[TsLabel]

    TrainPatch1 = torch.from_numpy(TrainPatch).to(torch.float32)
    TrainPatch1 = (TrainPatch1 - torch.min(TrainPatch1)) / (torch.max(TrainPatch1) - torch.min(TrainPatch1))
    TrainPatch1 = (TrainPatch1 - 0.5) / 0.5
    # print(TrainPatch1.shape)
    TrainPatch1 = TrainPatch1.permute(0, 3, 1, 2)
    # TrainPatch1 = TrainPatch1.reshape(TrainPatch1.shape[0],TrainPatch1.shape[1],-1).to(torch.float32)
    TrainPatch2 = torch.from_numpy(TrainPatch2).to(torch.float32)
    TrainPatch2 = TrainPatch2.permute(0, 3, 1, 2)
    TrainPatch2 = (TrainPatch2 - torch.min(TrainPatch2)) / (torch.max(TrainPatch2) - torch.min(TrainPatch2))
    TrainPatch2 = (TrainPatch2 - 0.5) / 0.5
    # TrainPatch2 = TrainPatch2.reshape(TrainPatch2.shape[0],TrainPatch2.shape[1],-1).to(torch.float32)
    TrainLabel1 = torch.from_numpy(TrLabel)
    TrainLabel1 = TrainLabel1.long()
    # TrainLabel1 = TrainLabel1.reshape(-1)

    TestPatch1 = torch.from_numpy(TestPatch).to(torch.float32)
    TestPatch1 = TestPatch1.permute(0, 3, 1, 2)
    TestPatch1 = (TestPatch1 - torch.min(TestPatch1)) / (torch.max(TestPatch1) - torch.min(TestPatch1))
    TestPatch1 = (TestPatch1 - 0.5) / 0.5
    TestPatch2 = torch.from_numpy(TestPatch2).to(torch.float32)
    TestPatch2 = TestPatch2.permute(0, 3, 1, 2)
    TestPatch2 = (TestPatch2 - torch.min(TestPatch2)) / (torch.max(TestPatch2) - torch.min(TestPatch2))
    TestPatch2 = (TestPatch2 - 0.5) / 0.5
    TestLabel1 = torch.from_numpy(TsLabel)
    TestLabel1 = TestLabel1.long()
    # TestLabel1 = TestLabel1.reshape(-1)
    print(TrainPatch1.shape)
    print(TestPatch1.shape)

    Classes = len(np.unique(TrainLabel1))
    if split == "train":
        # dataset = dataf.TensorDataset(TestPatch1, TestPatch2, TestLabel1)
        dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel1)
    else:
        dataset = dataf.TensorDataset(TestPatch1, TestPatch2, TestLabel1)
        # dataset = dataf.TensorDataset(TrainPatch1,TrainPatch2, TrainLabel1)
    return dataset


if __name__ == "__main__":
    # dataloader = HSIDataLoader({"data":{"padding":False, "select_spectral":[1,99,199]}})
    # train_loader = dataloader.generate_torch_dataset()
    # train_loader,X,Y = dataloader.generate_torch_dataset(split=True)
    # newX = dataloader.re_build_split(X, dataloader.patch_size)
    # print(newX.shape)

    # dataloader = HSIDataLoader(
    #    {"data":{"data_sign":"Pavia", "padding":False, "batch_size":256, "patch_size":16, "select_spectral":[]}})
    # train_loader,X,Y = dataloader.generate_torch_dataset(light_split=True)
    # print(X.shape)

    dataloader = HSIDataLoader(
        {"data": {"data_sign": "Houston", "padding": False, "batch_size": 256, "patch_size": 16,
                  "select_spectral": []}})
    train_loader, X, Y = dataloader.generate_torch_dataset(light_split=True)
    print(X.shape)
