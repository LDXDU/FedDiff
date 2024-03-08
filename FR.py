import torch
import torch.nn as nn


class FR(nn.Module):
    def __init__(self, low_channels, high_channels, c_kernel=3, r_kernel=3, use_att=False, use_process=True):
        super(FR, self).__init__()

        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        self.non_local_att = nn.Conv2d
        if self.l_c == self.h_c:
            print('Channel checked!')
        else:
            raise ValueError('Low and Hih channels need to be the same!')
        self.dcn_row = nn.Conv2d(self.l_c, self.h_c, kernel_size=self.r_k, stride=1, padding=self.r_k // 2)
        self.dcn_colum = nn.Conv2d(self.l_c, self.h_c, kernel_size=self.c_k, stride=1, padding=self.c_k // 2)
        self.sigmoid = nn.Sigmoid()
        if self.att == True:
            self.csa = self.non_local_att(self.l_c, self.h_c, 1, 1, 0)
        else:
            self.csa = None
        if use_process == True:
            self.preprocess = nn.Sequential(nn.Conv2d(self.l_c, self.h_c // 2, 1, 1, 0),
                                            nn.Conv2d(self.h_c // 2, self.l_c, 1, 1, 0))
        else:
            self.preprocess = None

    def forward(self, a_low, a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)
        else:
            a_low = a_low
            a_high = a_high

        a_low_c = self.dcn_colum(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        a_low_r = self.dcn_row(a_low)
        a_low_rw = self.sigmoid(a_low_r)
        a_low_rw = a_low_rw * a_high
        a_row = a_low + a_low_rw

        if self.csa is not None:
            a_FR = self.csa(a_row + a_colum)
        else:
            a_FR = a_row + a_colum
        return a_FR


if __name__ == '__main__':
    ########Test FR
    img_low = torch.randn(1, 2048, 4, 4)
    img_high = torch.randn(1, 2048, 4, 4)
    a = FR(2048, 2048)
    feature = a(img_low, img_high)
    print(feature.size())