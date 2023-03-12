import numpy as np
import scipy
from scipy import fftpack
from sklearn.metrics import mutual_info_score


def mi_dct(fmap1, fmap2):
    """
    Calculate Mutual Information between feature maps fmap1 and fmap2
    using features from discrete cosine transformation
    :param fmap1:
    :param fmap2:
    :return:
    """

    mui = 0
    for ch in range(fmap1.shape[2]):
        dct1 = fftpack.dct(fmap1[:, :, ch])
        dct2 = fftpack.dct(fmap2[:, :, ch])

        dct1 = dct1.flatten()
        dct2 = dct2.flatten()
        diff = np.abs(np.size(dct1) - np.size(dct2))
        dct2 = np.pad(dct2, (0, diff), 'constant', constant_values=(4, 6))
        mui += np.sum(mutual_info_score(dct1, dct2))

    return mui


def fmi_dct(fmap1, fmap2, fused):
    """
    Calculate Feature Mutual Information score between feature maps fmap1 and fmap2
    and fused map fused using features from discrete cosine transformation
    :param fmap1:
    :param fmap2:
    :param fused:
    :return:
    """
    mu1 = mi_dct(fmap1, fused)
    mu2 = mi_dct(fmap2, fused)

    return mu1 + mu2


def shanon_entropy(data_):
    info = 0
    for ch in range(data_.shape[2]):
        data = fftpack.dct(data_[:, :, ch])
        data = data.flatten()
        info += scipy.stats.entropy(data)

    return info
