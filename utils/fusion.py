import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms
from scipy import signal, ndimage
import open3d as o3d
import numpy.linalg as lin
import scipy.optimize as opt

from layer import Layer
from utils.cloud_utils import make_bev_color, make_bev_height_intensity, make_front_proj
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from utils.cloud_utils import np_to_o3d_cloud
import segmentation_models_pytorch as smp
import torchvision.models as models
from torchviz import make_dot
from torchsummary import summary

from utils.math import block_mean


def to01(image):
    image += np.abs(np.min(image))
    image *= 1.0 / np.max(image)
    return image


def get_layers(model):
    layers = np.array([])
    for name, m in model.named_modules():
        layers = np.append(layers, name)

    layers = [layer for layer in layers if not len(layer) == 0]
    return layers


class Decomposer:
    def __init__(self, lam=5):
        self.lam = lam

    def find_detail(self, image):
        gx = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ])
        gy = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])

        convx = signal.convolve2d(image, gx, boundary='symm', mode='same')
        convy = signal.convolve2d(image, gy, boundary='symm', mode='same')
        conv = self.lam * (convx ** 2 + convy ** 2)
        conv = to01(conv)
        return conv

    def __call__(self, image_: np.array):
        image = image_.copy()
        # Obtain base part
        channels = image.shape[2]
        if np.max(image) > 1.0:
            # image = to01(image)
            image *= 1.0 / np.max(image)

        base = np.zeros((image.shape[0], image.shape[1], channels), dtype=image.dtype)
        detail = np.zeros((image.shape[0], image.shape[1], channels), dtype=image.dtype)
        for ch in range(channels):
            I_d = self.find_detail(image[:, :, ch])
            I_b = np.maximum(image[:, :, ch] - I_d, 0)
            base[:, :, ch] = I_b
            detail[:, :, ch] = I_d

        return base, detail


def find_density_channel(cloud):
    density = np.arrray([])

    return density


def add_channel(cloud_):
    cloud = np.zeros((cloud_.shape[0], cloud_.shape[1], 3), dtype=np.float32)
    cloud[:, :, 0] = cloud_[:, :, 0]
    cloud[:, :, 1] = cloud_[:, :, 1]
    cloud[:, :, 2] = np.maximum(cloud_[:, :, 0], cloud_[:, :, 1])
    cloud[:, :, 2] = to01(cloud[:, :, 2])
    # cloud[:, :, 2] = 0
    return cloud


def fuse_base(cloud: np.array, image: np.array, alpha1=0.5, alpha2=0.5):
    fused = np.zeros((cloud.shape[0], cloud.shape[1], cloud.shape[2]))
    fused[:, :, 0] = alpha1 * np.maximum(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]) + alpha2 * image[:, :, 0]
    fused[:, :, 1] = alpha1 * np.maximum(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]) + alpha2 * image[:, :, 1]
    fused[:, :, 2] = alpha1 * np.maximum(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]) + alpha2 * image[:, :, 2]

    return fused


class DetailFusion(nn.Module):

    def __init__(self, img_det1: np.array, img_det2: np.array):
        super(DetailFusion, self).__init__()
        img1 = img_det1.copy()
        img2 = img_det2.copy()

        self.np_img1 = np.reshape(img1, (img1.shape[2], img1.shape[1], img1.shape[0]))
        self.np_img2 = np.reshape(img2, (img2.shape[2], img2.shape[1], img2.shape[0]))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.encoder = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.encoder = self.encoder.to('cuda')

        tensor_shape = (img1.shape[1], img1.shape[0], img1.shape[2])
        self.img_det1, self.img_det2 = np.reshape(img1, tensor_shape), np.reshape(img2, tensor_shape)
        self.img_det1, self.img_det2 = self.transform(self.img_det1), self.transform(self.img_det2)

        graph_nodes = get_graph_node_names(self.encoder)[0]
        return_nodes = {
            f'{layer}': f'{layer}' for layer in graph_nodes[1:49]
        }
        self.feature_extractor = create_feature_extractor(self.encoder, return_nodes)

    def multi_layer_fusion(self, feature_maps1, feature_maps2):
        assert (feature_maps1.shape == feature_maps2.shape)

        res_map1 = res_map2 = np.array([])
        for layer1, layer2 in zip(feature_maps1, feature_maps2):
            Ci1 = np.linalg.norm(layer1.feature_map, axis=1, ord=1)
            Ci1 = np.reshape(Ci1, (Ci1.shape[1], Ci1.shape[2]))
            res_map1 = np.append(res_map1, [Layer(Ci1)])

            Ci2 = np.linalg.norm(layer2.feature_map, axis=1, ord=1)
            Ci2 = np.reshape(Ci2, (Ci2.shape[1], Ci2.shape[2]))
            res_map2 = np.append(res_map2, [Layer(Ci2)])

        block_size = 1
        weight_maps1 = np.array([])
        weight_maps2 = np.array([])
        for i in range(np.size(res_map1)):
            # TODO: check this
            old_shape = res_map1[i].shape
            res_map1[i].feature_map = block_mean(res_map1[i].feature_map, block_size)
            res_map2[i].feature_map = block_mean(res_map2[i].feature_map, block_size)
            assert (old_shape == res_map1[i].shape)

            # Apply softmax
            weight_map = np.zeros((2, res_map1[i].shape[0], res_map1[i].shape[1]))
            weight_map[0, :, :] = res_map1[i].feature_map
            weight_map[1, :, :] = res_map2[i].feature_map
            softmax = torch.nn.Softmax(dim=0)
            weight_map = np.reshape(weight_map, (weight_map.shape[1], weight_map.shape[2], weight_map.shape[0]))
            weight_map = torchvision.transforms.ToTensor()(weight_map)
            weight_map = softmax(weight_map)
            weight_map1 = weight_map[0, :, :]
            weight_map2 = weight_map[1, :, :]

            # Upsampling feature maps to original size
            # Add minibatch size.
            weight_map1 = weight_map1[None, :]
            weight_map2 = weight_map2[None, :]
            weight_map1 = weight_map1[None, :]
            weight_map2 = weight_map2[None, :]

            # Upsample
            upsample = torch.nn.Upsample(size=(self.img_det1.shape[1], self.img_det1.shape[2]))
            weight_map1 = upsample(weight_map1)
            weight_map2 = upsample(weight_map2)

            weight_maps1 = np.append(weight_maps1, [Layer(weight_map1.cpu().numpy())])
            weight_maps2 = np.append(weight_maps2, [Layer(weight_map2.cpu().numpy())])

        fused = np.array([])
        for weight_map1, weight_map2 in zip(weight_maps1, weight_maps2):
            w1 = weight_map1.feature_map
            w1 = np.reshape(w1, (w1.shape[1], w1.shape[2], w1.shape[3]))
            w2 = weight_map2.feature_map
            w2 = np.reshape(w2, (w2.shape[1], w2.shape[2], w2.shape[3]))

            # print(w1.shape)
            # print(self.np_img1.shape)
            # TODO: make more accurate multiplication
            # assert(w1.shape == self.np_img1.shape)

            fused1 = w1 * self.np_img1
            fused2 = w2 * self.np_img2
            fused = np.append(fused, [Layer(fused1 + fused2)])

        final_map = fused[0].feature_map
        for layer in fused:
            final_map = np.maximum(final_map, layer.feature_map)

        final_map = to01(final_map)

        return final_map

    def forward(self):
        input1 = self.img_det1[None]
        input2 = self.img_det2[None]

        input1 = input1.to('cuda')
        input2 = input2.to('cuda')

        map1 = self.feature_extractor(input1)
        map2 = self.feature_extractor(input2)
        # print(f"Before feature extractor: {input1}, after: {map1}")

        feature_maps1 = np.array([])
        feature_maps2 = np.array([])
        for _, layer in map1.items():
            feature_maps1 = np.append(feature_maps1, [Layer(layer.cpu().numpy())])
        for _, layer in map2.items():
            feature_maps2 = np.append(feature_maps2, [Layer(layer.cpu().numpy())])

        detail_fused = self.multi_layer_fusion(feature_maps1, feature_maps2)

        return detail_fused


class BevImageFusion(nn.Module):
    def __init__(self, params):
        super(BevImageFusion, self).__init__()
        self.bev_width = params['bev_width']
        self.bev_height = params['bev_height']
        self.width = params['width']
        self.height = params['height']
        self.resolution = params['resolution']
        self.roi = params['roi']
        self.extrinsic = params['extrinsic']
        self.intrinsic = params['intrinsic']

    def forward(self, cloud: np.array, image: np.array):
        rgbd_image = np_to_o3d_cloud(cloud).project_to_rgbd_image(self.width, self.height, self.intrinsic,
                                                                  self.extrinsic,
                                                                  depth_max=10000, depth_scale=1)
        rgbd_image.color = o3d.t.geometry.Image(image)
        color_cloud = o3d.t.geometry.PointCloud()
        color_cloud = color_cloud.create_from_rgbd_image(rgbd_image, self.intrinsic, self.extrinsic, depth_max=10000,
                                                         depth_scale=1)
        color_cloud = np.c_[color_cloud.point.positions.numpy(), color_cloud.point.colors.numpy()]
        bev_color = make_bev_color(color_cloud, self.resolution, self.bev_height, self.bev_width,
                                   self.roi)
        bev = make_bev_height_intensity(cloud, self.resolution, self.bev_height, self.bev_width,
                                        self.roi)

        base_cloud, detail_cloud = Decomposer()(bev)
        # bev_color = cv2.cvtColor(bev_color, cv2.COLOR_RGB2HSV)
        base_img, detail_img = Decomposer()(bev_color)
        # bev_color = cv2.cvtColor(bev_color, cv2.COLOR_HSV2RGB)

        base_cloud = add_channel(base_cloud)
        detail_cloud = add_channel(detail_cloud)
        fused_base = fuse_base(base_cloud, base_img)
        fused_detail = DetailFusion(detail_cloud, detail_img)()

        return {
            'bev': bev,
            'bev_color': bev_color,
            'base_cloud': base_cloud,
            'detail_cloud': detail_cloud,
            'fused_base': fused_base,
            'fused_detail': fused_detail
        }


class FrontImageFusion(nn.Module):
    def __init__(self, params):
        super(FrontImageFusion, self).__init__()
        self.width = params['width']
        self.height = params['height']
        self.intrinsic = params['intrinsic']
        self.extrinsic = params['extrinsic']

    def forward(self, cloud, image_):
        image = image_.copy()
        front, colors = make_front_proj(cloud, self.width, self.height, self.intrinsic, self.extrinsic)
        base_cloud, detail_cloud = Decomposer()(front)
        indices = np.argwhere(colors == 0)
        image[indices[:, 0], indices[:, 1], indices[:, 2]] = 0

        ## Get base and detail components of image
        ## Probably need to convert to HSV
        ## In HSV channels are independent from each other
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        base_img, detail_img = Decomposer()(image)
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        base_cloud = add_channel(base_cloud)
        detail_cloud = add_channel(detail_cloud)
        fused_base = fuse_base(base_cloud, base_img)

        detail_fusion = DetailFusion(detail_img, detail_cloud)
        fused_detail = detail_fusion()

        return {
            'front': front,
            'image': image,
            'base_image': base_img,
            'detail_image': detail_img,
            'base_cloud': base_cloud,
            'detail_cloud': detail_cloud,
            'fused_base': fused_base,
            'fused_detail': fused_detail
        }


class FusionLayer(nn.Module):
    """
    Fusion layer for image and point cloud
    Projection argument can be 'bev' or 'front'
    """

    def __init__(self, params: dict, projection='bev'):
        super(FusionLayer, self).__init__()
        assert (projection == 'bev' or projection == 'front')
        assert ('bev_width' in params)
        assert ('bev_height' in params)
        assert ('resolution' in params)
        assert ('roi' in params)
        assert ('width' in params)
        assert ('height' in params)
        assert ('intrinsic' in params)
        assert ('extrinsic' in params)

        if projection == 'bev':
            self.fusion = BevImageFusion(params)
        else:
            self.fusion = FrontImageFusion(params)

    def forward(self, cloud: np.array, image: np.array):
        return self.fusion(cloud, image)
