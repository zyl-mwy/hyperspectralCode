# https://zhuanlan.zhihu.com/p/361893823
import torch
import numpy as np
# import List

# def preprocess(self, batched_inputs: List[torch.Tensor]):
#         """
#             Args:
#               batch_inputs: 图片张量列表
#             Return:
#               padded_images: 填充后的批量图片张量
#               image_sizes_orig: 原始图片尺寸信息
#         """
#         ## 保留原始图片尺寸
#         image_sizes_orig = [[image.shape[-2], image.shape[-1]] for image in batched_inputs]
#         ## 找到最大尺寸
#         max_size = max([max(image_size[0], image_size[1]) for image_size in image_sizes_orig])
        
#         ## 构造批量形状 (batch_size, channel, max_size, max_size)
#         batch_shape = (len(batched_inputs), batched_inputs[0].shape[0], max_size, max_size)

#         padded_images = batched_inputs[0].new_full(batch_shape, 0.0)
#         for padded_img, img in zip(padded_images, batched_inputs):
#             h, w=img.shape[1:]
#             padded_img[..., :h, :w].copy_(img)

#         return padded_images, np.array(image_sizes_orig)

def postprocess(self, padded_images: torch.Tensor, feature_maps: torch.Tensor, image_sizes_orig: np.array):
        """
            Args:
                padded_images: 填充后的图片张量
                feature_maps: 特征图张量
                image_size_orig: 原图尺寸
        """
        padded_size = padded_images.shape[-2:]
        feature_size=feature_maps.shape[-2:]

        ratio = feature_size[0] / float(padded_size[0])
        image_sizes_on_feature = (image_sizes_orig * ratio).astype(np.int16)
        
        crops = []
        for image, size in zip(feature_maps, image_sizes_on_feature):
            size = (self.odd(size[0]), self.odd(size[1]))
            if size[0] <= 2 or size[1] <= 2:
                continue
            crop=image[:, :size[0], :size[1]]
            crops.append(crop)

        return crops