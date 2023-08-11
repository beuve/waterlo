import torch
import torchvision.transforms as T
import augly.image as imaugs


def add_jpeg_noise(im: torch.tensor, device, quality: int = 100):
    im_r = im.view((-1, *im.shape[-3:]))
    jpegs = []
    for i in range(im_r.shape[0]):
        pil = T.ToPILImage()(im_r[i])
        jpeg = imaugs.functional.encoding_quality(pil, quality=quality)
        jpegs.append(T.ToTensor()(jpeg))
    jpegs = torch.stack(jpegs).to(device)
    return im + (jpegs - im.detach())
