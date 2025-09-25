import torch
from moment.utils.masking import Masking
from moment.models.layers.revin import RevIN
import torch.fft as fft
def dataconversion(original_ts, configs, input_mask=None, mask=None):
    """
    original_ts: [B, C, T] shaped torch.Tensor, normalized time series

    Returns:
        tuple of 3 torch.Tensors: (t_2, f_1, f_2)
    """
    device = original_ts.device  # 기준 디바이스 설정

    # 정규화 및 마스킹 도구 초기화
    mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))
    normalizer = RevIN(num_features=1, affine=configs.getattr("revin_affine", False)).to(device)

    # 마스크 생성 및 디바이스 정렬
    if mask is None:
        mask = mask_generator.generate_mask(x=original_ts, input_mask=input_mask)
    mask = mask.to(device)
    input_mask = input_mask.to(device) if input_mask is not None else torch.ones_like(mask)
    mask_t2 = input_mask
    # 시계열 정규화 (시간 도메인)
    t_2 = normalizer(x=original_ts, mask=mask * input_mask, mode="norm")

    # 주파수 도메인 변환 및 정규화
    f_1 = fft.fft(original_ts).abs()
    f_1 = f_1.to(device)
    # f_1 정규화
    dummy_mask_f1 = torch.ones(f_1.shape[0], f_1.shape[2], dtype=torch.bool, device=f_1.device)
    f_1 = normalizer(x=f_1, mask=dummy_mask_f1, mode="norm")

    # 주파수 변환 후 추가 처리 및 정규화
    f_2 = DataTransform_FD(f_1)
    f_2 = f_2.to(device)
    dummy_mask_f2 = torch.ones(f_2.shape[0], f_2.shape[2], dtype=torch.bool, device=f_2.device)
    f_2 = normalizer(x=f_2, mask=dummy_mask_f2, mode="norm")

    return t_2, f_1, f_2, mask_t2, dummy_mask_f1, dummy_mask_f2

def DataTransform_FD(sample):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape, device=x.device)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix