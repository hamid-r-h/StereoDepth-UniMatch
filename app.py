import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.onnx
from unimatch.unimatch import UniMatch
from utils.flow_viz import flow_to_image
from dataloader.stereo import transforms
from utils.visualization import vis_disparity
from torch import onnx

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Move model loading outside inference for efficiency
def load_model(task='stereo'):
    """Load model and checkpoint"""
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task=task
    )
    
    checkpoint_path = 'pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth'
    checkpoint_flow = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_flow['model'], strict=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    return model


@torch.no_grad()
def inference(image1, image2, model, task='stereo'):
    """Inference on an image pair for optical flow or stereo disparity prediction"""
    padding_factor = 32
    attn_type = 'self_swin2d_cross_swin1d'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 3
    # smaller inference size for faster speed
    max_inference_size = [640, 960]


    image1 = np.array(image1).astype(np.float32)
    image2 = np.array(image2).astype(np.float32)

    if len(image1.shape) == 2:  # gray image
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

 
    val_transform_list = [transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]

    val_transform = transforms.Compose(val_transform_list)

    sample = {'left': image1, 'right': image2}
    sample = val_transform(sample)
    image1 = sample['left'].unsqueeze(0)  # [1, 3, H, W]
    image2 = sample['right'].unsqueeze(0)  # [1, 3, H, W]
    print(image1.shape)

    # Move to GPU if available
    if torch.cuda.is_available():
        image1 = image1.cuda()
        image2 = image2.cuda()

    nearest_size = [int(torch.ceil(torch.tensor(image1.size(-2)) / padding_factor)) * padding_factor,
                    int(torch.ceil(torch.tensor(image1.size(-1)) / padding_factor)) * padding_factor]

    inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # Resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

    results_dict = model(image1, image2,
                         attn_type=attn_type,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         num_reg_refine=num_reg_refine,
                         task=task,
                         )

    #flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]

    pred_disp = F.interpolate(results_dict.unsqueeze(1), size=ori_size, mode='bilinear', align_corners=True).squeeze(1)  # [1, H, W]
    pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    disp = pred_disp[0].cpu()

    output = vis_disparity(disp.numpy(), return_rgb=True)

    return disp

# Load model for ONNX conversion
def load_model_for_conversion(task='stereo'):
    """Load model and checkpoint for ONNX conversion"""
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task=task
    )
    
    checkpoint_path = 'pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth'
    checkpoint_flow = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_flow['model'], strict=True)
    
    model.eval()
    return model

# Convert model to ONNX
def convert_to_onnx(onnx_file_path,image1, image2, task='stereo'):
    """Inference on an image pair for optical flow or stereo disparity prediction"""
    padding_factor = 32
    attn_type = 'self_swin2d_cross_swin1d'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 3
    # smaller inference size for faster speed
    max_inference_size = [640, 960]
    """Convert PyTorch model to ONNX format"""
    if torch.cuda.is_available():
       torch.cuda.empty_cache()
    model = load_model_for_conversion(task='stereo')

    # Example input - Adjust according to your model's input shape
    batch_size = 1
    height = 640
    width = 960
    input_channels = 3
    dummy_input1 = torch.randn(batch_size, input_channels, height, width)  # Static input sizes
    print(dummy_input1.shape)
    dummy_input2 = torch.randn(batch_size, input_channels, height, width)
    # Move inputs to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input1 = dummy_input1.cuda()
        dummy_input2 = dummy_input2.cuda()
    with torch.no_grad():

        # Adjusted torch.onnx.export call
        torch.onnx.export(
            model, 
            (dummy_input1, dummy_input2),  # model inputs (passed separately as two arguments)
            onnx_file_path,                # output file
            export_params=True,            # store the trained parameter weights inside the model file
            opset_version=15,              # ONNX version
            do_constant_folding=True,      # optimize by folding constant nodes
            input_names=['left_image', 'right_image'],  # input tensor names
            output_names=['output'],       # output tensor name
            dynamic_axes={'left_image': {0: 'batch_size'},  # support variable batch sizes
                          'right_image': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
   
    print(f"Model has been successfully converted to {onnx_file_path}")

    print(f"Model has been successfully converted to {onnx_file_path}")


