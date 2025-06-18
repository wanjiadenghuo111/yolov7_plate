import torch
import onnx
import argparse
from pathlib import Path
import torch.nn as nn
from plate_recognition.plateNet import myNet_ocr_color

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/plate_rec_color.pth', help='模型权重路径')
    parser.add_argument('--output', type=str, default='weights/plate_rec_color.onnx', help='ONNX输出路径')
    parser.add_argument('--img-size', nargs='+', type=int, default=[48, 168], help='输入图像尺寸[H, W]')
    parser.add_argument('--batch-size', type=int, default=1, help='批量大小')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='设备(cpu/cuda)')
    parser.add_argument('--dynamic', action='store_true', help='启用动态轴')
    return parser.parse_args()

def export_onnx(opt):
    # 1. 加载字符集和颜色类别
    plateName = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
    num_classes = len(plateName)
    color_num = 5
    
    device = torch.device(opt.device)
    
    # 2. 加载权重与配置
    check_point = torch.load(opt.weights, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point.get('cfg', None)  # 从权重中读取cfg
    
    # 3. 若cfg不存在，根据权重形状推断（示例配置，需根据实际调整）
    if cfg is None:
        print("警告：权重文件中未找到cfg，使用推断的配置")
        cfg = [8, 8, 16, 16, 'M', 32, 32, 'M', 48, 48, 'M', 64, 64]
    
    # 4. 初始化模型（使用推断的cfg）
    model = myNet_ocr_color(
        num_classes=num_classes,
        export=True,
        cfg=cfg,
        color_num=color_num
    )
    model = model.to(device)
    
    # 5. 筛选可匹配的权重
    matched_state = {}
    for name, param in model_state.items():
        if name in model.state_dict() and model.state_dict()[name].shape == param.shape:
            matched_state[name] = param
        else:
            print(f"跳过不匹配层: {name}, 模型形状: {model.state_dict()[name].shape}, 权重形状: {param.shape}")
    
    # 6. 加载匹配的权重
    model.load_state_dict(matched_state, strict=False)
    model.eval()
    
    # 7. 准备输入张量
    img = torch.zeros(opt.batch_size, 3, opt.img_size[0], opt.img_size[1]).to(device)
    
    # 8. 导出ONNX
    onnx_path = opt.output
    input_names = ['input']
    output_names = ['char_sequence', 'color_classification']
    
    dynamic_axes = None
    if opt.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 3: 'width'},
            'char_sequence': {0: 'sequence_length', 1: 'batch_size'},
            'color_classification': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model, 
        img, 
        onnx_path,
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # 9. 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX导出成功: {onnx_path}")
        with torch.no_grad():
            char_output, color_output = model(img)
            print(f"字符输出形状: {char_output.shape}")
            print(f"颜色输出形状: {color_output.shape}")
    except Exception as e:
        print(f"ONNX验证失败: {e}")

if __name__ == '__main__':
    opt = parse_opt()
    export_onnx(opt)