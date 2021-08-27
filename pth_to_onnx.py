
import torch,os,json,sys, onnxruntime
sys.path.append('/data/zs/zhangsheng/aicashier/pymodels/handtrack/pth_to_onnx') 
from builder import SiamRPNPPTemple, SiamRPNPPSearchPre, SiamRPNPPSearchPost

config = {'model_path': '', 'model_name': '', 'input_size':'', 'Templepp_onnx':'', 'SearchPre_onnx':'', 'SearchPost_onnx':''}
config_filename='/data/zs/zhangsheng/aicashier/pymodels/handtrack/pth_to_onnx/config_siamonnx_pre_post.json'
with open(config_filename) as conf_file:
    jload = json.load(conf_file)
    for key in config.keys():
        if key in jload:
            config[key] = jload[key]
# print(config)

def main(siam):
    if siam == 'SiamRPNPPTemple':
        model = SiamRPNPPTemple()
        input_names = ['z']
        output_names = ['zf']
        onnx_filename = config['Templepp_onnx']
        input_size = 127
        sample_inputs = (torch.zeros(1, 3, input_size, input_size, requires_grad=False))

    elif siam == 'SiamRPNPPSearchPre':
        model = SiamRPNPPSearchPre()
        input_names = ['zf', 'x']
        output_names = ['cls_kernel0', 'cls_search0', 'loc_kernel0', 'loc_search0', \
               'cls_kernel1', 'cls_search1', 'loc_kernel1', 'loc_search1', \
               'cls_kernel2', 'cls_search2', 'loc_kernel2', 'loc_search2']
        onnx_filename = config['SearchPre_onnx']
        input_size = 255
        sample_inputs = (torch.zeros(3, 1, 256, 7, 7, requires_grad=False), 
            torch.zeros(1, 3, input_size, input_size, requires_grad=False))

    else:
        model = SiamRPNPPSearchPost()
        input_names = ['cls_internal0', 'cls_internal1', 'cls_internal2','loc_internal0', 'loc_internal1', 'loc_internal2']
        output_names = ['cls','loc']
        onnx_filename = config['SearchPost_onnx']
        input_size = 25
        sample_inputs = ( torch.zeros(1, 256, input_size, input_size, requires_grad=False),
                        torch.zeros(1, 256, input_size, input_size, requires_grad=False),
                        torch.zeros(1, 256, input_size, input_size, requires_grad=False),
                        torch.zeros(1, 256, input_size, input_size, requires_grad=False),
                        torch.zeros(1, 256, input_size, input_size, requires_grad=False),
                        torch.zeros(1, 256, input_size, input_size, requires_grad=False))

    pretrained_dict = torch.load(config['model_path'])
    pretrained_dict = {'model.'+key: value for key, value in pretrained_dict.items()}
    required_params = {}
    for name, param in model.named_parameters():
        if name in pretrained_dict:
            required_params[name] = pretrained_dict[name]
    for name, param in model.named_buffers():
        if name in pretrained_dict:
            required_params[name] = pretrained_dict[name]
    model.load_state_dict(required_params)
    model.eval()
    # for i in list(required_params.keys()):
    #     print(i)
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: 'batch'}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch'}
    # print(input_names)
    torch.onnx.export(
        model=model.cpu(),
        args=sample_inputs,
        f=onnx_filename,
        export_params=True,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamic_axes = dynamic_axes)
    print('dynamic_axes',dynamic_axes)
    print('54')
    print(onnx_filename)

    providers = [
    # ('CUDAExecutionProvider', {
    #     'device_id': 0,
    #     'arena_extend_strategy': 'kNextPowerOfTwo',
    #     'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
    #     'cudnn_conv_algo_search': 'EXHAUSTIVE',
    #     'do_copy_in_default_stream': True,
    # }),
    'CPUExecutionProvider',
    ]

    test = onnxruntime.InferenceSession(onnx_filename, providers = providers)
    print('56')
    print([node.name for node in test.get_inputs()])


if __name__ =="__main__":
    main('SiamRPNPPTemple')
    main('SiamRPNPPSearchPre')
    main('SiamRPNPPSearchPost')


