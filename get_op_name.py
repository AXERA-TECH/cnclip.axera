import onnx


def get_filtered_op_names(onnx_file_path, keyword):
    # 加载 ONNX 模型
    model = onnx.load(onnx_file_path)
    filtered_op_names = []
    # 遍历图中的所有节点
    for node in model.graph.node:
        if keyword in node.name:
            # 构建操作名称和类型的字符串
            op_name = f"{node.name}:{node.op_type}"
            filtered_op_names.append(op_name)
    return filtered_op_names


def save_to_txt(filtered_op_names, output_txt_path):
    with open(output_txt_path, 'w') as f:
        f.write('"layer_names": [\n')
        for i, name in enumerate(filtered_op_names):
            if i == len(filtered_op_names) - 1:
                f.write(f'          "{name}"\n')
            else:
                f.write(f'          "{name}",\n')
        f.write(']')


if __name__ == "__main__":

    onnx_file_path = 'cnclip.axera/build_output/image_encoder/frontend/optimized.onnx'
    
    output_txt_path = 'MatMul_qk.txt'
    keyword = 'MatMul_qk_'
    
    # 获取筛选后的操作名称
    filtered_op_names = get_filtered_op_names(onnx_file_path, keyword)
    # 将筛选后的操作名称保存到文本文件中
    save_to_txt(filtered_op_names, output_txt_path)
    