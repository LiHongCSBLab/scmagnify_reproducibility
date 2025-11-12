import csv
import yaml
import os
import re

class NoQuotesDumper(yaml.Dumper):
    def represent_scalar(self, tag, value, style=None):
        if tag == 'tag:yaml.org,2002:str':
            # 如果是可以解析为数字的字符串，不加引号
            try:
                float(value)
                return self.represent_scalar('tag:yaml.org,2002:float', value)
            except ValueError:
                pass
        return super().represent_scalar(tag, value, style)

# 读取 CSV 并生成 YAML 配置文件
def generate_yaml(csv_file, output_dir):
    # 创建输出目录（如果没有的话）
    os.makedirs(output_dir, exist_ok=True)

    # 打开 CSV 文件
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')
        
        # 获取表头（参数名）
        header = next(reader)
        
        # 遍历 CSV 每一行，生成对应的 YAML 配置文件
        for row in reader:
            # 跳过空行或无效行
            if not row or len(row) != len(header):
                continue
            
            trail = row[0]  # 第一列是 trail，假设为数字
            params = {header[i]: row[i] for i in range(1, len(header))}
            
            # 如果参数值是数字，转换为整数或浮点数，但保留数组形式的字符串
            for key in params:
                if re.match(r"^\[.*\]$", params[key]):  # 跳过数组形式的字符串
                    continue
                try:
                    if "." in params[key]:
                        params[key] = float(params[key])  # 转为浮点数
                    else:
                        params[key] = int(params[key])  # 转为整数
                except ValueError:
                    pass  # 保留非数字的字符串值

            # 构建配置字典
            config = {"trail": int(trail), **params}
            
            # 输出 YAML 文件路径
            yaml_file = os.path.join(output_dir, f"trail_{trail}.yaml")
            
            # 写入 YAML 文件
            with open(yaml_file, 'w') as yaml_out:
                yaml.dump(config, yaml_out, Dumper=NoQuotesDumper, default_flow_style=False)
            print(f"Generated YAML for trail {trail}: {yaml_file}")


# 示例用法
if __name__ == "__main__":
    csv_file = "grn_trails.csv"  # 输入 CSV 文件路径
    output_dir = "./conf/grn_params"  # 输出目录
    
    generate_yaml(csv_file, output_dir)



