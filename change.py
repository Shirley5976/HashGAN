def create_nolabel_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            image_path = parts[0]
            labels = ['0'] * (len(parts) - 1)
            new_line = f"{image_path} {' '.join(labels)}\n"
            outfile.write(new_line)

# 使用示例
input_file = './data/cifar10/database.txt'  # 输入的label文件路径
output_file = './data/cifar10/database_nolabel.txt'  # 输出的nolabel文件路径

create_nolabel_file(input_file, output_file)
