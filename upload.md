# 0. 登录
ssh <user>@<server>

# 1. 创建并激活环境
conda create -n cifar10_resnet_env_zh python=3.9.21 -y
conda activate cifar10_resnet_env_zh

# 2. 安装 PyTorch + 科学计算依赖（conda 版本）
看自己原本是什么版本的 PyTorch，这里以 2.5.0 为例
conda install pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy==2.0.2 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.6.1 \
             matplotlib==3.9.4 seaborn==0.13.2 tqdm==4.67.1 -y

# 3. 安装 pip-check-reqs
确保终端激活了 conda 环境后，安装下列命令，如开头有所示（cifar10_resnet_env_zh）[PATH]>...
pip install pip-check-reqs

# 4. 进入项目目录（假设已上传解压）
cd ~/your_project_dir

# 5. 创建精简版 requirements.txt
# 在项目根目录下创建 requirements.txt，内容如下：注意中英文符号区别
numpy==2.0.2
matplotlib==3.9.4
pandas==2.2.3
scikit-learn==1.6.1
seaborn==0.13.2
torch==2.5.0+cu121
torchvision==0.20.0+cu121
tqdm==4.67.1
scipy==1.13.1

# 6. 保证环境与 requirements 一致，用conda安装依赖也行，**记得确保当前环境是刚刚创建的全新环境并且在对应执行命令的终端激活了**
pip install -r requirements.txt

# 7. 检查依赖完备性（以根目录 + 忽略 tests 为例）
pip-missing-reqs --ignore-file=tests/* .
pip-extra-reqs   --ignore-file=tests/* .

# 8. 运行训练脚本
可以先把项目根目录下创建data和results文件夹，data文件夹用于存放数据集，results文件夹用于存放训练结果
python train.py \--model ResNet20 \ --batch-size 256 \--epochs 100