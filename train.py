import random
import gc
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import csv

from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===================== 参数解析 =====================
parser = argparse.ArgumentParser(description='CIFAR-10 / CINIC-10 Training & Finetuning with ResNet')

parser.add_argument('--dataset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'CIFAR100', 'Cinic10', 'TinyImageNet', 'Food101', 'Caltech256'],
                    help='Dataset to train/fine-tune on')

parser.add_argument('--model', type=str, default='ResNet18',
                    choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                             'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202'],
                    help='Choose ResNet model to train')

parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size for training')

parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate')

parser.add_argument('--epochs', type=int, default=164,
                    help='Total training epochs')

parser.add_argument('--results-dir', type=str, default='./results',
                    help='Directory to save results')

parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of worker threads for data loading')

parser.add_argument('--valid', type=bool, default=True,
                    help='Whether to run validation')

parser.add_argument('--pretrained-path', type=str, default=None,
                    help='预训练模型 pth 路径（如: results/ResNet18/ResNet18_best.pth）')

parser.add_argument(
    '--resume-path',
    type=str,
    default=None,
    help='从检查点恢复训练的路径（.pth），会加载模型和优化器状态并从对应 epoch 继续'
)

# 显式布尔类型参数：--freeze-backbone True/False
parser.add_argument(
    '--freeze-backbone',
    type=bool,
    default=False,
    help='是否开启分阶段微调：True=按阶段冻结/解冻 backbone，False=全网常规训练'
)

args = parser.parse_args()


# ===================== 一些辅助函数 =====================
def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    train_losses, val_losses, train_accs, val_accs, lr_history,
                    extra=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "lr_history": lr_history,
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved to {path} (epoch={epoch})")


def visualize_predictions_random(model, dataset, classes, device, num_images=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_images)

    fig = plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for idx, img_idx in enumerate(indices):
            image, label = dataset[img_idx]  # dataset 返回的是 (Tensor, label)
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

            ax = plt.subplot(2, 5, idx + 1)
            ax.set_title(
                f'Pred: {classes[pred.item()]}\nTrue: {classes[label]}')
            ax.axis('off')

            # 反归一化（和原来一致）
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            inp = image.numpy().transpose((1, 2, 0))
            inp = std.numpy() * inp + mean.numpy()
            inp = np.clip(inp, 0, 1)

            ax.imshow(inp)


def get_unique_path(base_path: str) -> str:
    """
    若 `base_path` 已存在，则自动追加 _1, _2 ... 后缀，返回一个未存在的路径。
    """
    if not os.path.exists(base_path):
        return base_path

    dir_name, file_name = os.path.split(base_path)
    name, ext = os.path.splitext(file_name)

    idx = 1
    while True:
        new_name = f"{name}_{idx}{ext}"
        candidate = os.path.join(dir_name, new_name)
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def visualize_predictions(model, data_loader, classes, device, num_images=10):
    """Visualize model predictions（保留原函数以防仍有调用）"""
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return

                images_so_far += 1
                ax = plt.subplot(2, 5, images_so_far)
                ax.set_title(
                    f'Pred: {classes[preds[j]]}\nTrue: {classes[labels[j]]}')
                ax.axis('off')

                # Denormalize image
                mean = torch.tensor([0.4914, 0.4822, 0.4465])
                std = torch.tensor([0.2023, 0.1994, 0.2010])
                inp = inputs.cpu()[j].numpy().transpose((1, 2, 0))
                inp = std.numpy() * inp + mean.numpy()
                inp = np.clip(inp, 0, 1)

                ax.imshow(inp)

                if images_so_far == num_images:
                    break


def setup_finetune(model, phase, base_lr):
    """
    根据当前微调阶段，设置 requires_grad 和 optimizer 参数组。

    phase:
      0: 全网训练（普通训练，不区分阶段）
      1: 冻结 backbone，仅训练 fc（线性探测）
      2: 解冻 layer3 + layer4 + fc（主微调）
      3: 解冻 layer2 3 + layer4 + fc（更精细微调）

    返回:
      optimizer
    """
    # 兜底：不微调，直接全网训练
    if phase == 0:
        for n, p in model.named_parameters():
            p.requires_grad = True
        param_groups = [{"params": model.parameters(), "lr": base_lr}]
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
        return optimizer

    # 微调模式：先全部冻结
    for n, p in model.named_parameters():
        p.requires_grad = False

    param_groups = []

    if phase == 1:
        # 只训练 fc（线性探测）
        for n, p in model.named_parameters():
            if n.startswith("fc"):
                p.requires_grad = True

        param_groups.append(
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("fc")
                ],
                "lr": base_lr  # 例如 0.1
            }
        )

    elif phase == 2:
        # 主微调：解冻 layer3 + layer4 + fc
        for n, p in model.named_parameters():
            if n.startswith("fc") or n.startswith("layer4") or n.startswith("layer3"):
                p.requires_grad = True

        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("fc")],
                "lr": base_lr * 0.7,  # 0.07
            }
        )
        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("layer4")],
                "lr": base_lr * 0.3,  # 0.03
            }
        )
        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("layer3")],
                "lr": base_lr * 0.15,  # 0.015
            }
        )

    elif phase == 3:
        # 更精细微调：解冻 layer2 + layer3 + layer4 + fc
        for n, p in model.named_parameters():
            if (n.startswith("fc")
                    or n.startswith("layer4")
                    or n.startswith("layer3")
                    or n.startswith("layer2")):
                p.requires_grad = True

        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("fc")],
                "lr": base_lr * 0.4,  # 0.04
            }
        )
        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("layer4")],
                "lr": base_lr * 0.15,  # 0.015
            }
        )
        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("layer3")],
                "lr": base_lr * 0.08,  # 0.008
            }
        )
        param_groups.append(
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("layer2")],
                "lr": base_lr * 0.04,  # 0.004
            }
        )

    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    return optimizer


# ===================== 主逻辑 =====================
def main():
    # Get model name
    model_name = args.model

    # Create model-specific results directory
    model_results_dir = os.path.join(args.results_dir, model_name)
    model_plot_dir = os.path.join(model_results_dir, 'plots')

    # Create necessary directories
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(model_plot_dir, exist_ok=True)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CUDA optimization settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")

    # Hyperparameters
    epochs = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    # Data preprocessing（当前仅为 CIFAR 风格数据设置）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    cinic_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        # 为cinic10添加随机擦除增强
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    print("Loading datasets...")
    if args.dataset == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        if args.model in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202']:
            num_classes = 10
            classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            classes = val_dataset.classes
            num_classes = len(classes)
    elif args.dataset == 'Cinic10':
        cinic_root = './data/cinic'
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(cinic_root, 'train'),
            transform=cinic_transform_train
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(cinic_root, 'valid'),
            transform=transform_test
        )
        classes = val_dataset.classes
        num_classes = len(classes)
    else:
        raise ValueError(f"暂未实现数据集：{args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    val_num = len(val_dataset)
    train_num = len(train_dataset)
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    print(f"Training set size: {train_num}, Validation set size: {val_num}")

    # Select model
    model_dict = {
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152,
        'ResNet20': ResNet20,
        'ResNet32': ResNet32,
        'ResNet44': ResNet44,
        'ResNet56': ResNet56,
        'ResNet110': ResNet110,
        'ResNet1202': ResNet1202
    }

    print(f"Using {model_name} model for training...")
    model = model_dict[model_name]().to(device)

    # 加载预训练权重
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        ckpt = torch.load(args.pretrained_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained from {args.pretrained_path}")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    else:
        if args.pretrained_path is not None:
            print(f"预训练路径不存在: {args.pretrained_path}，跳过加载预训练")

    # 重建 fc（保证输出类别数正确）
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes).to(device)
        print(f"Reset final fc layer: in_features={in_features}, out_features={num_classes}")

    # 文件路径
    raw_model_filename = os.path.join(model_results_dir, f"{model_name}_best.pth")
    model_filename = get_unique_path(raw_model_filename)
    results_csv_path = os.path.join(model_results_dir, f"{model_name}_results.csv")
    results_txt_path = os.path.join(model_results_dir, f"{model_name}_results.txt")

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器 & 微调阶段初始化
    if args.freeze_backbone:
        current_phase = 1
        optimizer = setup_finetune(model, phase=current_phase, base_lr=LR)
        print(f"[Finetune] Start in phase {current_phase}: only fc trainable.")
    else:
        current_phase = 0
        optimizer = setup_finetune(model, phase=current_phase, base_lr=LR)
        print("[Train] Full model training (no staged finetuning).")

    # 学习率调度：CINIC-10 用 Cosine + 分阶段，其他保持接近原逻辑
    if args.dataset == 'Cinic10':

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"[Cinic10] Using CosineAnnealingLR(T_max={epochs})")
    else:
        if model_name in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[82, 123], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[82, 123], gamma=0.1)

    start_epoch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_history = []
    best_acc = 0.0
    best_loss = float('inf')
    best_epoch = 0
    best_train_acc, best_train_loss = 0.0, 0.0

    # 断点恢复
    if args.resume_path is not None and os.path.exists(args.resume_path):
        print(f"[Resume] Loading checkpoint from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[Resume] Optimizer state loaded.")

        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("[Resume] Scheduler state loaded.")

        start_epoch = ckpt.get("epoch", 0)

        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        train_accs = ckpt.get("train_accs", [])
        val_accs = ckpt.get("val_accs", [])
        lr_history = ckpt.get("lr_history", [])

        extra = ckpt.get("extra", {})
        best_acc = extra.get("best_acc", 0.0)
        best_loss = extra.get("best_loss", float('inf'))
        best_epoch = extra.get("best_epoch", 0)

        print(f"[Resume] Resume from epoch {start_epoch}, best_acc={best_acc:.4f}, best_loss={best_loss:.4f}")
    else:
        if args.resume_path is not None:
            print(f"[Resume] resume-path 不存在：{args.resume_path}，跳过恢复。")

    # CSV logging
    with open(results_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy',
                             'Val Accuracy', 'Learning Rate', 'Epoch Time(s)'])

    start_time = time.time()

    # ===================== 训练循环 =====================
    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            # 微调模式下：根据 epoch 切换阶段（只改 optimizer param_group，scheduler 保持 Cosine/Step）
            if args.freeze_backbone:
                if epoch < 2:
                    phase = 1
                elif epoch < 60:
                    phase = 2
                else:
                    phase = 3

                if phase != current_phase:
                    print(f"[Finetune] Switch to phase {phase} at epoch {epoch + 1}")
                    current_phase = phase
                    optimizer = setup_finetune(model, phase=current_phase, base_lr=LR)
                    if current_phase == 1:
                        model.apply(lambda m: isinstance(m, nn.BatchNorm2d) and m.eval())
                    if args.dataset == 'Cinic10':
                        # 切换阶段时把余弦退火也绑定一下
                        scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch)

            # train
            print(f"------- Epoch {epoch + 1} training start -------")
            model.train()
            train_acc = 0.0
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

                _, predict = torch.max(outputs, dim=1)
                train_acc += torch.eq(predict, labels).sum().item()

            train_loss = running_loss / train_steps
            train_accurate = train_acc / train_num

            if not args.valid:
                scheduler.step()
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']

                val_loss = 0.0
                val_accurate = 0.0

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_accurate)
                val_accs.append(val_accurate)
                lr_history.append(current_lr)

                epoch_time = time.time() - epoch_start
                print(f'Current learning rate: {current_lr}')
                print(f'[epoch {epoch + 1}] train_loss: {train_loss:.3f} '
                      f'train_accuracy:{train_accurate:.3f} (no validation) time: {epoch_time:.1f}s')

                with open(results_csv_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                         f"{train_accurate:.6f}", f"{val_accurate:.6f}",
                                         f"{current_lr:.6f}", f"{epoch_time:.1f}"])
                continue

            # valid=True 时的验证频率控制
            do_validation = (epoch <= 100) or (epoch > 100 and epoch % 2 == 1)

            if not do_validation:
                scheduler.step()
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']

                val_loss = 0.0
                val_accurate = 0.0

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_accurate)
                val_accs.append(val_accurate)
                lr_history.append(current_lr)

                epoch_time = time.time() - epoch_start
                print(f'Current learning rate: {current_lr}')
                print(f'[epoch {epoch + 1}] train_loss: {train_loss:.3f} '
                      f'train_accuracy:{train_accurate:.3f} (skip validation) '
                      f'time: {epoch_time:.1f}s')

                with open(results_csv_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                         f"{train_accurate:.6f}", f"{val_accurate:.6f}",
                                         f"{current_lr:.6f}", f"{epoch_time:.1f}"])
                continue

            # 验证
            model.eval()
            val_acc = 0.0
            val_running_loss = 0.0

            all_preds = []
            all_labels = []

            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for step, val_data in enumerate(val_bar):
                    val_images, val_labels = val_data
                    val_images, val_labels = val_images.to(device), val_labels.to(device)

                    outputs = model(val_images)
                    loss = loss_function(outputs, val_labels)

                    val_running_loss += loss.item()

                    _, predict = torch.max(outputs, dim=1)
                    val_acc += torch.eq(predict, val_labels).sum().item()

                    all_preds.extend(predict.cpu().numpy())
                    all_labels.extend(val_labels.cpu().numpy())

            val_loss = val_running_loss / val_steps
            val_accurate = val_acc / val_num

            scheduler.step()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_accurate)
            val_accs.append(val_accurate)
            lr_history.append(current_lr)

            epoch_time = time.time() - epoch_start

            print(f'Current learning rate: {current_lr}')
            print(f'[epoch {epoch + 1}] train_loss: {train_loss:.3f} val_loss:{val_loss:.3f} '
                  f'train_accuracy:{train_accurate:.3f} val_accuracy: {val_accurate:.3f} time: {epoch_time:.1f}s')

            with open(results_csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                     f"{train_accurate:.6f}", f"{val_accurate:.6f}",
                                     f"{current_lr:.6f}", f"{epoch_time:.1f}"])

            # 周期性保存 latest 检查点
            if (epoch + 1) % 10 == 0:
                latest_ckpt_path = os.path.join(model_results_dir, f"{model_name}_latest.pth")
                save_checkpoint(
                    latest_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    train_losses, val_losses, train_accs, val_accs, lr_history,
                    extra={
                        "best_acc": best_acc,
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                    },
                )

            if val_loss < best_loss:
                best_loss = val_loss
                print(f"Validation loss improved to {best_loss:.6f}")

            # ================= 仅在 acc 变好时保存 best & 可视化 =================
            if val_accurate > best_acc:
                best_acc = val_accurate
                best_epoch = epoch + 1
                best_train_acc = train_accurate
                best_train_loss = train_loss

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accurate,
                    "val_loss": val_loss,
                    "train_accuracy": train_accurate,
                    "train_loss": train_loss,
                    "epoch": epoch + 1
                }, model_filename)
                print(f"Found better model, saved to: {model_filename}")

                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=classes, yticklabels=classes)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'{model_name} Confusion Matrix (Epoch {epoch+1})')
                confusion_matrix_path = os.path.join(
                    model_plot_dir, f"confusion_matrix.png")
                plt.savefig(confusion_matrix_path)
                plt.close()
                print(f"Saved confusion matrix to: {confusion_matrix_path}")

                visualize_predictions_random(model, val_dataset, classes, device)
                pred_path = os.path.join(model_plot_dir, f"predictions.png")
                plt.savefig(pred_path)
                plt.close()
                print(f"Saved prediction visualization to: {pred_path}")

            # ================= 每轮都绘制 loss/acc 和 lr 曲线，保证完整 =================
            plt.figure(figsize=(11, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1),
                     train_losses, label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1),
                     val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} Loss Curves')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accs) + 1),
                     train_accs, label='Training Accuracy')
            plt.plot(range(1, len(val_accs) + 1),
                     val_accs, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'{model_name} Accuracy Curves')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            loss_acc_path = os.path.join(
                model_plot_dir, f"loss_acc_curves.png")
            plt.savefig(loss_acc_path)
            plt.close()
            print(f"Saved loss and accuracy curves to: {loss_acc_path}")

            plt.figure(figsize=(9, 5))
            plt.plot(range(1, len(lr_history) + 1), lr_history)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'{model_name} Learning Rate Curve')
            plt.grid(True)
            plt.yscale('log')
            lr_path = os.path.join(model_plot_dir, f"lr_curve.png")
            plt.savefig(lr_path)
            plt.close()
            print(f"Saved learning rate curve to: {lr_path}")

    except KeyboardInterrupt:
        interrupt_ckpt_path = os.path.join(
            model_results_dir, f"{model_name}_interrupt.pth"
        )
        print("收到 Ctrl+C，中断训练，正在保存中断检查点...")
        try:
            save_checkpoint(
                interrupt_ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_losses, val_losses, train_accs, val_accs, lr_history,
                extra={
                    "best_acc": best_acc,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                },
            )
        except Exception as e:
            print(f"保存中断检查点失败: {e}")
        finally:
            try:
                plt.close('all')
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("安全退出。")

    print(f"Training complete. Best model saved to: {model_filename}")
    print(
        f"Best validation accuracy: {best_acc:.4f}, Best validation loss: {best_loss:.6f} (Epoch {best_epoch})")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    with open(results_txt_path, 'w') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"训练参数:\n")
        f.write(f"  批量大小: {BATCH_SIZE}\n")
        f.write(f"  初始学习率: {LR}\n")
        f.write(f"  训练轮数: {epoch + 1}/{epochs}\n\n")

        f.write(f"训练结果:\n")
        f.write(f"  最佳验证准确率: {best_acc:.6f} (轮次 {best_epoch})\n")
        f.write(f"  最佳验证损失: {best_loss:.6f}\n")
        f.write(f"  最佳训练准确率: {best_train_acc:.6f}\n")
        f.write(f"  最佳训练损失: {best_train_loss:.6f}\n\n")

        f.write(f"训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n")
        f.write(
            f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n\n")

        f.write(f"文件位置:\n")
        f.write(f"  最佳模型: {model_filename}\n")
        f.write(f"  CSV结果: {results_csv_path}\n")
        f.write(f"  可视化结果: {model_plot_dir}\n")

    print(f"Saved summary results to: {results_txt_path}")

    try:
        if hasattr(train_loader, 'dataset'):
            train_loader = None
        if hasattr(val_loader, 'dataset'):
            val_loader = None
    except Exception as e:
        print(f"清理 DataLoader 时出错: {e}")

    try:
        plt.close('all')
    except Exception as e:
        print(f"关闭 matplotlib 图形时出错: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"All results saved to: {model_results_dir}")


if __name__ == "__main__":
    main()