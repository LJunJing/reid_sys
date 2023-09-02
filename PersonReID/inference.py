import os
import os.path as osp
import random
import shutil
import torch
import torchvision.transforms as transformers
from PIL import Image
from PersonReID.model.FFusion import FFusion, FFusion_cnn, FFusion_deit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    torch.cuda.set_device(1)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    global img
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def get_img(file_dir, num, tar_dir=None):
    path = os.listdir(file_dir)  # 获得图片的原始路径
    # file_num = len(pathDir) # 数据总量
    # img_num = int(file_num * rate)
    sample = random.sample(path, num)  # 随机选取img_num数量的样本图片
    # print(sample)
    if tar_dir:
        for name in sample:
            shutil.copy(file_dir+name, tar_dir+name)
    return sample

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def check_dir(file):
    if os.path.exists(file):
        sz = os.path.getsize(file)
        if sz:
            del_files(file)
    else:
        os.makedirs(file)

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat.detach().cpu().numpy()
    return dist_mat

def load(model_name, dataset_name, gallery_num):
    # -----------加载模型----------------
    model_factory = {
        "ResNet50": FFusion_cnn,
        "DeiT": FFusion_deit,
        "FFusion": FFusion
    }
    dataset_classes = {
        "Market1501": 751,
        "DukeMTMC-reID": 702
    }
    model = model_factory[model_name](num_classes=dataset_classes[dataset_name])
    model.load_param('PersonReID/logs/{}/{}_100.pth'.format(dataset_name, model_name)) #加载预训练好的参数
    model.to(device)
    model.eval()

    infer_transforms = transformers.Compose([
        transformers.Resize([256, 128]),
        transformers.ToTensor(),
        transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query_dir = 'PersonReID/data/{}/query/'.format(dataset_name)
    query_tardir = 'PersonReID/data/inference/query/'
    check_dir(query_tardir)  # 检查目录是否存在并清除上一次采样的数据

    gallery_dir = 'PersonReID/data/{}/bounding_box_test/'.format(dataset_name)
    gallery_tardir = 'PersonReID/data/inference/gallery/'
    check_dir(gallery_tardir)

    query_sample = get_img(query_dir, 1, query_tardir)  # query_sample存放采样的query图片的名称
    query_img = read_image(query_tardir+query_sample[0])
    query_trans = infer_transforms(query_img).unsqueeze(0).to(device)
    query_feat = model(query_trans)

    gallery_sample = get_img(gallery_dir, gallery_num, gallery_tardir)  # gallery_sample存放采样的gallery图片的名称
    gallery2dis = {}
    for sample in gallery_sample:
        gallery_img = read_image(gallery_tardir+sample)
        gallery_trans = infer_transforms(gallery_img).unsqueeze(0).to(device)
        gallery_feat = model(gallery_trans)
        distance = euclidean_distance(query_feat, gallery_feat)
        gallery2dis[sample] = '%.3f' % float(distance)
    return query_sample, gallery2dis
