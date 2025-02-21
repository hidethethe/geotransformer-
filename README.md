# geotransformer-

# 数据加载
作者在这里使用了cfg设置训练参数和argparser设置文件路径的方法，值得学习

```
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=False, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict

```

```
# 学习示例
from easydict import EasyDict as edict

# 创建一个配置对象
config = edict()

# 设置一些配置项
config.seed = 42  # 随机种子

# 使用嵌套字典
config.model = edict()  # model 配置部分
config.model.type = "CNN"  # 模型类型

# 修改配置项
config.seed = 5
```

代码中还有一个鸡肋代码。由于 `make_cfg()` 只是返回 `_C`，并没有创建新的对象，因此 `cfg` 和 `_C` 都指向同一个 `EasyDict` 对象。
```
def make_cfg():
    return _C
    
    
cfg = make_cfg()
```
