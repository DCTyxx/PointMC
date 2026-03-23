import __init__

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).absolute().parent.parent))


def prepare(i, o):
    print(f"Processed data will be saved in:\n{o}")

    o.mkdir(exist_ok=True)

    labels_dict = {
        "ceiling": 0,
        "floor": 1,
        "wall": 2,
        "beam": 3,
        "column": 4,
        "window": 5,
        "door": 6,
        "table": 7,
        "chair": 8,
        "sofa": 9,
        "bookcase": 10,
        "board": 11,
        "clutter": 12,
        "stairs": 12
    }

    for area_number in range(1, 7):
        print(f'Reencoding point clouds of area {area_number:d}')
        dir = i / f'Area_{area_number:d}'
        if not dir.exists():
            warnings.warn(f'Area {area_number:d} not found')
            continue
        for pc_path in sorted(list(dir.iterdir())):
            if not pc_path.is_dir:
                continue
            pc_name = f'{area_number:d}_' + pc_path.stem + '.pt'
            pc_file = o / pc_name

            if pc_file.exists():
                continue

            points_xyz = []
            points_col = []
            points_lbl = []
            for elem in sorted(list(pc_path.glob('Annotations/*.txt'))):
                label = elem.stem.split('_')[0]
                try:
                    points = torch.from_numpy(np.loadtxt(elem, dtype=np.float32))
                except Exception as e:
                    print(elem)
                    raise e
                label_id = labels_dict[label]
                points_xyz.append(points[:, :3])
                points_col.append(points[:, 3:])
                points_lbl.append(torch.full((points.shape[0],), label_id, dtype=torch.uint8))

            if points_xyz == []:
                continue

            points_xyz = torch.cat(points_xyz, dim=0)
            points_xyz = points_xyz - points_xyz.min(dim=0)[0]
            points_col = torch.cat(points_col, dim=0).type(torch.uint8)
            points_lbl = torch.cat(points_lbl, dim=0)

            torch.save((points_xyz, points_col, points_lbl), pc_file)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=False, default='s3dis')

    args, opts = parser.parse_known_args()

    i = Path(args.i)
    o = i.parent / args.o

    prepare(i, o)

