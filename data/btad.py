import os
import json


class BTADSolver(object):
    CLSNAMES = [
        '01', '02', '03',
    ]

    def __init__(self, root='data/btad'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}') # species is 'ok' or 'ko'
                for specie in species:
                    is_abnormal = True if specie not in ['ok'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}') # btad/01/train/ok or btad/01/test/ko
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = BTADSolver(root='data/btad')
    runner.run()
    # nums = 0
    # with open('./data/btad/meta.json') as f:
    #     data = json.load(f)
    #     for phase in data.keys():
    #         for cls_name in data[phase].keys():
    #             nums += len(data[phase][cls_name])
    # print("len(train) + len(test) = ", nums) # 2830

