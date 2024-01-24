import os
import json


class DAGMSolver(object):
    CLSNAMES = [
        'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 
        'Class6', 'Class7', 'Class8', 'Class9', 'Class10',
    ]

    def __init__(self, root='data/dagm'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(Train={}, Test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}/{cls_name}'
            for phase in ['Train', 'Test']:
                cls_info = []
                label_files_path = os.path.join(cls_dir, phase, 'Label/Labels.txt')
                with open(label_files_path, 'r') as file:
                    lines = file.readlines()
                for line in lines[1:]:
                    fields = line.strip().split('\t')
                    is_abnormal = True if fields[1] == '1' else False
                    img_path = os.path.join(cls_dir, phase, fields[2])
                    mask_path =os.path.join(cls_dir, phase, 'Label', fields[-1]) if is_abnormal else ''

                    info_img = dict(
                            img_path=img_path,
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name='defect' if is_abnormal else 'good',
                            anomaly=1 if is_abnormal else 0,
                        )
                    cls_info.append(info_img)
                info[phase][cls_name] = cls_info
                
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
            
if __name__ == '__main__':
    runner = DAGMSolver(root='data/dagm')
    runner.run()
    # nums = 0
    # with open('./data/dagm/meta.json') as f:
    #     data = json.load(f)
    #     for phase in data.keys():
    #         for cls_name in data[phase].keys():
    #             nums += len(data[phase][cls_name])
    # print("len(train) + len(test) = ", nums)
    