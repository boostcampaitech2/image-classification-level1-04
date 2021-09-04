import os
import argparse
from re import sub
import torch
import pandas as pd
from tqdm import tqdm
import glob
import numpy as np

import data_loader.data_loaders as module_data
import data_loader.transforms as module_trsfm
import model.model as module_arch
from parse_config import ConfigParser
from torch.utils.data import DataLoader
from data_loader.datasets import MaskSubmitDataset

def main(config):
    submission_path = 'saved/submissions'
    test_dir = '/opt/ml/input/data/eval'

    if config['data_loader']['args']['is_main'] == True:
        logger = config.get_logger('test')

        # setup transforms instances
        trsfm, default_trsfm = config.init_ftn('transforms_select', module_trsfm)()

        # setup data_loader instances
        data_loader = config.init_obj('data_loader', module_data,
                                    trsfm=trsfm, default_trsfm=default_trsfm)
        _, _, submit_data_loader = data_loader.split_validation()
        print(submit_data_loader)

        # build model architecture
        model = config.init_obj('arch', module_arch)
        logger.info(model)
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        
        filst = sorted(glob.glob(f"saved/models/{config['name']}/*/*.pth"), key=os.path.getctime)
        checkpoint = torch.load(filst[-1])
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        all_predictions = []
        with torch.no_grad():
            for data in tqdm(submit_data_loader):
                data = data.to(device)
                pred = model(data)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
        
        df_submission = submit_data_loader.dataset.df
        df_submission['ans'] = all_predictions
        save_name = '_'.join(str(config.resume).split(os.sep)[2:])
        save_path = os.path.join(submission_path, f'submission_{save_name}.csv')
        df_submission.to_csv(save_path, index=False)
        print(f'test inference is saved at {save_path}!')
        print('test inference is done!')

    elif config['data_loader']['args']['is_main'] == False:
        logger = config.get_logger('test')

        trsfm, default_trsfm = config.init_ftn('transforms_select', module_trsfm)()

        filst = sorted(glob.glob(f"{submission_path}/*.csv"), key=os.path.getctime)

        submission = pd.read_csv(filst[-1])
        error_set = set([1, 2, 4, 5, 7, 8, 10, 11, 16, 17])
        df_tocheck = submission[submission['ans'].apply(lambda x : x in error_set)]

        image_dir = os.path.join(test_dir, 'images_face_crop')
        check_image_paths = [os.path.join(image_dir, img_id) for img_id in df_tocheck.ImageID]
        dataset = MaskSubmitDataset(crop=True, transform=default_trsfm, image_glob=check_image_paths)
        submit_data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        # build model architecture
        model = config.init_obj('arch', module_arch)
        logger.info(model)
        logger.info('Loading checkpoint: {} ...'.format(config.resume))

        filst = sorted(glob.glob(f"saved/models/{config['name']}/*/*.pth"), key=os.path.getctime)
        checkpoint = torch.load(filst[-1])
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        tocheck_predictions = []
        with torch.no_grad():
            for data in tqdm(submit_data_loader):
                data = data.to(device)
                pred = model(data)
                pred = pred.argmax(dim=-1)
                tocheck_predictions.extend(pred.cpu().numpy())
        df_tocheck['ans2'] = tocheck_predictions

        trans_labels = []
        for i in zip(df_tocheck.ans, df_tocheck.ans2):
            trans_labels.append(trans_label(*i))
        df_tocheck['ans_trans'] = np.array(trans_labels).reshape(-1, 1)
        submission_age = df_tocheck.drop(['ans', 'ans2'], axis=1)

        df = pd.merge(submission, submission_age, how='outer', left_on='ImageID', right_on='ImageID')
        df.ans_trans[df.ans_trans.isna()] = df.ans[df.ans_trans.isna()]
        df.ans_trans = df.ans_trans.astype(int)
        df = df.drop('ans', axis=1)
        submission = df.rename({"ans_trans":"ans"}, axis=1)

        # 제출할 파일을 저장합니다.
        save_name = '_'.join(str(config.resume).split(os.sep)[2:])
        save_path = os.path.join(test_dir, 'submission.csv')
        submission.to_csv(save_path, index=False)
        print(f'test inference is saved at {save_path}!')
        print('test inference is done!') 

# 레이블 변환 함수
def trans_label(ans, ans2):
    # ans 원래 Label 18가지
    # ans 마스크 착용 여부만
    # ans2에서는 성별과 나이

    label = 0
    # Mask착용 여부
    if ans >= 6 and ans <= 11:
        label += 6
    elif ans >= 12:
        label += 12

    # 성별 여부
    if (ans // 3) % 2 == 1:
        label += 3

    # ans2에서 성별 제외
    if ans2 >= 4:
        ans2 -= 4
    # 나이에 대한 보정
    if ans2 <= 2:
        label += 1
    elif ans2 == 3:
        label += 2
    return label
        
        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)