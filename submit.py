import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import data_loader.data_loaders as module_data
import data_loader.transforms as module_trsfm
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    submission_path = 'saved/models/submissions'
    test_dir = '/opt/ml/input/data/eval'

    if config['m'] == 'main':

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
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
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

    else:
        logger = config.get_logger('test')
        filst = sorted(os.path.join(submission_path, '*', ), key=os.path.getctime)
        df_submission = os.path.join(submission_path, filst[-1])
     
        image_dir = os.path.join(test_dir, 'images_facecrop')

        image_paths = [os.path.join(image_dir, img_id) for img_id in df_submission.ImageID]
        dataset = TestDataset(image_paths, A_transform['VIT_test'])
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)


        # build model architecture
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
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
        save_path = os.path.join(data_loader.eval_dir, f'submission_{save_name}.csv')
        df_submission.to_csv(save_path, index=False)
        print(f'test inference is saved at {save_path}!')
        print('test inference is done!')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-m', '--model', default='main', type=bool,
                      help='main vs sub (default: main)')

    config = ConfigParser.from_args(args)
    main(config)