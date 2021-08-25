import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=1,
        submit=True
    )

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
        for data in tqdm(data_loader):
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    
    df_submission = data_loader.dataset.df
    df_submission['ans'] = all_predictions
    save_name = '_'.join(str(config.resume).split(os.sep)[2:])
    save_path = os.path.join(data_loader.dataset.test_dir_path, f'submission_{save_name}.csv')
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

    config = ConfigParser.from_args(args)
    main(config)