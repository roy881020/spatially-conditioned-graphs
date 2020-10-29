import os
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket
from pocket.data import HICODet

from models import ModelWithGT, ModelWith1Mask, ModelWith2Masks, ModelWithNone, ModelWithVec
from utils import preprocessed_collate, PreprocessedDataset, test

MODELS = {
    'baseline': ModelWithNone,
    'GT': ModelWithGT,
    '2Masks': ModelWith2Masks,
    '1Mask': ModelWith1Mask,
    'handcraft': ModelWithVec,
}

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    hico_test = HICODet(None, '../Incubator/InteractRCNN/hicodet/instances_test2015.json')

    testset = PreprocessedDataset('./preprocessed/test2015')
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=preprocessed_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    net = MODELS[args.model_name]()

    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint["epoch"]

    net.cuda()
    timer = pocket.utils.HandyTimer(maxlen=1)
    
    with timer:
        test_ap = test(net, test_loader, hico_test)
    print("Epoch: {} | test mAP: {:.4f}, total time: {:.2f}s".format(
        epoch, test_ap.mean(), timer[0]
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--batch-size', default=2, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--human-thresh', default=0.5, type=float)
    parser.add_argument('--object-thresh', default=0.5, type=float)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    
    args = parser.parse_args()
    print(args)

    main(args)
