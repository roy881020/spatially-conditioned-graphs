import os
import torch
import argparse
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

import pocket
from pocket.data import HICODet
from pocket.utils import DetectionAPMeter

from models import *
from utils import preprocessed_collate, PreprocessedDataset, test

MODELS = {
    'baseline': ModelWithNone,
    'gt': ModelWithGT,
    'gt1': ModelWithGT1,
    'gt_': ModelWithOnlyGT,
    'gt_1': ModelWithOnlyGT1,
    '2mask': ModelWith2Masks,
    '1mask': ModelWith1Mask,
    'handcraft': ModelWithVec,
}

@torch.no_grad()
def test_c(net, test_loader):
    net.eval()
    ap_test = DetectionAPMeter(117, algorithm='11P')
    for batch in tqdm(test_loader):
        batch_cuda = pocket.ops.relocate_to_cuda(batch)
        output = net(batch_cuda)
        if output is None:
            continue
        for result in output:
            ap_test.append(
                torch.cat(result["scores"]),
                torch.cat(result["labels"]),
                torch.cat(result["gt_labels"])
            )
    return ap_test.eval()

def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    hico_test = HICODet(None, '../Incubator/InteractRCNN/hicodet/instances_test2015.json')
    hico_train = HICODet(None, '../Incubator/InteractRCNN/hicodet/instances_train2015.json')

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
        #test_ap = test(net, test_loader, hico_test)
        test_ap = test_c(net, test_loader)
    torch.save(test_ap, 'map.pt')
    for name, n, ap in zip(hico_train.verbs, hico_train.anno_action, test_ap):
        print(name, n, ap.item())
    print("\nEpoch: {} | test mAP: {:.4f}, total time: {:.2f}s".format(
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
