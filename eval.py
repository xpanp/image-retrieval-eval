import argparse
import zoo
import dataset

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='pattern',
        help='ukbench/holidays/oxford5k/paris6k/pattern')

    parser.add_argument('--datadir', type=str, default='image-pattern',
        help='dataset dir')

    parser.add_argument('--datapth', type=str, default='pattern.pth',
        help='dataset feature pth')

    parser.add_argument('--method', type=str, default='mae',
        help='vit/mae/swin/vgg/resnet')

    parser.add_argument('--model', type=str, default='mae_vit_base_patch16',
        help='model name, such as mae_vit_base_patch16/vit_base_patch16_224')

    parser.add_argument('--ckp', type=str, default='mae_finetuned_vit_base.pth',
        help='checkpoint')
    
    parser.add_argument('--no_extract', action='store_true', default=False,
        help='no extract feature')

    parser.add_argument('--no_eval', action='store_true', default=False,
        help='no evaluate method')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    
    d = dataset.create_dataset(args.dataset, args.datadir, args.datapth)
    if not args.no_extract:
        m = zoo.create_model(args.method, args.model, args.ckp)
        m.extract(d)
    
    if not args.no_eval:
        d.evaluate()

if __name__ == '__main__':
    main()