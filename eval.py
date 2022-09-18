import argparse
import zoo
import dataset

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='holidays',
        help='ukbench/holidays/oxford5k/paris6k')

    parser.add_argument('--datadir', type=str, default='C:\\Users\\phs\\Desktop\\pytest\\dataset\\ukbench-100',
        help='dataset dir')

    parser.add_argument('--datapth', type=str, default='holidays.pth',
        help='dataset feature pth')

    parser.add_argument('--method', type=str, default='vit',
        help='vit/mae/swin/vgg/resnet')

    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
        help='model name, such as mae_vit_base_patch16/vit_base_patch16_224')

    parser.add_argument('--ckp', type=str, default='mae_finetuned_vit_base.pth',
        help='checkpoint')
    
    parser.add_argument('--extract', action='store_true', default=True,
        help='extract feature')

    parser.add_argument('--eval', action='store_true', default=True,
        help='evaluate method')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    
    d = dataset.create_dataset(args.dataset, args.datadir, args.datapth)
    if args.extract:
        m = zoo.create_model(args.method, args.model, args.ckp)
        m.extract(d)
    
    if args.eval:
        d.evaluate()

if __name__ == '__main__':
    main()