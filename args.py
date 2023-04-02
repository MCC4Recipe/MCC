import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='tri-joint parameters')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default=[0], type=list)

    # data
    parser.add_argument('--img_path', default='../im2recipe/data/')
    parser.add_argument('--data_path', default='../im2recipe/data/food_data/')
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--vocab_path', type=str, default='../im2recipe/data/new_word_dict.pkl',
                        help='path for vocabulary wrapper')

    # model
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--model_name', default='model', type=str)

    # im2recipe model
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--nRNNs', default=1, type=int)
    parser.add_argument('--srnnDim', default=1024, type=int)
    parser.add_argument('--irnnDim', default=300, type=int)
    parser.add_argument('--imfeatDim', default=2048, type=int)
    parser.add_argument('--stDim', default=1024, type=int)
    parser.add_argument('--ingrW2VDim', default=300, type=int)
    parser.add_argument('--maxSeqlen', default=20, type=int)
    parser.add_argument('--maxIngrs', default=20, type=int)
    parser.add_argument('--maxImgs', default=5, type=int)
    parser.add_argument('--numClasses', default=1048, type=int)

    # training
    parser.add_argument('--resume_recipe', default='', type=str)
    parser.add_argument('--resume_image', default='', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ingrW2V', default='data/vocab.bin', type=str)

    # SCAN
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax', default=9., type=float, help='Attention softmax temperature.')
    parser.add_argument('--lambda_lse', default=6., type=float, help='LogSumExp temp.')
    parser.add_argument('--agg_func', default="LogSumExp", help='LogSumExp|Mean|Max|Sum')

    # NAAF
    parser.add_argument('--thres', default=0, type=float, help='Optimal learning  boundary.')
    parser.add_argument('--alpha', default=2.0, type=float, help='Initial penalty parameter.')
    parser.add_argument('--thres_safe', default=0, type=float, help='Optimal learning  boundary.')
    parser.add_argument('--mean_neg', default=0, type=float, help='Mean value of mismatched distribution.')
    parser.add_argument('--stnd_neg', default=0, type=float, help='Standard deviation of mismatched distribution.')
    parser.add_argument('--mean_pos', default=0, type=float, help='Mean value of matched distribution.')
    parser.add_argument('--stnd_pos', default=0, type=float, help='Standard deviation of matched distribution.')

    # MedR / Recall@1 / Recall@5 / Recall@10
    parser.add_argument('--embtype', default='image', type=str)  # [image|recipe] query type
    parser.add_argument('--medR', default=1000, type=int)

    # dataset
    parser.add_argument('--maxlen', default=20, type=int)
    parser.add_argument('--vocab', default='vocab.txt', type=str)
    parser.add_argument('--dataset', default='../data/recipe1M/', type=str)
    parser.add_argument('--sthdir', default='../data/', type=str)

    return parser
