import argparse
import datetime
from tqdm import tqdm
import os
import numpy as np
import torch
import lpips

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir')
    parser.add_argument('-v','--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
    parser.add_argument("--eval_type", type=str, default="evaluni", help="evaluation type")
    parser.add_argument("--repeat", type=int, default=10, help="repeat count for sampling z")
    opt = parser.parse_args()

    assert (opt.repeat > 1)
    data_dir = os.path.join('result', opt.expid, opt.eval_type, str(opt.epoch))
    assert (os.path.exists(data_dir))

    # initialize the model
    loss_fn = lpips.LPIPS(net='alex', version=opt.version)
    if (opt.use_gpu):
        loss_fn.cuda()

    # crawl directory
    files_list = list(sorted(os.listdir(opt.dir)))
    files_dict = {}
    for filename in files_list:
        index = filename.split('_')[0]
        if index in files_dict:
            files_dict[index].append(filename)
        else:
            files_dict[index] = [filename]
    total = len(files_dict)

    # stores distances
    dist_all = {}
    for i, index in enumerate(tqdm(files_dict, total=total)):
        dist_all[index] = []
        files = files_dict[index]
        for ff, file0 in enumerate(files[:-1]):
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir, file0))) # RGB image from [-1,1]
            if (opt.use_gpu):
                img0 = img0.cuda()
            for file1 in files[ff+1:]:
                img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir, file1)))
                if (opt.use_gpu):
                    img1 = img1.cuda()
                # compute distance
                with torch.no_grad():
                    dist01 = loss_fn.forward(img0, img1).squeeze().cpu().item()
                dist_all[index].append(dist01)

    # calculate results
    dist_res = np.zeros((total, 2), dtype=np.float32)
    for i, index in enumerate(dist_all):
        dists = dist_all[index]
        dist_res[i,0] = np.mean(np.array(dists))  # avg of dists for index
        dist_res[i,1] = np.std(np.array(dists))/np.sqrt(len(dists))  # stderr of dists for index

    dist_avg = np.mean(dist_res[:,0])
    dist_stderr = np.mean(dist_res[:,1])
    print(" - LPIPS (Variety): dist = {:.3f}, stderr = {:.6f}".format(dist_avg, dist_stderr))
    mark = 'a' if os.path.exists(os.path.join(data_dir, "{}_lpips_variety.txt".format(opt.eval_type))) else 'w'
    with open(os.path.join(data_dir, "{}_lpips_variety.txt".format(opt.eval_type)), mark) as f:
        f.write("{}\n".format(datetime.datetime.now()))
        f.write(" - LPIPS (Variety): dist = {:.3f}, stderr = {:.6f}\n".format(dist_avg, dist_stderr))


if __name__ == '__main__':
    main()
