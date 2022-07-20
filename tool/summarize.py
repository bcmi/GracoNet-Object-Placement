import argparse
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--expid", type=str, required=True, help="experiment name")
parser.add_argument("--eval_type", type=str, default="eval", help="evaluation type")
args = parser.parse_args()

if args.eval_type == "eval":
    sumup_list = ['acc', 'fid']
elif args.eval_type == "evaluni":
    sumup_list = ['lpips_variety']
else:
    raise NotImplementedError

eval_dir = os.path.join('result', args.expid, args.eval_type)
assert (os.path.exists(eval_dir))

dt_ms = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
div_title = '=' * 100
div_line = '-' * 43
div_record = '-' * 95

eval_resall_file = os.path.join('result', args.expid, '{}_resall.txt'.format(args.eval_type))
with open(eval_resall_file, 'w') as _out:
    _out.write("{}\nStatistic Time: {}\n".format(div_title, dt_ms))
    eps = sorted(list(map(int, os.listdir(eval_dir))))
    for ep in eps:
        _out.write("{} Epoch {} {}\n".format(div_line, ep, div_line))
        for sumup_item in sumup_list:
            res_file = os.path.join(eval_dir, str(ep), '{}_{}.txt'.format(args.eval_type, sumup_item))
            if os.path.exists(res_file):
                with open(res_file, 'r') as f:
                    res_ep = f.read()
                _out.write("{}\n".format(res_ep.rstrip('\n')))
            else:
                _out.write("{}\n".format("Skipping {} ...".format(sumup_item)))
        _out.write("{}\n".format(div_record))
    _out.write("{}\n\n\n".format(div_title))
