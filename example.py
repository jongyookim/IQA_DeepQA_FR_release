import sys
import theano.sandbox.cuda
theano.sandbox.cuda.use(sys.argv[1] if len(sys.argv) > 1 else 'cuda0')

from IQA_DeepQA_FR_release import train_iqa as tm

tm.train_iqa(
    config_file='IQA_DeepQA_FR_release/configs/FR_sens_1.yaml',
    section='fr_sens_LIVE',
    tr_te_file='outputs/tr_va_live.txt',
    snap_path='outputs/FR/FR_sens_LIVE_1/',
)
