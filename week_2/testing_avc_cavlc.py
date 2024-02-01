from video_codecs.utils_avc import enc_cavlc, dec_cavlc
import numpy as np

testing_seq = np.array([-13, -10,  -8,   3,  11,   4,   1,  -2,  -2,   1,  -1,  -2,  -1,   2,   2,  -2]
)
testing_seq = np.reshape(testing_seq, (4, 4), order='F')
output = enc_cavlc(testing_seq, 8, 8)
rec = dec_cavlc(output, 8, 8)
breakpoint()