CUDA_VISIBLE_DEVICES=1 python ./UAMFD_a_test.py -train_dataset weibo \
                                        -test_dataset weibo \
                                        -batch_size 8 \
                                        -epochs 50 \
                                        -val 0 \
                                        -is_sample_positive 1.0 \
                                        -duplicate_fake_times 0 \
                                        -network_arch UAMFDv2 \
                                        -is_filter 0 \
                                        -not_on_12 1
# -checkpoint /groupshare/CIKM_ying_output//gossip/19_814_89.pkl
#CUDA_VISIBLE_DEVICES=0

# nohup python -u test.py > test.log 2>&1 &
# https://blog.csdn.net/m0_38024592/article/details/103336210  相关介绍

#nohup python -u ./UAMFD.py -train_dataset weibo -test_dataset weibo -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > Weibo21.log 2>&1 &
#[1] 2166247


#origin
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD.py -train_dataset weibo -test_dataset weibo -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > Weibo21_50.log 2>&1 &


#dis
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_a_test.py -train_dataset weibo -test_dataset weibo -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > Weibo21_dis_50.log 2>&1 &


#xai
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_b_test.py -train_dataset weibo -test_dataset weibo -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > Weibo21_dis_v2.1.7_50.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_c_test.py -train_dataset weibo -test_dataset weibo -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > Weibo21_dis_v2.2.1_50.log 2>&1 &







