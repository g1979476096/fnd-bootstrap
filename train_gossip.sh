CUDA_VISIBLE_DEVICES=1 python ./UAMFD_b_test.py -train_dataset gossip \
                                        -test_dataset gossip \
                                        -batch_size 8 \
                                        -epochs 50 \
                                        -val 0 \
                                        -is_sample_positive 1.0 \
                                        -duplicate_fake_times 0 \
                                        -network_arch UAMFDv2 \
                                        -is_filter 0 \
                                        -not_on_12 1
# -checkpoint /groupshare/CIKM_ying_output//gossip/19_814_89.pkl \

#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip.log 2>&1 &
#2232037


#test
#no vgg
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_a_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 30 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_no_vgg.log 2>&1 &
#2347817

#解缠
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_a_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v3_50.log 2>&1 &
#30122

#xai
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_b_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 100 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v4.2.25_100.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_c_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 100 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v4.3.6_100.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_d_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 100 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v4.3.4_50.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_e_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 100 -val 1 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v4.3.4_50.log 2>&1 &
#python ./UAMFD_e_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 100 -val 1 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1


#0416 解缠 消融
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_a_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v5.1_50.log 2>&1 &

#0418
#CUDA_VISIBLE_DEVICES=0 nohup python -u ./UAMFD_a_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v5.2_50.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_f_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v5.5_50.log 2>&1 &



#0419
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_f_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v6.1_50.log 2>&1 &

#0429 加权BCE thresh设为 0.5
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_f_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v6.2_50.log 2>&1 &


#0513 不同模式
#CUDA_VISIBLE_DEVICES=1 nohup python -u ./UAMFD_g_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v7.2_50.log 2>&1 &


#1124
#CUDA_VISIBLE_DEVICES=0 nohup python -u ./UAMFD_g_test.py -train_dataset gossip -test_dataset gossip -batch_size 8  -epochs 50 -val 0 -is_sample_positive 1.0 -duplicate_fake_times 0 -network_arch UAMFDv2 -is_filter 0 -not_on_12 1 > gossip_dis_v9.6_50.log 2>&1 &


