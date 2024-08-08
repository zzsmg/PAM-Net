Change the path to the correct dataset！

#### Train

 ```shell
 python main.py --net='Pam_ffa_1 or Pam_ffa_2' --loss='Total or L1' --crop --crop_size=240 --bs=2 --steps=500000 --eval_step=5000 --perloss
 ```

#### Test

*Put  models in the `PAM-FFA-Net/trained_models`folder.*

*Put your hazy images and prior in `PAM-FFA-Net/test/hazy(prior)`*

 ```shell
 python test.py
```