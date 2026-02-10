sudo ip addr add 172.16.0.100/24 dev enx207bd51a3217
sudo ip link set enx207bd51a3217 up
ping 172.16.0.2


python -m real_deployment.model_controller_del_ee


## install frankx
conda install frankx
unzip frankx.zip