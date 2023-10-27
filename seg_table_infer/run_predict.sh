# clear

# if [ "$1" = "gpa" ] 
# then
#     qos=normal
#     part=gpu-a100
#     constraint=""
# else
#     if [ "$1" = "feit" ] 
#     then
#         qos=feit
#         part=feit-gpu-a100
#         constraint=""

#     else
#         qos=gpgpudeeplearn
#         part=deeplearn
#         if [ "$1" = "dl" ] 
#         then
#             constraint=""
#         else
#             constraint="--constraint=$1"
#         fi
#     fi
# fi


# if [ "$5" = "test" ] 
# then
#     test="-t"
# else
#     test=""
# fi


# declare -u expname
# expname=`basename $4 .yaml`
# expname=${expname: -8}


# g=$(($2<4?$2:4))

# srun -p $part   \
#     --gres=gpu:$g --ntasks-per-node=1 --ntasks=1\
#     --cpus-per-task=$3 \
#     --exclude=spartan-gpgpu086 \
#     --job-name=${expname} \
#     --qos=$qos \
#     --time=3-00:00:00 \
#     --mem-per-cpu=4G $constraint \
#     python -u segment/predict.py \
#     --weights last.pt \
#     --source test_images --save-txt


python -u segment/predict.py \
--weights last.pt \
--source test_images --save-txt