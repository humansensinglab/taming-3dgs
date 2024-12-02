# Training for Ours (Big): Same number of gaussians as 3DGS
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle_big python train.py -s /data/user_storage/saswat/tandt/mip360/bicycle -i images_4 --eval --densification_interval 100 --mode "final_count" --budget  5987095 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=flowers_big python train.py -s /data/user_storage/saswat/tandt/mip360/flowers -i images_4 --eval --densification_interval 100 --mode "final_count" --budget  3618411 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=2 OAR_JOB_ID=garden_big python train.py -s /data/user_storage/saswat/tandt/mip360/garden -i images_4 --eval --densification_interval 100 --mode "final_count" --budget  5728191 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=3 OAR_JOB_ID=stump_big python train.py -s /data/user_storage/saswat/tandt/mip360/stump -i images_4 --eval --densification_interval 100 --mode "final_count" --budget  4867429 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=4 OAR_JOB_ID=treehill_big python train.py -s /data/user_storage/saswat/tandt/mip360/treehill -i images_4 --eval --densification_interval 100 --mode "final_count" --budget  3770257 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=5 OAR_JOB_ID=room_big python train.py -s /data/user_storage/saswat/tandt/mip360/room -i images_2 --eval --densification_interval 100 --mode "final_count" --budget 1548960 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=6 OAR_JOB_ID=counter_big python train.py -s /data/user_storage/saswat/tandt/mip360/counter -i images_2 --eval --densification_interval 100 --mode "final_count" --budget 1190919 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=7 OAR_JOB_ID=kitchen_big python train.py -s /data/user_storage/saswat/tandt/mip360/kitchen -i images_2 --eval --densification_interval 100 --mode "final_count" --budget 1803735 --optimizer_type default --test_iterations 30000 --sh_lower &
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom_big python train.py -s /data/user_storage/saswat/tandt/db/playroom --eval --densification_interval 100 --budget  2326100 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=bonsai_big python train.py -s /data/user_storage/saswat/tandt/mip360/bonsai -i images_2 --eval --densification_interval 100 --budget 1252367 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=2 OAR_JOB_ID=truck_big python train.py -s /data/user_storage/saswat/tandt/tandt/truck --eval --densification_interval 100 --budget  2584171 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=3 OAR_JOB_ID=train_big python train.py -s /data/user_storage/saswat/tandt/tandt/train --eval --densification_interval 100 --budget 1085480 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=4 OAR_JOB_ID=drjohnson_big python train.py -s /data/user_storage/saswat/tandt/db/drjohnson --eval --densification_interval 100 --budget  3273600 --optimizer_type default --test_iterations 30000 --sh_lower &
wait;

# Training for Ours: Fewer number of gaussians as set by the budget
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle_budget python train.py -s /data/user_storage/saswat/tandt/mip360/bicycle -i images_4 --eval --densification_interval 500 --mode "multiplier" --budget 15 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=flowers_budget python train.py -s /data/user_storage/saswat/tandt/mip360/flowers -i images_4 --eval --densification_interval 500 --mode "multiplier" --budget 15 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=2 OAR_JOB_ID=garden_budget python train.py -s /data/user_storage/saswat/tandt/mip360/garden -i images_4 --eval --densification_interval 500 --mode "multiplier" --budget 15 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=3 OAR_JOB_ID=stump_budget python train.py -s /data/user_storage/saswat/tandt/mip360/stump -i images_4 --eval --densification_interval 500 --mode "multiplier" --budget 15 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=4 OAR_JOB_ID=treehill_budget python train.py -s /data/user_storage/saswat/tandt/mip360/treehill -i images_4 --eval --densification_interval 500 --mode "multiplier" --budget 15 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=5 OAR_JOB_ID=room_budget python train.py -s /data/user_storage/saswat/tandt/mip360/room -i images_2 --eval --densification_interval 500 --mode "multiplier" --budget 2 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=6 OAR_JOB_ID=counter_budget python train.py -s /data/user_storage/saswat/tandt/mip360/counter -i images_2 --eval --densification_interval 500 --mode "multiplier" --budget 2 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=7 OAR_JOB_ID=kitchen_budget python train.py -s /data/user_storage/saswat/tandt/mip360/kitchen -i images_2 --eval --densification_interval 500 --mode "multiplier" --budget 2 --optimizer_type default --test_iterations 30000 --sh_lower &
wait;
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom_budget python train.py -s /data/user_storage/saswat/tandt/db/playroom --eval --densification_interval 500 --mode "multiplier" --budget 5 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=bonsai_budget python train.py -s /data/user_storage/saswat/tandt/mip360/bonsai -i images_2 --eval --densification_interval 500 --mode "multiplier" --budget 2 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=truck_budget python train.py -s /data/user_storage/saswat/tandt/tanksandtemples/truck --eval --densification_interval 500 --mode "multiplier" --budget 2 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=train_budget python train.py -s /data/user_storage/saswat/tandt/tanksandtemples/train --eval --densification_interval 500 --mode "multiplier" --budget 2 --optimizer_type default --test_iterations 30000 --sh_lower &
CUDA_VISIBLE_DEVICES=1 OAR_JOB_ID=drjohnson_budget python train.py -s /data/user_storage/saswat/tandt/db/drjohnson --eval --densification_interval 500 --mode "multiplier" --budget 5 --optimizer_type default --test_iterations 30000 --sh_lower &
wait;

# Rendering
CUDA_VISIBLE_DEVICES=0 python render.py -m ./output/bicycle_big &
CUDA_VISIBLE_DEVICES=1 python render.py -m ./output/flowers_big &
CUDA_VISIBLE_DEVICES=2 python render.py -m ./output/garden_big &
CUDA_VISIBLE_DEVICES=3 python render.py -m ./output/stump_big &
CUDA_VISIBLE_DEVICES=4 python render.py -m ./output/treehill_big &
CUDA_VISIBLE_DEVICES=5 python render.py -m ./output/room_big &
CUDA_VISIBLE_DEVICES=6 python render.py -m ./output/counter_big &
CUDA_VISIBLE_DEVICES=7 python render.py -m ./output/kitchen_big &
wait;
CUDA_VISIBLE_DEVICES=0 python render.py -m ./output/playroom_big &
CUDA_VISIBLE_DEVICES=1 python render.py -m ./output/bonsai_big &
CUDA_VISIBLE_DEVICES=2 python render.py -m ./output/truck_big &
CUDA_VISIBLE_DEVICES=3 python render.py -m ./output/train_big &
CUDA_VISIBLE_DEVICES=4 python render.py -m ./output/drjohnson_big &
wait;

CUDA_VISIBLE_DEVICES=0 python render.py -m output/bicycle_budget &
CUDA_VISIBLE_DEVICES=1 python render.py -m output/flowers_budget &
CUDA_VISIBLE_DEVICES=2 python render.py -m output/garden_budget &
CUDA_VISIBLE_DEVICES=3 python render.py -m output/stump_budget &
CUDA_VISIBLE_DEVICES=4 python render.py -m output/treehill_budget &
CUDA_VISIBLE_DEVICES=5 python render.py -m output/room_budget &
CUDA_VISIBLE_DEVICES=6 python render.py -m output/counter_budget &
CUDA_VISIBLE_DEVICES=7 python render.py -m output/kitchen_budget &
wait;
CUDA_VISIBLE_DEVICES=0 python render.py -m output/bonsai_budget &
CUDA_VISIBLE_DEVICES=1 python render.py -m output/truck_budget &
CUDA_VISIBLE_DEVICES=2 python render.py -m output/train_budget &
CUDA_VISIBLE_DEVICES=3 python render.py -m output/drjohnson_budget &
CUDA_VISIBLE_DEVICES=4 python render.py -m output/playroom_budget &
wait;

# Evaluation

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/bicycle_budget &
CUDA_VISIBLE_DEVICES=1 python metrics.py -m output/flowers_budget &
CUDA_VISIBLE_DEVICES=2 python metrics.py -m output/garden_budget &
CUDA_VISIBLE_DEVICES=3 python metrics.py -m output/stump_budget &
CUDA_VISIBLE_DEVICES=4 python metrics.py -m output/treehill_budget &
CUDA_VISIBLE_DEVICES=5 python metrics.py -m output/room_budget &
CUDA_VISIBLE_DEVICES=6 python metrics.py -m output/counter_budget &
CUDA_VISIBLE_DEVICES=7 python metrics.py -m output/kitchen_budget &
CUDA_VISIBLE_DEVICES=1 python metrics.py -m output/bonsai_budget &
CUDA_VISIBLE_DEVICES=2 python metrics.py -m output/truck_budget &
CUDA_VISIBLE_DEVICES=3 python metrics.py -m output/train_budget &
CUDA_VISIBLE_DEVICES=4 python metrics.py -m output/drjohnson_budget &
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/playroom_budget &
wait;

CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output/bicycle_big &
CUDA_VISIBLE_DEVICES=1 python metrics.py -m ./output/flowers_big &
CUDA_VISIBLE_DEVICES=2 python metrics.py -m ./output/garden_big &
CUDA_VISIBLE_DEVICES=3 python metrics.py -m ./output/stump_big &
CUDA_VISIBLE_DEVICES=4 python metrics.py -m ./output/treehill_big &
CUDA_VISIBLE_DEVICES=5 python metrics.py -m ./output/room_big &
CUDA_VISIBLE_DEVICES=6 python metrics.py -m ./output/counter_big &
CUDA_VISIBLE_DEVICES=7 python metrics.py -m ./output/kitchen_big &
CUDA_VISIBLE_DEVICES=0 python metrics.py -m ./output/playroom_big &
CUDA_VISIBLE_DEVICES=1 python metrics.py -m ./output/bonsai_big &
CUDA_VISIBLE_DEVICES=2 python metrics.py -m ./output/truck_big &
CUDA_VISIBLE_DEVICES=3 python metrics.py -m ./output/train_big &
CUDA_VISIBLE_DEVICES=4 python metrics.py -m ./output/drjohnson_big &
wait;
