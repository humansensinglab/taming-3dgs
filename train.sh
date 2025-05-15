python train.py -s data//bicycle -i images_4 -m ./eval/bicycle_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 15  --densification_interval 500 --mode multiplier
python train.py -s data//flowers -i images_4 -m ./eval/flowers_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 15  --densification_interval 500 --mode multiplier
python train.py -s data//garden -i images_4 -m ./eval/garden_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 15  --densification_interval 500 --mode multiplier
python train.py -s data//stump -i images_4 -m ./eval/stump_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 15  --densification_interval 500 --mode multiplier
python train.py -s data//treehill -i images_4 -m ./eval/treehill_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 15  --densification_interval 500 --mode multiplier
python train.py -s data//room -i images_2 -m ./eval/room_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 2  --densification_interval 500 --mode multiplier
python train.py -s data//counter -i images_2 -m ./eval/counter_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 2  --densification_interval 500 --mode multiplier
python train.py -s data//kitchen -i images_2 -m ./eval/kitchen_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 2  --densification_interval 500 --mode multiplier
python train.py -s data//bonsai -i images_2 -m ./eval/bonsai_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 2  --densification_interval 500 --mode multiplier
python train.py -s data//truck -m ./eval/truck_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 2  --densification_interval 500 --mode multiplier
python train.py -s data//train -m ./eval/train_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 2  --densification_interval 500 --mode multiplier
python train.py -s data//drjohnson -m ./eval/drjohnson_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 5  --densification_interval 500 --mode multiplier
python train.py -s data//playroom -m ./eval/playroom_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 5  --densification_interval 500 --mode multiplier
python render.py -m ./eval/bicycle_budget
python render.py -m ./eval/flowers_budget
python render.py -m ./eval/garden_budget
python render.py -m ./eval/stump_budget
python render.py -m ./eval/treehill_budget
python render.py -m ./eval/room_budget
python render.py -m ./eval/counter_budget
python render.py -m ./eval/kitchen_budget
python render.py -m ./eval/bonsai_budget
python render.py -m ./eval/truck_budget
python render.py -m ./eval/train_budget
python render.py -m ./eval/drjohnson_budget
python render.py -m ./eval/playroom_budget
python metrics.py -m ./eval/bicycle_budget
python metrics.py -m ./eval/flowers_budget
python metrics.py -m ./eval/garden_budget
python metrics.py -m ./eval/stump_budget
python metrics.py -m ./eval/treehill_budget
python metrics.py -m ./eval/room_budget
python metrics.py -m ./eval/counter_budget
python metrics.py -m ./eval/kitchen_budget
python metrics.py -m ./eval/bonsai_budget
python metrics.py -m ./eval/truck_budget
python metrics.py -m ./eval/train_budget
python metrics.py -m ./eval/drjohnson_budget
python metrics.py -m ./eval/playroom_budget
python train.py -s data//bicycle -i images_4 -m ./eval/bicycle_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 5987095  --densification_interval 100 --mode final_count
python train.py -s data//flowers -i images_4 -m ./eval/flowers_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 3618411  --densification_interval 100 --mode final_count
python train.py -s data//garden -i images_4 -m ./eval/garden_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 5728191  --densification_interval 100 --mode final_count
python train.py -s data//stump -i images_4 -m ./eval/stump_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 4867429  --densification_interval 100 --mode final_count
python train.py -s data//treehill -i images_4 -m ./eval/treehill_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 3770257  --densification_interval 100 --mode final_count
python train.py -s data//room -i images_2 -m ./eval/room_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 1548960  --densification_interval 100 --mode final_count
python train.py -s data//counter -i images_2 -m ./eval/counter_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 1190919  --densification_interval 100 --mode final_count
python train.py -s data//kitchen -i images_2 -m ./eval/kitchen_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 1803735  --densification_interval 100 --mode final_count
python train.py -s data//bonsai -i images_2 -m ./eval/bonsai_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 1252367  --densification_interval 100 --mode final_count
python train.py -s data//truck -m ./eval/truck_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 2584171  --densification_interval 100 --mode final_count
python train.py -s data//train -m ./eval/train_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 1085480  --densification_interval 100 --mode final_count
python train.py -s data//drjohnson -m ./eval/drjohnson_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 3273600  --densification_interval 100 --mode final_count
python train.py -s data//playroom -m ./eval/playroom_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 2326100  --densification_interval 100 --mode final_count
python render.py -m ./eval/bicycle_big
python render.py -m ./eval/flowers_big
python render.py -m ./eval/garden_big
python render.py -m ./eval/stump_big
python render.py -m ./eval/treehill_big
python render.py -m ./eval/room_big
python render.py -m ./eval/counter_big
python render.py -m ./eval/kitchen_big
python render.py -m ./eval/bonsai_big
python render.py -m ./eval/truck_big
python render.py -m ./eval/train_big
python render.py -m ./eval/drjohnson_big
python render.py -m ./eval/playroom_big
python metrics.py -m ./eval/bicycle_big
python metrics.py -m ./eval/flowers_big
python metrics.py -m ./eval/garden_big
python metrics.py -m ./eval/stump_big
python metrics.py -m ./eval/treehill_big
python metrics.py -m ./eval/room_big
python metrics.py -m ./eval/counter_big
python metrics.py -m ./eval/kitchen_big
python metrics.py -m ./eval/bonsai_big
python metrics.py -m ./eval/truck_big
python metrics.py -m ./eval/train_big
python metrics.py -m ./eval/drjohnson_big
python metrics.py -m ./eval/playroom_big