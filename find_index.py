import json
import numpy as np
import sys

target_folder = sys.argv[1]
easy_list = []
med_list = []
hard_list = []
## easy ##
with open('./data/labels/test_easy.json','r') as f:
    easy = list((json.load(f)['scenes']).items())
    for i in range(len(easy)):
        easy_list.append(easy[i][0])

## medium ##
with open('./data/labels/test_medium.json','r') as f:
    medium = list((json.load(f)['scenes']).items())
    for i in range(len(medium)):
        med_list.append(medium[i][0])


## hard ##
with open('./data/labels/test_hard.json','r') as f:
    hard = np.array(list((json.load(f)['scenes']).items()))
    for i in range(len(hard)):
        hard_list.append(hard[i][0])
assert (len(easy_list)+len(med_list)+len(hard_list)) == 1861


with open(target_folder+'results_img_pairs.json','r') as f:
    results = json.load(f)

easy_dict = {}
medium_dict = {}
hard_dict = {}

app_dist_easy = []
app_dist_med = []
app_dist_hard = []

gt_inst_easy = []
gt_inst_med = []
gt_inst_hard = []

angle_diff_easy = []
angle_diff_med = []
angle_diff_hard = []

epi_dist_easy = []
epi_dist_med = []
epi_dist_hard = []

## index ##
for scene, value in results.items():
    scene_name = scene.split(',')[0]
    if scene_name in easy_list:
        easy_dict[scene]=value
        app_dist_easy.append([float(i) for i in value['app_dist']])
        epi_dist_easy.append([float(i) for i in value['epi_dist']])
        gt_inst_easy.append([float(i) for i in value['gt_inst']])
        angle_diff_easy.append([float(i) for i in value['angle_diff']])
    elif scene_name in med_list:
        medium_dict[scene]=value
        app_dist_med.append([float(i) for i in value['app_dist']])
        epi_dist_med.append([float(i) for i in value['epi_dist']])
        gt_inst_med.append([float(i) for i in value['gt_inst']])
        angle_diff_med.append([float(i) for i in value['angle_diff']])
    elif scene_name in hard_list:
        hard_dict[scene]=value
        app_dist_hard.append([float(i) for i in value['app_dist']])
        epi_dist_hard.append([float(i) for i in value['epi_dist']])
        gt_inst_hard.append([float(i) for i in value['gt_inst']])
        angle_diff_hard.append([float(i) for i in value['angle_diff']])
    else:
        raise ValueError("Please use the results of test.json")

app_dist_easy = np.reshape(np.concatenate(app_dist_easy),(-1,1))
epi_dist_easy = np.reshape(np.concatenate(epi_dist_easy),(-1,1))
gt_inst_easy = np.reshape(np.concatenate(gt_inst_easy),(-1,1))
angle_diff_easy = np.reshape(np.concatenate(angle_diff_easy),(-1,1))

app_dist_med = np.reshape(np.concatenate(app_dist_med),(-1,1))
epi_dist_med = np.reshape(np.concatenate(epi_dist_med),(-1,1))
gt_inst_med = np.reshape(np.concatenate(gt_inst_med),(-1,1))
angle_diff_med = np.reshape(np.concatenate(angle_diff_med),(-1,1))

app_dist_hard = np.reshape(np.concatenate(app_dist_hard),(-1,1))
epi_dist_hard = np.reshape(np.concatenate(app_dist_hard),(-1,1))
gt_inst_hard = np.reshape(np.concatenate(gt_inst_hard),(-1,1))
angle_diff_hard = np.reshape(np.concatenate(angle_diff_hard),(-1,1))

assert (app_dist_easy.shape[0] + app_dist_med.shape[0] + app_dist_hard.shape[0] == 47132221)
assert (epi_dist_easy.shape[0] + epi_dist_med.shape[0] + epi_dist_hard.shape[0] == 47132221)
assert (gt_inst_easy.shape[0] + gt_inst_med.shape[0] + gt_inst_hard.shape[0] == 47132221)
assert (angle_diff_easy.shape[0] + angle_diff_med.shape[0] + angle_diff_hard.shape[0] == 47132221)

with open(target_folder+'results_img_pairs_easy.json','w') as f:
    json.dump(easy_dict,f)
with open(target_folder+'results_img_pairs_medium.json','w') as f:
    json.dump(medium_dict,f)
with open(target_folder+'results_img_pairs_hard.json','w') as f:
    json.dump(hard_dict,f)

## app_dist ##
np.save(target_folder+'app_dist_easy.npy',app_dist_easy)
np.save(target_folder+'app_dist_medium.npy',app_dist_med)
np.save(target_folder+'app_dist_hard.npy',app_dist_hard)

## epi_dist ##
np.save(target_folder+'epi_dist_easy.npy',epi_dist_easy)
np.save(target_folder+'epi_dist_medium.npy',epi_dist_med)
np.save(target_folder+'epi_dist_hard.npy',epi_dist_hard)

## gt_inst ##
np.save(target_folder+'gt_inst_np_easy.npy',gt_inst_easy)
np.save(target_folder+'gt_inst_np_medium.npy',gt_inst_med)
np.save(target_folder+'gt_inst_np_hard.npy',gt_inst_hard)

## gt_inst ##
np.save(target_folder+'angle_diff_easy.npy',angle_diff_easy)
np.save(target_folder+'angle_diff_medium.npy',angle_diff_med)
np.save(target_folder+'angle_diff_hard.npy',angle_diff_hard)
