import numpy as np
import numpy.typing as npt
from pandas import DataFrame

import materials
import common as ak


RERUN_SUBSTR = "_rep"
PERM_SUBSTR = "_perm"
TOUGH_SUBSTR = "tough_"
MAIN_SUBSTR  = "main_"
SUB_SUBSTR = "sub_"
CHALLENGE_MODE_SUFFIX = "#"

RERUN_SUFFIX = " (rerun)"
TOUGH_SUFFIX = " (tough)"
PERM_SUFFIX = " (permanent)"


def get_stage_info(df: DataFrame, stage_dict, include_reruns=True):

    stage_san_cost = {} # stage_id -> sanity cost
    stage_names = {} # stage_id -> stage_name
    stage_names_rev = {} # stage_name -> stage_id
    stage_lmd = {} # stage_id -> lmd

    known_stage_ids = np.array(np.unique(df.stageId), dtype="U32")

    rerun_indices = np.flatnonzero(np.core.defchararray.find(known_stage_ids,RERUN_SUBSTR)!=-1)
    rerun_stage_ids = known_stage_ids[rerun_indices]

    perm_indices = np.flatnonzero(np.core.defchararray.find(known_stage_ids,PERM_SUBSTR)!=-1)
    perm_stage_ids = known_stage_ids[perm_indices]

    tough_indices = np.flatnonzero(np.core.defchararray.find(known_stage_ids,TOUGH_SUBSTR)!=-1)
    tough_stage_ids = known_stage_ids[tough_indices]

    for stage_id, info in stage_dict["stages"].items():

        if stage_id[-1] == CHALLENGE_MODE_SUFFIX:
            continue

        san_cost = info['apCost']
        code_ = info['code']
        lmd = info['goldGain'] * ak.LMD_EXP_3STAR_MULT

        if stage_id in tough_stage_ids:
            code = code_ + TOUGH_SUFFIX
        else:
            code = code_

        stage_san_cost[stage_id] = san_cost
        stage_names[stage_id] = code
        stage_names_rev[code] = stage_id
        stage_lmd[stage_id] = lmd

        ## Why is this needed, if it is?
        potential_rerun_id = stage_id + RERUN_SUBSTR
        if potential_rerun_id in rerun_stage_ids:
            code = code_ + RERUN_SUFFIX

            stage_san_cost[potential_rerun_id] = san_cost
            stage_names[potential_rerun_id] = code
            stage_names_rev[code] = potential_rerun_id
            stage_lmd[stage_id] = lmd

        potential_perm_id = stage_id + PERM_SUBSTR
        if potential_perm_id in perm_stage_ids:
            code = code_ + PERM_SUFFIX

            stage_san_cost[potential_perm_id] = san_cost
            stage_names[potential_perm_id] = code
            stage_names_rev[code] = potential_perm_id
            stage_lmd[stage_id] = lmd


    return stage_san_cost, stage_names, stage_names_rev, stage_lmd


def get_event_info(stage_names: dict) -> (dict, dict):
    event_names = {}
    event_names_rev = {}

    for k, v in stage_names.items():
        identifier = k.split("_")[0]
        code = v.split("-")[0]

        event_names[identifier] = code
        event_names_rev[code] = identifier

    return event_names, event_names_rev


def get_main_stage_ids(psdf: DataFrame):
    all_stage_ids = np.array(np.unique(psdf.stageId), dtype="U32")

    main_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,MAIN_SUBSTR)!=-1)
    sub_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,SUB_SUBSTR)!=-1)

    indices = np.concatenate((main_indices, sub_indices))
    main_stage_ids = all_stage_ids[indices]

    return main_stage_ids

## TODO: make a parameter in get_main_stage_ids()
def get_main_and_perm_stage_ids(psdf: DataFrame):
    all_stage_ids = np.array(np.unique(psdf.stageId), dtype="U32")

    main_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,MAIN_SUBSTR)!=-1)
    sub_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,SUB_SUBSTR)!=-1)
    perm_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,PERM_SUBSTR)!=-1)

    indices = np.concatenate((main_indices, sub_indices, perm_indices))
    main_stage_ids = all_stage_ids[indices]

    return main_stage_ids


def get_event_ids(event: str, psdf: DataFrame, event_names_rev: dict, remove_permanent=True, remove_reruns=False):
    all_stage_ids = np.array(np.unique(psdf.stageId), dtype="U32")
    event_substr = event_names_rev[event]
    event_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,event_substr)!=-1)
    matching_ids = all_stage_ids[event_indices]

    if remove_permanent:
        non_perm_indices = np.flatnonzero(np.core.defchararray.find(matching_ids,PERM_SUBSTR)==-1)
        matching_ids = matching_ids[non_perm_indices]

    if remove_reruns:
        non_rerun_indices = np.flatnonzero(np.core.defchararray.find(matching_ids,RERUN_SUBSTR)==-1)
        matching_ids = matching_ids[non_rerun_indices]

    return matching_ids


def get_san_cost(stage_ids, stage_san_cost: dict):
    n_stages = len(stage_ids)
    stage_san = np.zeros(n_stages)

    for i, v in enumerate(stage_ids):
        stage_san[i] = float(stage_san_cost[v])

    return stage_san


def get_main_drops(drop_matrix: npt.NDArray, msv: npt.NDArray, material_ids: npt.NDArray) -> npt.NDArray:
    n_stages = len(drop_matrix)
    main_mats = np.empty(n_stages, dtype="U32")

    for i, drops in enumerate(drop_matrix):
        idx = np.argmax(drops*msv)
        main_mats[i] = material_ids[idx]

    return main_mats


def print_stage_drops(stage_name: str, drop_matrix, stage_ids, item_ids, stage_names_rev: dict, item_names):
    stage_id = stage_names_rev[stage_name]
    idx = np.where(stage_ids == stage_id)[0][0]

    drops = []

    for i, v in enumerate(drop_matrix[idx]):
        if v > 0:
            drops.append((item_names[item_ids[i]], v))

    #rates = np.array(drops[:,1])
    drops = np.array(drops)
    rates = drops[:,1]
    indices = np.flip(np.argsort(rates))
    drops = drops[indices]

    for n, v in drops:
        v = float(v)
        if n == "LMD":
            print("{}: {:d}".format(n, int(v)))
        else:
            print("{}: {:.2f}%".format(n, v*100))


def get_stage_efficiency(drop_matrix, msv, stage_san):
    san_return = np.matmul(drop_matrix, msv)
    stage_efficiency = san_return/stage_san
    return stage_efficiency


def get_drop_matrix(psdf: DataFrame, stage_ids, item_ids, stage_lmd: dict):
    n_stages = len(stage_ids)
    n_items = len(item_ids)

    lmd_idx = np.where(item_ids == materials.LMD_ID)[0]

    drop_matrix = np.zeros((n_stages, n_items))
    coeff_matrix = np.zeros(n_items)
    helper_matrix = np.ones(n_items)

    for v in psdf.values:
        try:
            stage_idx = np.where(stage_ids == v[0])[0]
            item_idx = np.where(item_ids == v[1])[0]
            drop_matrix[stage_idx, item_idx] = v[3]/v[2]
            drop_matrix[stage_idx, lmd_idx] = stage_lmd[str(v[0])]

            coeff_matrix[item_idx] += v[3]
            helper_matrix[item_idx] += v[2]
        except Exception as e:
            #print("Error: {} \"{}\"".format(v, e))
            pass

    return drop_matrix, coeff_matrix/helper_matrix


def drop_matrix_cutoff(drop_matrix: npt.NDArray, threshold: float) -> npt.NDArray:

    for i in range(len(drop_matrix)):
        for j in range(len(drop_matrix[i])):
            if drop_matrix[i][j] < threshold:
                drop_matrix[i][j] = 0

    return drop_matrix
def filter_drop_matrix(drop_matrix, stage_san, stage_ids):
    drop_sum = np.sum(drop_matrix, axis=1)
    has_drops = np.where(drop_sum > 0)[0]
    has_san = np.where(stage_san > 0)[0]

    #indices = np.unique(np.concatenate((has_drops, has_san)))
    indices = np.intersect1d(has_drops, has_san)

    return drop_matrix[indices], stage_san[indices], stage_ids[indices]


