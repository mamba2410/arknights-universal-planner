import numpy as np
import numpy.typing as npt
from pandas import DataFrame

import materials
import common as ak

PURE_GOLD_LMD_VALUE = 500
TIER1_EXP_VALUE = 200
TIER2_EXP_VALUE = 400
TIER3_EXP_VALUE = 1000
TIER4_EXP_VALUE = 2000


def get_craft_matrix(craft_dict: dict, material_ids, item_names_rev):
    n_mats = len(material_ids)
    lmd_idx = np.where(material_ids == materials.LMD_ID)[0]
    pure_gold_idx = np.where(material_ids == materials.PURE_GOLD_ID)[0]
    tier1_exp_idx = np.where(material_ids == materials.TIER1_EXP_ID)[0]
    tier2_exp_idx = np.where(material_ids == materials.TIER2_EXP_ID)[0]
    tier3_exp_idx = np.where(material_ids == materials.TIER3_EXP_ID)[0]
    tier4_exp_idx = np.where(material_ids == materials.TIER4_EXP_ID)[0]

    craft_matrix = np.zeros((n_mats, n_mats))
    subprod_matrix = np.zeros((n_mats, n_mats))
    recipes = [v for v in craft_dict["workshopFormulas"].values()]

    #craft_matrix[lmd_idx, pure_gold_idx] = PURE_GOLD_LMD_VALUE
    craft_matrix[tier2_exp_idx, tier1_exp_idx] = TIER2_EXP_VALUE/TIER1_EXP_VALUE
    craft_matrix[tier3_exp_idx, tier2_exp_idx] = TIER3_EXP_VALUE/TIER2_EXP_VALUE
    craft_matrix[tier4_exp_idx, tier3_exp_idx] = TIER4_EXP_VALUE/TIER3_EXP_VALUE


    ## Assume only one recipe per item
    for r in recipes:
        item_idx = np.where(material_ids == r["itemId"])[0]

        if len(item_idx) <= 0:
            continue

        craft_matrix[item_idx,lmd_idx] = r["goldCost"]

        for c in r["costs"]:
            c_idx = np.where(material_ids == c["id"])[0]
            if len(c_idx) <= 0:
                continue
            craft_matrix[item_idx,c_idx] = float(c["count"])

        total_w = 0
        for s in r["extraOutcomeGroup"]:
            total_w += float(s["weight"])
            s_idx = np.where(material_ids == s["itemId"])[0]
            subprod_matrix[item_idx,s_idx] = float(s["weight"])

        subprod_matrix[item_idx] /= total_w

    ## Manual recipes

    chip_t1_ids  = ak.get_material_ids(item_names_rev, materials.chip_t1)
    chip_t2_ids  = ak.get_material_ids(item_names_rev, materials.chip_t2)
    dualchip_ids = ak.get_material_ids(item_names_rev, materials.dualchips)
    _, chip_t1_indices, _ = np.intersect1d(material_ids, chip_t1_ids, return_indices=True)
    _, chip_t2_indices, _ = np.intersect1d(material_ids, chip_t2_ids, return_indices=True)
    _, dualchip_indices, _ = np.intersect1d(material_ids, dualchip_ids, return_indices=True)

    ## remove swapping T1 chip types
    for i in chip_t1_indices:
        for j in range(n_mats):
            #craft_matrix[i][j] = 0
            craft_matrix[j][i] = 0

    ## remove swapping T2 chip types
    for i in chip_t2_indices:
        for j in range(n_mats):
            #craft_matrix[i][j] = 0
            craft_matrix[j][i] = 0

    ## add dualchip crafting
    for i, k in enumerate(dualchip_indices):
        craft_matrix[k][chip_t2_indices[i]] = 2

    return craft_matrix, subprod_matrix


def print_craft_materials(item_name: str, craft_matrix, item_ids, item_names: dict, item_names_rev: dict):
    item_id = item_names_rev[item_name]
    idx = np.where(item_ids == item_id)[0][0]

    for i, v in enumerate(craft_matrix[idx]):
        if v > 0:
            print("{}: {:d}".format(
                item_names[item_ids[i]],
                int(v)
                ))


def get_breakdown_matrix(craft_matrix: npt.NDArray, material_ids: npt.NDArray,
        item_names_rev: dict) -> npt.NDArray:

    n_mats = len(material_ids)
    breakdown_matrix = np.matrix.copy(craft_matrix.T)

    tier_1_ids = ak.get_material_ids(item_names_rev, materials.tier_1_names)
    tier_2_ids = ak.get_material_ids(item_names_rev, materials.tier_2_names)
    tier_3_ids = ak.get_material_ids(item_names_rev, materials.tier_3_names)
    ids_combined = np.concatenate((tier_1_ids, tier_2_ids))
    ids_combined, ids_indices, _ = np.intersect1d(material_ids, ids_combined, return_indices=True)
    _, tier_3_indices, _ = np.intersect1d(material_ids, tier_3_ids, return_indices=True)

    chip_ids = ak.get_material_ids(item_names_rev, materials.CHIP_NAMES)
    chip_t1_ids = ak.get_material_ids(item_names_rev, materials.chip_t1)
    chip_t2_ids = ak.get_material_ids(item_names_rev, materials.chip_t2)
    dualchip_ids = ak.get_material_ids(item_names_rev, materials.dualchips)
    _, chip_t1_indices, _ = np.intersect1d(material_ids, chip_t1_ids, return_indices=True)
    _, chip_t2_indices, _ = np.intersect1d(material_ids, chip_t2_ids, return_indices=True)
    _, dualchip_indices, _ = np.intersect1d(material_ids, dualchip_ids, return_indices=True)


    rocc2_idx = np.where(material_ids == item_names_rev["Orirock Cube"])[0][0]
    rocc3_idx = np.where(material_ids == item_names_rev["Orirock Cluster"])[0][0]
    lmd_idx = np.where(material_ids == materials.LMD_ID)[0][0]

    ## remove deconstructing T3->T2 and T2->T1
    rocc_before = breakdown_matrix[rocc2_idx][rocc3_idx]
    for i in ids_indices:
        for j in range(n_mats):
            breakdown_matrix[i][j] = 0
        breakdown_matrix[i][i] = 1

    ## remove lmd cost
    #for j in range(n_mats):
    #    breakdown_matrix[lmd_idx][j] = 0

    ## make T3 mats and LMD persist
    breakdown_matrix[lmd_idx][lmd_idx] = 1
    for i in tier_3_indices:
        if i == rocc3_idx:
            breakdown_matrix[rocc2_idx][rocc3_idx] = rocc_before
            continue
        breakdown_matrix[i][i] = 1

    ## remove swapping T1 chip types
    for i in chip_t1_indices:
        breakdown_matrix[i][i] = 1

    ## remove swapping T2 chip types
    for i in chip_t2_indices:
        breakdown_matrix[i][i] = 1

    return breakdown_matrix

