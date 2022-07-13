import numpy as np
import numpy.typing as npt
import requests
import pandas
from pandas import DataFrame
import scipy
from cost_packet import CostPacket
import json
import os
import materials

LOCAL_BASE = "./akdata/"
GAMEDATA_BASE = "https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/"
ITEM_TABLE_LOC = "/gamedata/excel/item_table.json"
STAGE_TABLE_LOC = "/gamedata/excel/stage_table.json"
CRAFT_TABLE_LOC = "/gamedata/excel/building_data.json"
CHAR_TABLE_LOC = "/gamedata/excel/character_table.json"
MODULE_TABLE_LOC = "/gamedata/excel/uniequip_table.json"
LEVEL_TABLE_LOC = "/gamedata/excel/gamedata_const.json"
PENGSTATS_BASE = "https://penguin-stats.io/PenguinStats/api/v2/result/matrix?"

RERUN_SUBSTR = "_rep"
PERM_SUBSTR = "_perm"
TOUGH_SUBSTR = "tough_"
MAIN_SUBSTR  = "main_"
SUB_SUBSTR = "sub_"
CHALLENGE_MODE_SUFFIX = "#"
CHIP_SUBSTR = "Chip"

RERUN_SUFFIX = " (rerun)"
TOUGH_SUFFIX = " (tough)"
PERM_SUFFIX = " (permanent)"

LMD_EXP_3STAR_MULT = 1.2
PURE_GOLD_LMD_VALUE = 500
TIER1_EXP_VALUE = 200
TIER2_EXP_VALUE = 400
TIER3_EXP_VALUE = 1000
TIER4_EXP_VALUE = 2000

COST_DTYPE = [("item_id", "U32", 4), ("count", "int32", 4)]
MATERIAL_DTYPE = [("item_id", "U32"), ("count", "int32")]

def get_dict(remote_base: str, lang: str, table: str, local: bool, local_base: str) -> dict:
    local_table = local_base+lang+table
    #local_table_dir = os.path.dirname(os.path.realpath(local_table))
    local_table_dir = os.path.dirname(local_table)
    
    if not os.path.exists(local_table_dir):
        os.makedirs(local_table_dir, exist_ok=True)
        
    if local and os.path.exists(local_table):
        print("Reading from local file")
        with open(local_table, "r") as f:
            ret = json.load(f)
    else:
        print("Reading from remote")
        response = requests.get(remote_base+lang+table)
        ret = response.json()
        with open(local_table, "w") as f:
            json.dump(ret, f)
    return ret

def get_item_dict(lang="zh_CN", local=False):
    return get_dict(GAMEDATA_BASE, lang, ITEM_TABLE_LOC, local, LOCAL_BASE)

def get_stage_dict(lang="zh_CN", local=False):
    return get_dict(GAMEDATA_BASE, lang, STAGE_TABLE_LOC, local, LOCAL_BASE)

def get_craft_dict(lang="zh_CN", local=False):
    return get_dict(GAMEDATA_BASE, lang, CRAFT_TABLE_LOC, local, LOCAL_BASE)

def get_char_dict(lang="zh_CN", local=False):
    return get_dict(GAMEDATA_BASE, lang, CHAR_TABLE_LOC, local, LOCAL_BASE)

def get_module_dict(lang="zh_CN", local=False):
    return get_dict(GAMEDATA_BASE, lang, MODULE_TABLE_LOC, local, LOCAL_BASE)

def get_level_dict(lang="zh_CN", local=False):
    return get_dict(GAMEDATA_BASE, lang, LEVEL_TABLE_LOC, local, LOCAL_BASE)

def get_pengstats_df(show_closed_stages=True, server="CN"):
    url = PENGSTATS_BASE+f"server={server}"

    if show_closed_stages:
        url += "&show_closed_zones=true"

    response = requests.get(url)

    df = pandas.DataFrame.from_records(response.json()['matrix'])

    return df


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
        lmd = info['goldGain'] * LMD_EXP_3STAR_MULT
        
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


def get_item_info(item_dict: dict) -> (dict, dict):

    item_names = {} # item_id -> item_name
    item_names_rev = {} # item_name -> item_id
    
    for item_id, info in item_dict["items"].items():
        item_name = info["name"]

        item_names[item_id] = item_name
        item_names_rev[item_name] = item_id

    return item_names, item_names_rev

def get_event_info(stage_names: dict) -> (dict, dict):
    event_names = {}
    event_names_rev = {}

    for k, v in stage_names.items():
        identifier = k.split("_")[0]
        code = v.split("-")[0]

        event_names[identifier] = code
        event_names_rev[code] = identifier

    return event_names, event_names_rev

def get_char_info(char_dict: dict) -> (dict, dict):
    char_names = {} # char_id -> char_name
    char_names_rev = {} # char_name -> char_id
    
    for k, v in char_dict.items():
        char_names[k] = v["name"]
        char_names_rev[v["name"]] = k
       
    return char_names, char_names_rev

def get_char_translations(char_dict: dict) -> (dict, dict):
    char_names_en = {}
    char_names_en_rev = {}
    
    for k, v in char_dict.items():
        char_names_en[k] = v["appellation"]
        char_names_en_rev[v["appellation"]] = k
        
    return char_names_en, char_names_en_rev

def get_char_rarities(char_dict: dict) -> dict:
    char_rarities = {}
    
    for k, v in char_dict.items():
        char_rarities[k] = v["rarity"]
        
    return char_rarities

def get_level_info(level_dict: dict) -> (npt.NDArray, npt.NDArray, npt.NDArray):
    xp_map = np.array(level_dict["characterExpMap"], dtype="int")
    lmd_map = np.array(level_dict["characterUpgradeCostMap"], dtype="int")
    elite_map = np.array(level_dict["evolveGoldCost"], dtype="int")
    max_level_map_ = np.array(level_dict["maxLevel"], dtype=object, ndmin=1)
    max_level_shape = (len(max_level_map_), len(max_level_map_[-1]))
    max_level_map = np.zeros(max_level_shape, dtype="int")
    for i, a in enumerate(max_level_map_):
        for j, b in enumerate(a):
            max_level_map[i][j] = b                  
    
    return xp_map, lmd_map, elite_map, max_level_map
    

def get_material_ids(item_names_rev: dict, names: list):
    n_mats = len(names)
    material_ids = np.empty(n_mats, dtype="U32")

    for i, name in enumerate(names):
        material_ids[i] = item_names_rev[name]

    return material_ids


def get_main_stage_ids(psdf: DataFrame):
    all_stage_ids = np.array(np.unique(psdf.stageId), dtype="U32")

    main_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,MAIN_SUBSTR)!=-1)
    sub_indices = np.flatnonzero(np.core.defchararray.find(all_stage_ids,SUB_SUBSTR)!=-1)

    indices = np.concatenate((main_indices, sub_indices))
    main_stage_ids = all_stage_ids[indices]

    return main_stage_ids

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
    
    chip_t1_ids = get_material_ids(item_names_rev, materials.chip_t1)
    chip_t2_ids = get_material_ids(item_names_rev, materials.chip_t2)
    dualchip_ids = get_material_ids(item_names_rev, materials.dualchips)
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

def get_san_cost(stage_ids, stage_san_cost: dict):
    n_stages = len(stage_ids)
    stage_san = np.zeros(n_stages)

    for i, v in enumerate(stage_ids):
        stage_san[i] = float(stage_san_cost[v])

    return stage_san


def filter_drop_matrix(drop_matrix, stage_san, stage_ids):
    drop_sum = np.sum(drop_matrix, axis=1)
    has_drops = np.where(drop_sum > 0)[0]
    has_san = np.where(stage_san > 0)[0]

    #indices = np.unique(np.concatenate((has_drops, has_san)))
    indices = np.intersect1d(has_drops, has_san)

    return drop_matrix[indices], stage_san[indices], stage_ids[indices]

def drop_matrix_cutoff(drop_matrix: npt.NDArray, threshold: float) -> npt.NDArray:

    for i in range(len(drop_matrix)):
        for j in range(len(drop_matrix[i])):
            if drop_matrix[i][j] < threshold:
                drop_matrix[i][j] = 0
    
    return drop_matrix


def get_craft_constraint_matrix(craft_matrix, subprod_matrix, byprod_rate):
    craft_sum = np.sum(craft_matrix, axis=1)
    subprod_sum = np.sum(craft_matrix, axis=1)

    identity = np.identity(len(craft_sum))

    has_craft = np.where(craft_sum > 0)[0]
    has_subprod = np.where(subprod_sum > 0)[0]

    indices = np.intersect1d(has_craft, has_subprod)

    ccm = identity - craft_matrix + byprod_rate*subprod_matrix

    return ccm[indices]

def get_stage_efficiency(drop_matrix, msv, stage_san):
    san_return = np.matmul(drop_matrix, msv)
    stage_efficiency = san_return/stage_san
    return stage_efficiency


def print_craft_materials(item_name: str, craft_matrix, item_ids, item_names: dict, item_names_rev: dict):
    item_id = item_names_rev[item_name]
    idx = np.where(item_ids == item_id)[0][0]
    
    for i, v in enumerate(craft_matrix[idx]):
        if v > 0:
            print("{}: {:d}".format(
                item_names[item_ids[i]],
                int(v)
                ))

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

def get_main_drops(drop_matrix: npt.NDArray, msv: npt.NDArray, material_ids: npt.NDArray) -> npt.NDArray:
    n_stages = len(drop_matrix)
    main_mats = np.empty(n_stages, dtype="U32")

    for i, drops in enumerate(drop_matrix):
        idx = np.argmax(drops*msv)
        main_mats[i] = material_ids[idx]

    return main_mats
        


            
## Assume homogeneous array of COST_DTYPE
def sum_skill_slice(array: npt.NDArray) -> npt.NDArray:
    ## TODO: use np.flatnonzero?
    length = len(array) * len(array[0][0])
    flat = np.empty(length, dtype=[("item_id", "U32"), ("count", "int32")])

    j = 0
    for thing in array:
        thing_len = len(thing[0])
        for k in range(thing_len):
            flat[j] = (thing["item_id"][k], thing["count"][k])
            j += 1
    
    new_array = collapse_item_list(flat)
    
    return new_array

def collapse_item_list(array: npt.NDArray) -> npt.NDArray:
    unique_ids = np.unique(list(filter(None, array["item_id"])))
    n_uniques = len(unique_ids)
    uniques = np.empty(n_uniques, dtype=[("item_id", "U32"), ("count", "int32")])
    uniques["item_id"] = unique_ids
    
    ## TODO: Better way to do this with np.unique returning indices
    for item_id, count in array:
        item_idx = np.where(unique_ids == item_id)[0]
        uniques["count"][item_idx] += count
        
    return uniques
        


## TODO: hardcoded sizes
def get_all_char_all_costs(char_dict: dict, module_dict: dict, level_dict: dict, n_operators: int) \
    -> (CostPacket, npt.NDArray):
    
    level_exp_map, level_lmd_map, elite_lmd_map, max_level_map = get_level_info(level_dict)
    char_rarity_dict = get_char_rarities(char_dict)
    
    level_costs = np.empty(np.shape(level_lmd_map), dtype=COST_DTYPE)
    for e in range(len(level_lmd_map)):
        for l in range(len(level_lmd_map[-1])):
            level_costs[e][l]["item_id"][0] = materials.LMD_ID
            level_costs[e][l]["count"][0]   = max(level_lmd_map[e][l], 0)
            level_costs[e][l]["item_id"][1] = materials.EXP_ID
            level_costs[e][l]["count"][1]   = max(level_exp_map[e][l], 0)
            
    elite_costs = np.zeros((n_operators, 2), dtype=COST_DTYPE)
    skill_costs = np.zeros((n_operators, 6), dtype=COST_DTYPE)
    mastery_costs = np.zeros((n_operators, 3, 3), dtype=COST_DTYPE)
    module_costs = np.zeros((n_operators, 2, 3), dtype=COST_DTYPE)

    char_ids = np.empty(n_operators, dtype="U32")
    char_n_modules = np.zeros(n_operators, dtype="uint32")
    for char_idx, (char_id, v) in enumerate(char_dict.items()):
        char_ids[char_idx] = char_id
        rarity = char_rarity_dict[char_id]

        ## Promotion costs
        for p_idx, p in enumerate(v["phases"]):
            cost_arr = p["evolveCost"]
            if p["evolveCost"] is not None:
                c_idx = 0
                for c in cost_arr:
                    elite_costs[char_idx][p_idx-1]["item_id"][c_idx] = c["id"]
                    elite_costs[char_idx][p_idx-1]["count"][c_idx] = c["count"]
                    c_idx += 1
                if elite_lmd_map[rarity][p_idx-1] > 0:
                    elite_costs[char_idx][p_idx-1]["item_id"][c_idx] = materials.LMD_ID
                    elite_costs[char_idx][p_idx-1]["count"][c_idx] = elite_lmd_map[rarity][p_idx-1]
                
        ## General skills
        for s_idx, s in enumerate(v["allSkillLvlup"]):
            if s["lvlUpCost"] is not None:
                for c_idx, c in enumerate(s["lvlUpCost"]):
                    skill_costs[char_idx][s_idx]["item_id"][c_idx] = c["id"]
                    skill_costs[char_idx][s_idx]["count"][c_idx] = c["count"]

        ## Skill masteries
        for s_idx, s in enumerate(v["skills"]):
            for m_idx, m in enumerate(s["levelUpCostCond"]):
                if m["levelUpCost"] is not None:
                    for c_idx, c in enumerate(m["levelUpCost"]):
                        mastery_costs[char_idx][s_idx][m_idx]["item_id"][c_idx] = c["id"]
                        mastery_costs[char_idx][s_idx][m_idx]["count"][c_idx] = c["count"]

    ## Modules
    for i, v in enumerate(module_dict["equipDict"].values()):
        char_id = v["charId"]
        cost = v["itemCost"]

        if cost is None:
            continue

        char_idx = np.where(char_ids == char_id)[0][0]
        module_number = char_n_modules[char_idx]

        module_level = 0
        if "1" in cost.keys():
            for module_level, costs in cost.items():
                module_level = int(module_level) - 1
                for item_idx, item in enumerate(costs):
                    module_costs[char_idx][module_number][module_level]["item_id"][item_idx] = item["id"]
                    module_costs[char_idx][module_number][module_level]["count"][item_idx] = int(item["count"])
        else:
            for item_idx, item in enumerate(cost.values()):
                module_costs[char_idx][module_number][module_level]["item_id"][item_idx] = item["id"]
                module_costs[char_idx][module_number][module_level]["count"][item_idx] = int(item["count"])
        
        char_n_modules[char_idx] += 1
        
    return CostPacket(char_ids, elite_costs, skill_costs, mastery_costs,
                      module_costs, char_rarity_dict, level_costs, max_level_map), char_n_modules



