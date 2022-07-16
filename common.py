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

CHIP_SUBSTR = "Chip"

LMD_EXP_3STAR_MULT = 1.2

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


def get_item_info(item_dict: dict) -> (dict, dict):

    item_names = {} # item_id -> item_name
    item_names_rev = {} # item_name -> item_id

    for item_id, info in item_dict["items"].items():
        item_name = info["name"]

        item_names[item_id] = item_name
        item_names_rev[item_name] = item_id

    return item_names, item_names_rev


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


def get_craft_constraint_matrix(craft_matrix, subprod_matrix, byprod_rate):
    craft_sum = np.sum(craft_matrix, axis=1)
    subprod_sum = np.sum(craft_matrix, axis=1)

    identity = np.identity(len(craft_sum))

    has_craft = np.where(craft_sum > 0)[0]
    has_subprod = np.where(subprod_sum > 0)[0]

    indices = np.intersect1d(has_craft, has_subprod)

    ccm = identity - craft_matrix + byprod_rate*subprod_matrix

    return ccm[indices]


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


def materials_to_csv(file: str, array: npt.NDArray, item_names: dict) -> None:
    with open(file, "w") as f:
        for item_id, count in array:
            f.write("{},{},{}\n".format(item_id, item_names[item_id], count))


def csv_to_materials(file: str, mats: npt.NDArray, item_names: dict) -> npt.NDArray:
    raw = np.genfromtxt(file, delimiter=",", dtype=[("item_id", "U32"), ("_","U32"), ("count", "int32")])
    ret = np.zeros(len(mats), dtype=[("item_id", "U32"), ("count", "int32")])

    ret["item_id"] = mats
    for i in range(len(mats)):
        try:
            mat_idx = np.where(raw["item_id"] == mats[i])[0][0]
            ret["count"][i] = raw["count"][mat_idx]
        except Exception as e:
            print("Error: {}\n{}".format(e, item_names[mats[i]]))

    return ret
