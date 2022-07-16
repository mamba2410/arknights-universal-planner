import numpy as np
import numpy.typing as npt
from pandas import DataFrame

import char, stage, materials
import common as ak
from cost_packet import CostPacket


def get_char_translations(char_dict: dict) -> (dict, dict):
    char_names_en = {}
    char_names_en_rev = {}

    for k, v in char_dict.items():
        char_names_en[k] = v["appellation"]
        char_names_en_rev[v["appellation"]] = k

    return char_names_en, char_names_en_rev

## Deprecated (For EN)
def get_char_info(char_dict: dict) -> (dict, dict):
    char_names = {} # char_id -> char_name
    char_names_rev = {} # char_name -> char_id

    for k, v in char_dict.items():
        char_names[k] = v["name"]
        char_names_rev[v["name"]] = k

    return char_names, char_names_rev


def get_char_rarities(char_dict: dict) -> dict:
    char_rarities = {}

    for k, v in char_dict.items():
        char_rarities[k] = v["rarity"]

    return char_rarities


## TODO: hardcoded sizes
def get_all_char_all_costs(char_dict: dict, module_dict: dict, level_dict: dict, n_operators: int) \
    -> (CostPacket, npt.NDArray):

    level_exp_map, level_lmd_map, elite_lmd_map, max_level_map = ak.get_level_info(level_dict)
    char_rarity_dict = get_char_rarities(char_dict)

    level_costs = np.empty(np.shape(level_lmd_map), dtype=ak.COST_DTYPE)
    for e in range(len(level_lmd_map)):
        for l in range(len(level_lmd_map[-1])):
            level_costs[e][l]["item_id"][0] = materials.LMD_ID
            level_costs[e][l]["count"][0]   = max(level_lmd_map[e][l], 0)
            level_costs[e][l]["item_id"][1] = materials.EXP_ID
            level_costs[e][l]["count"][1]   = max(level_exp_map[e][l], 0)

    elite_costs = np.zeros((n_operators, 2), dtype=ak.COST_DTYPE)
    skill_costs = np.zeros((n_operators, 6), dtype=ak.COST_DTYPE)
    mastery_costs = np.zeros((n_operators, 3, 3), dtype=ak.COST_DTYPE)
    module_costs = np.zeros((n_operators, 2, 3), dtype=ak.COST_DTYPE)

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


def sum_operator_cost(operators: npt.NDArray, char_names_rev: dict,
                      packet_all: CostPacket) -> npt.NDArray:
    costs = []
    for op in operators:
        op_cost = op.get_cost(char_names_rev, packet_all)
        costs.append(op_cost)

    costs = np.concatenate(costs)
    total_cost = ak.collapse_item_list(costs)
    return total_cost


