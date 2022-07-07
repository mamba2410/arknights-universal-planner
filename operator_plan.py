import numpy as np
import numpy.typing as npt
import common as ak
import json
import copy
from cost_packet import CostPacket

class OperatorPlan:
    def __init__( self,
                    name = "",
                    level_range = (1, 1),
                    elite_range = (0, 0),
                    skill_range = (1, 1),
                    mastery_range = [
                        (0, 0),
                        (0, 0),
                        (0, 0)
                    ],
                    module_range = [
                        (0, 0)
                    ]
                ):
        
        self.name = name
        
        level_range_invalid = level_range[0] < 1 or level_range[0] > 90 or \
                              level_range[1] < 1 or level_range[1] > 90
        
        elite_range_invalid = elite_range[0] < 0 or elite_range[0] > 2 or \
                              elite_range[1] < 0 or elite_range[1] > 2 or \
                              elite_range[0] > elite_range[1]
        
        skill_range_invalid = skill_range[0] < 1 or skill_range[0] > 7 or \
                              skill_range[1] < 1 or skill_range[1] > 7 or \
                              skill_range[0] > skill_range[1]
        
        if level_range_invalid:
            raise ValueError("Level range invalid: {} -> {}".format(level_range[0], level_range[1]))
            
        if elite_range_invalid:
            raise ValueError("Elite range invalid: {} -> {}".format(elite_range[0], elite_range[1]))
            
        if skill_range_invalid:
            raise ValueError("Skill range invalid: {} -> {}".format(skill_range[0], skill_range[1]))
            
        self.elite_range = elite_range
        self.skill_range = skill_range
        self.level_range = level_range
        
        for r in mastery_range:
            range_invalid =   r[0] < 0 or r[0] > 3 or \
                              r[1] < 0 or r[1] > 3 or \
                              r[0] > r[1]
            if range_invalid:
                raise ValueError("Mastery range invalid: {} -> {}".format(r[0], r[1]))
            
        self.mastery_range = mastery_range
                                 
        for r in module_range:
            range_invalid =   r[0] < 0 or r[0] > 3 or \
                              r[1] < 0 or r[1] > 3 or \
                              r[0] > r[1]
        if range_invalid:
            raise ValueError("Module range invalid: {} -> {}".format(r[0], r[1]))
                                 
        self.module_range = module_range

    
    def get_char_all_costs(self, char_names_rev: dict, cost_packet: CostPacket) -> CostPacket:
    
        char_id = char_names_rev[self.name]
        char_idx = np.where(cost_packet.char_ids == char_id)[0][0]
        rarity = cost_packet.char_rarities[char_id]

        e_cost = cost_packet.elite_costs[char_idx]
        s_cost = cost_packet.skill_costs[char_idx]
        m_cost = cost_packet.mastery_costs[char_idx]
        d_cost = cost_packet.module_costs[char_idx]
        max_level = cost_packet.max_level_map[rarity]
    
        ret = CostPacket([char_id], e_cost, s_cost, m_cost, d_cost,
                         cost_packet.char_rarities, cost_packet.level_costs, max_level)
        return ret

    ## TODO: Error checking in here
    def get_total_cost(self, cost_packet: CostPacket) -> npt.NDArray:
        
        _, e_cost, s_cost, m_cost, d_cost, _, l_cost, max_level = cost_packet.unpack()
        
        mats_combined = np.zeros(2+6+9+2+3, dtype=ak.COST_DTYPE)
        
        j = 0
        for v in e_cost[self.elite_range[0] : self.elite_range[1]]:
            mats_combined[j] = v
            j += 1

        for v in s_cost[self.skill_range[0]-1 : self.skill_range[1]-1]:
            mats_combined[j] = v
            j += 1

        for i in range(len(self.mastery_range)):
            for v in m_cost[i][self.mastery_range[i][0] : self.mastery_range[i][1]]:
                mats_combined[j] = v
                j += 1
                
        for i in range(len(self.module_range)):
            for v in d_cost[i][self.module_range[i][0] : self.module_range[i][1]]:
                mats_combined[j] = v
                j += 1
             
        level_start = self.level_range[0]
        for i in range(self.elite_range[0], self.elite_range[1]+1):
            if self.elite_range[1] > i:
                level_end = max_level[i]
            else:
                level_end = self.level_range[1]
            mats_combined[j]["item_id"][0] = ak.LMD_ID
            mats_combined[j]["count"][0] = np.sum(l_cost[i][level_start-1:level_end-1]["count"], axis=0)[0]
            mats_combined[j]["item_id"][1] = ak.EXP_ID
            mats_combined[j]["count"][1] = np.sum(l_cost[i][level_start-1:level_end-1]["count"], axis=0)[1]
            j += 1

        mats_combined = ak.sum_skill_slice(mats_combined)
        return mats_combined
        
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__)
    
    def as_operator(self, name: str):
        new = copy.deepcopy(self)
        new.name = name
        return new
    
    def get_cost(self, char_names_rev: dict, cost_packet: CostPacket) -> npt.NDArray:
        char_cost_packet = self.get_char_all_costs(char_names_rev, cost_packet)
        char_cost = self.get_total_cost(char_cost_packet)
        return char_cost
    
    def pretty_print(self, item_names: dict, mats: npt.NDArray) -> None:
        print("{}:".format(self.name))
        if self.elite_range[1] > self.elite_range[0]:
            print("\tE{} -> E{}".format(self.elite_range[0], self.elite_range[1]))
            
        if self.skill_range[1] > self.skill_range[0]:
            print("\tSL{} -> SL{}".format(self.skill_range[0], self.skill_range[1]))
            
        for i in range(len(self.mastery_range)):
            if self.mastery_range[i][1] > self.mastery_range[i][0]:
                print("\tS{}M{} -> S{}M{}".format(i+1, self.mastery_range[i][0],
                                                  i+1, self.mastery_range[i][1]))
        for i in range(len(self.module_range)):
            if self.module_range[i][1] > self.module_range[i][0]:
                print("\tM{}L{} -> M{}L{}".format(i+1, self.module_range[i][0], i+1, self.module_range[i][1]))
        
        print("Cost:")
        for i, c in mats:
            try:
                print("\t{}: {}".format(item_names[str(i)], c))
            except:
                print("\t{}: {}".format(i, c))
        