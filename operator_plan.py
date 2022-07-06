import numpy as np
import numpy.typing as npt
import common as ak
import json
import copy
from cost_packet import CostPacket

class OperatorPlan:
    def __init__( self,
                    name = "",
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
        self.elite_range = elite_range
        self.skill_range = skill_range
        self.mastery_range = mastery_range
        self.module_range = module_range

    
    def get_char_all_costs(self, char_names_rev: dict, cost_packet: CostPacket) -> CostPacket:
    
        char_id = char_names_rev[self.name]
        char_idx = np.where(cost_packet.char_ids == char_id)[0][0]

        e_cost = cost_packet.elite_costs[char_idx]
        s_cost = cost_packet.skill_costs[char_idx]
        m_cost = cost_packet.mastery_costs[char_idx]
        d_cost = cost_packet.module_costs[char_idx]
    
        ret = CostPacket([char_id], e_cost, s_cost, m_cost, d_cost)
        return ret

    def get_total_cost(self, cost_packet: CostPacket) -> npt.NDArray:
        
        _, e_cost, s_cost, m_cost, d_cost = cost_packet.unpack()
        
        mats_combined = np.zeros(2+6+9+2, dtype=[
            ("item_id", "U32", 4),
            ("count", "uint32", 4),
        ])
        
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
        print("\tE{} -> E{}".format(self.elite_range[0], self.elite_range[1]))
        print("\tSL{} -> SL{}".format(self.skill_range[0], self.skill_range[1]))
        for i in range(len(self.mastery_range)):
            print("\tS{}M{} -> S{}M{}".format(i+1, self.mastery_range[i][0], i+1, self.mastery_range[i][1]))
        for i in range(len(self.module_range)):
            print("\tM{}L{} -> M{}L{}".format(i+1, self.module_range[i][0], i+1, self.module_range[i][1]))
        
        print("Cost:")
        for i, c in mats:
            try:
                print("\t{}: {}".format(item_names[str(i)], c))
            except:
                print("\t{}: {}".format(i, c))
        