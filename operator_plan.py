import numpy as np
import numpy.typing as npt
import common as ak

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

    
    def get_total_cost(self, e_cost: npt.NDArray, s_cost: npt.NDArray,
                       m_cost: npt.NDArray, d_cost: npt.NDArray) -> npt.NDArray:
        
        mats_combined = np.zeros(2+6+9+2, dtype=[
            ("item_id", "uint32", 3),
            ("count", "uint32", 3),
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

        mats_combined = ak.sum_skill_slice(mats_combined)
        return mats_combined