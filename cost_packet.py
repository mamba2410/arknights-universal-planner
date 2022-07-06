import numpy.typing as npt

class CostPacket:
    def __init__(self, char_ids: npt.NDArray,
                 elite_costs: npt.NDArray, skill_costs: npt.NDArray,
                 mastery_costs: npt.NDArray, module_costs: npt.NDArray
                ):
        self.char_ids = char_ids
        self.elite_costs = elite_costs
        self.skill_costs = skill_costs
        self.mastery_costs = mastery_costs
        self.module_costs = module_costs
    
    def unpack(self):
        return (self.char_ids, self.elite_costs, self.skill_costs, self.mastery_costs, self.module_costs)