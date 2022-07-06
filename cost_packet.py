import numpy.typing as npt

class CostPacket:
    def __init__(self, char_ids: npt.NDArray,
                 elite_costs: npt.NDArray, skill_costs: npt.NDArray,
                 mastery_costs: npt.NDArray, module_costs: npt.NDArray,
                 char_rarities: npt.NDArray, level_costs: npt.NDArray,
                 max_level_map: npt.NDArray
                ):
        self.char_ids = char_ids
        self.elite_costs = elite_costs
        self.skill_costs = skill_costs
        self.mastery_costs = mastery_costs
        self.module_costs = module_costs
        self.char_rarities = char_rarities
        self.level_costs = level_costs
        self.max_level_map = max_level_map
    
    def unpack(self):
        return (self.char_ids, self.elite_costs, self.skill_costs, self.mastery_costs, self.module_costs,
                self.char_rarities, self.level_costs, self.max_level_map)