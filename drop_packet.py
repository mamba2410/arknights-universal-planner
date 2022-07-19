import numpy as np
from numpy.typing import NDArray

class DropPacket:

    def __init__(self,
            drop_matrix: NDArray,
            stage_ids: NDArray,
            item_ids: NDArray,
            san_cost: NDArray,
            ):

        self.drop_matrix = drop_matrix
        self.stage_ids = stage_ids
        self.item_ids = item_ids
        self.san_cost = san_cost

        self.n_stages = len(stage_ids)
        self.n_items = len(item_ids)


    def get_drop_chance(self, item_id: str, stage_id: str) -> float:
        item_idx = np.where(self.item_ids == item_id)[0][0]
        stage_idx = np.where(self.stage_ids == stage_id)[0][0]
        return self.drop_matrix[stage_idx, item_idx]


    def get_espd(self, item_id: str, stage_id: str) -> float:
        stage_idx = np.where(self.stage_ids == stage_id)[0][0]
        chance = self.get_drop_chance(item_id, stage_id)
        san = self.san_cost[stage_idx]
        return san/chance


    def drop_matrix_cutoff(self, threshold: float):
        #indices = np.where(self.drop_matrix < threshold)[0]
        #self.drop_matrix[indices] = 0

        for i in range(self.n_stages):
            for j in range(self.n_items):
                if self.drop_matrix[i][j] < threshold:
                    self.drop_matrix[i][j] = 0


    def filter_stages(self):
        drop_sum = np.sum(self.drop_matrix, axis=1)
        has_drops = np.where(drop_sum > 0)[0]
        has_san = np.where(self.san_cost > 0)[0]

        indices = np.intersect1d(has_drops, has_san)

        self.drop_matrix = self.drop_matrix[indices]
        self.san_cost = self.san_cost[indices]
        self.stage_ids = self.stage_ids[indices]

        self.n_stages = len(self.stage_ids)
