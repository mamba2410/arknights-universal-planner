import copy
import numpy as np

ORUNDUM_PER_OP = 180
ORUNDUM_PER_PULL = 600


class PullsInventory:
    
    def __init__(self,
                 orundum = 0,
                 op = 0,
                 hh_singles = 0,
                 hh_tens = 0,
                 cost = 0,
                ):
        
        self.orundum = orundum
        self.op = op
        self.hh_singles = hh_singles
        self.hh_tens = hh_tens
        self.cost = cost
    
    
    def __add__(self, p):
        if type(p) == PullsInventory:
            c = PullsInventory()
            c.orundum = self.orundum + p.orundum
            c.op = self.op + p.op
            c.hh_singles = self.hh_singles + p.hh_singles
            c.hh_tens = self.hh_tens + p.hh_tens
            c.cost = self.cost + p.cost
            return c
    
    def __mul__(self, b):
        if type(b) == int:
            c = PullsInventory()
            c.orundum = b * self.orundum
            c.op = b * self.op
            c.hh_singles = b * self.hh_singles
            c.hh_tens = b*self.hh_tens
            c.cost = b*self.cost
            return c
        
    def __rmul__(self, n):
        return self.__mul__(n)
        
    def __sub__(self, p):
        if type(p) == PullsInventory:
            c = PullsInventory()
            c.orundum = self.orundum - p.orundum
            c.op = self.op - p.op
            c.hh_singles = self.hh_singles - p.hh_singles
            c.hh_tens = self.hh_tens - p.hh_tens
            c.cost = self.cost - p.cost
            return c
        
    def __str__(self):
        string = ""
        if self.orundum != 0:
            string += "\tOrundum: {}\n".format(self.orundum)
        if self.op != 0:
            string += "\tOP: {}\n".format(self.op)
        if self.hh_singles != 0:
            string += "\tSingles: {}\n".format(self.hh_singles)
        if self.hh_tens != 0:
            string += "\tTens: {}\n".format(self.hh_tens)
        if self.cost > 0:
            string += "\tCost: ${:.02f}\n".format(self.cost)
        
        return string
    
    def to_orundum(self, use_op=False):
        total  = self.orundum
        if use_op:
            total += self.op * ORUNDUM_PER_OP
        total += self.hh_singles * ORUNDUM_PER_PULL
        total += self.hh_tens * 10 * ORUNDUM_PER_PULL
        return total
    
    def to_pulls(self, use_op=False):
        orundum = self.to_orundum(use_op=use_op)
        pulls = orundum/ORUNDUM_PER_PULL
        return pulls
    
    def to_op(self):
        orundum = self.to_orundum()
        op = orundum/ORUNDUM_PER_OP
        return op
    
    def pulls_per_cost(self, use_op=False):
        if self.cost <= 0:
            return 0
        
        pulls = self.to_pulls(use_op=use_op)
        return pulls/self.cost
    
    def collect(self):
        c = PullsInventory()
        orundum = self.to_orundum()
        if orundum > 0:
            c.op = int(orundum/ORUNDUM_PER_OP)
            c.orundum = orundum % ORUNDUM_PER_OP
            c.cost = self.cost
        else:
            c.orundum = orundum
            c.cost = self.cost
        
        return c
    
    def spend_op(self, n=0):
        if n == 0:
            n = self.op
            
        if n > self.op:
            raise ValueException("Don't own enough OP to spend")
        
        self.op -= n
        self.orundum += n*ORUNDUM_PER_OP
        
    
    ## For now, assume target_ is purely hh_singles
    ## For a mixed currency, just subtract them
    def try_spend(self, target_, use_op = True):
        target = copy.deepcopy(target_)
        result = copy.deepcopy(self)
        
        ## While we have 10x hh remaining and we have more than 10 pulls to go
        while target.to_pulls() >= 10 and result.hh_tens > 0:
            target.hh_tens -= 1
            result.hh_tens -= 1
        
        ## While we have pulls left and own single hh tickets
        while target.to_pulls() >= 1 and result.hh_singles > 0:
            target.hh_singles -= 1
            result.hh_singles -= 1
            
        ## While we have pulls left and own enough orundum for a pull
        while target.to_pulls() >= 1 and result.orundum >= ORUNDUM_PER_PULL:
            target.orundum -= ORUNDUM_PER_PULL
            result.orundum -= ORUNDUM_PER_PULL
            
        ## If we're allowed to spend OP, need to do more pulls and own OP
        while use_op and target.to_pulls() >= 1 and result.op > 0:
            ## Convert one OP to orundum
            result.spend_op(1)
            ## Try to pull with orundum
            while target.to_pulls() >= 1 and result.orundum >= ORUNDUM_PER_PULL:
                target.orundum -= ORUNDUM_PER_PULL
                result.orundum -= ORUNDUM_PER_PULL
        
        
        ## Exhausted resources, finished pulling
        
        ## Sanity check, if we could pull and still need to pull, warn us (shouldn't happen)
        if result.to_pulls() >= 1 and target.to_pulls() >= 1 and use_op:
            print("Error: Something went wrong with pulling")
        
        ## If we ran out of resources, but still need to pull
        if target.to_pulls() >= 1:
            ## Tell us how much more orundum we need in order to
            ## do the pulls
            result.orundum -= int(target.to_pulls() * ORUNDUM_PER_PULL)
        
        return result


## Recurring
daily = PullsInventory(orundum = 100) ## Missions
daily_with_pass = PullsInventory(orundum = 200 + 100) ## Pass + missions
weekly = PullsInventory(orundum = 500 + 1800) ## Missions + anni
weekly_with_pass = weekly + daily_with_pass*7
green_shop = PullsInventory(orundum = 600, hh_singles = 4) ## TODO: Check
gold_shop = PullsInventory(hh_singles = 1 + 2 + 5, hh_tens = 1 + 2) ## Gold cert shop pulls
calendar_login = PullsInventory(hh_singles=1)

## Buy things
pack_monthly = PullsInventory(op = 42, hh_tens = 1, cost = 25.99)
card_monthly_op_only = PullsInventory(op = 6, cost = 4.99)
card_monthly = PullsInventory(orundum = 30*200, op = 6, cost = 4.99)
op_t1_first = PullsInventory(op = 1 + 2, cost = 0.99)
op_t2_first = PullsInventory(op = 6 + 6, cost = 4.99)
op_t3_first = PullsInventory(op = 20 + 20, cost = 14.99)
op_t4_first = PullsInventory(op = 40 + 40, cost = 29.99)
op_t5_first = PullsInventory(op = 66 + 66, cost = 49.99)
op_t6_first = PullsInventory(op = 130 + 130, cost = 99.99)
op_t1 = PullsInventory(op = 1, cost = 0.99)
op_t2 = PullsInventory(op = 6 + 1, cost = 4.99)
op_t3 = PullsInventory(op = 20 + 4, cost = 14.99)
op_t4 = PullsInventory(op = 40 + 10, cost = 29.99)
op_t5 = PullsInventory(op = 66 + 24, cost = 49.99)
op_t6 = PullsInventory(op = 130 + 55, cost = 99.99)
duck_lord = PullsInventory(op = 13, cost = 6.99)
insta_e2 = PullsInventory(op = 37, cost = 19.99)
selector_pack = PullsInventory(hh_tens = 1, cost = 29.99)
party_pack = PullsInventory(op = 24, hh_tens = 2, cost = 29.99) ## price?
smol_pack = PullsInventory(orundum = 4500, cost = 9.99) ## price?
big_pack = PullsInventory(op = 90, hh_tens = 1, cost = 49.99) ## price?
new_year_pack = PullsInventory(op=51, hh_tens=1, orundum=2100, cost=29.99) ## price?


## Misc static
free_banner = PullsInventory(hh_singles = 14, hh_tens = 1) ## Limited banner free pulls
mining_permit = PullsInventory(orundum = 14*(1800+300)/2) ## Very crude estimate
skin_15 = PullsInventory(op = 15)
skin_18 = PullsInventory(op = 18)
skin_21 = PullsInventory(op = 21)
skin_24 = PullsInventory(op = 24)


def calc_pack_efficiencies(use_op=True):
    # Easy way of assigning names
    pack_list = [
        (pack_monthly, "Monthly hh pack"),
        (card_monthly, "Monthly card"),
        (card_monthly_op_only, "Monthly card (op only)"),
        (op_t1_first, "1 OP pack (first)"),
        (op_t2_first, "6 OP pack (first)"),
        (op_t3_first, "20 OP pack (first)"),
        (op_t4_first, "40 OP pack (first)"),
        (op_t5_first, "66 OP pack (first)"),
        (op_t6_first, "130 OP pack (first)"),
        (op_t1, "1 OP pack"),
        (op_t2, "6 OP pack"),
        (op_t3, "20 OP pack"),
        (op_t4, "40 OP pack"),
        (op_t5, "66 OP pack"),
        (op_t6, "130 OP pack"),
        (duck_lord, "Duck lord's purse"),
        (insta_e2, "Instant E2 pack"),
        (selector_pack, "6* selector pack"),
        (party_pack, "RI party pack"),
        (smol_pack, "Anniversary update pack"),
        (big_pack, "Anniversary carnival pack"),
        (new_year_pack, "RI New Year pack"),
    ]
    n_packs = len(pack_list)
    
    ## Table of stats for paid packs
    paid_packs = np.empty((n_packs), dtype=[
        ("name", np.unicode_, 32),
        ("eff", "float64"), ## Efficiency (pulls/$)
        ("fpulls", "float64"), ## number of pulls (as decimal)
        ("pulls", "int64"), ## number of pulls (rounded down to nearest integer)
        ("cost", "float64")
    ])
    
    ## Populate stats table
    for i, (p, name) in enumerate(pack_list):
        paid_packs['name'][i] = name
        paid_packs['cost'][i] = p.cost
        paid_packs['eff'][i] = p.pulls_per_cost(use_op=use_op)
        n_pulls = p.to_pulls(use_op=use_op)
        paid_packs['pulls'][i] = int(n_pulls)
        paid_packs['fpulls'][i] = n_pulls
    
    ## Order in decreasing efficiency
    order_eff = np.flip(np.argsort(paid_packs['eff']))
    paid_packs = paid_packs[order_eff]

    print("Pack pulls-per-cost efficiency (higher is better) (doesn't factor in other items)")
    for i, (name, eff, fpulls, pulls, cost) in enumerate(paid_packs):
        print("\t{:02}: {}: {:.4} p/$ ({} pulls)".format(i+1, name, eff, pulls))


