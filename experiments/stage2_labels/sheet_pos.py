import random
from sheet import *
random.seed(3)
pos=[x for x in L if x['verdict']=='same' and gap(C,x['set_a'],x['set_b'])>60 and major(C[x['set_a']])==major(C[x['set_b']])]
random.shuffle(pos)
sheet(pos[:5],'pos_0.jpg',size=330); sheet(pos[5:10],'pos_1.jpg',size=330)
