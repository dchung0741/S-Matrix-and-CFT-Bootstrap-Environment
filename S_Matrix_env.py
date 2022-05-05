from itertools import zip_longest
from math import sqrt
from numpy import array
from numpy.linalg import norm
from numpy.random import rand
from sympy.functions.combinatorial.factorials import binomial
import pandas as pd


class S_Matrix:

    Computed_Coeff = {}

    def __init__(self, s_spec_dict: dict, t_spec_dict: dict, interested_constraints: list, u_channel = True, lowest_s_power = 2):

        self.s_spec_dict = s_spec_dict
        self.t_spec_dict = t_spec_dict

        self.s_is_t = True if self.s_spec_dict == self.t_spec_dict else False
        self.u_ch = u_channel

        self.constraints = interested_constraints

        self.s_constraints = []
        self.t_constraints = []
        for k in self.constraints:
            
            self.s_constraints += [ (k, q) for q in range(k//2 + 1, k - lowest_s_power + 1) ]
            self.t_constraints += [ (k, k - q) for q in range(k//2 + 1, k - lowest_s_power + 1) ]
        

        self.s_Spin_Dict = {}
        self.s_Spec_List = []

        self.t_Spin_Dict = {}
        self.t_Spec_List = []

        for spec_s, spec_t in zip_longest(self.s_spec_dict, self.t_spec_dict):

            if spec_s is not None:
                
                self.s_Spin_Dict[spec_s] = len(s_spec_dict[spec_s])
                
                for i in s_spec_dict[spec_s]:
                    self.s_Spec_List += [i['m'], i['cl']]
                    

            if spec_t is not None:

                self.t_Spin_Dict[spec_t] = len(t_spec_dict[spec_t])

                for i in t_spec_dict[spec_t]:
                    self.t_Spec_List += [i['m'], i['cl']]
                    
        

        if self.s_is_t is True:

            self.Full_Spec = array(self.s_Spec_List)
        
        else:
            self.Full_Spec = array(self.s_Spec_List + self.t_Spec_List)
        
        self.dim = len(self.Full_Spec)


    @classmethod
    def Spectrum_Gen(cls, 
                    s_spin_num: dict, s_mass_spec: list, s_cl_spec: list, 
                    t_spin_num: dict, t_mass_spec: list, t_cl_spec: list, 
                    interested_constraint: list,
                    u_channel: bool,
                    lowest_s_power = 2):
        
        s_spec_dict = {}
        s_cnt = 0

        for spin in s_spin_num:
            s_spec_dict[spin] = []
            
            for _ in range(s_spin_num[spin]):
                s_spec_dict[spin].append({'m': s_mass_spec[s_cnt], 'cl': s_cl_spec[s_cnt]})
                s_cnt += 1

        if s_mass_spec == t_mass_spec and s_cl_spec == t_cl_spec:
            t_spec_dict = s_spec_dict
        
        else:
            t_spec_dict = {}
            t_cnt = 0

            for spin in t_spin_num:
                t_spec_dict[spin] = []
                
                for _ in range(t_spin_num[spin]):
                    t_spec_dict[spin].append({'m': t_mass_spec[s_cnt], 'cl': t_cl_spec[s_cnt]})
                    t_cnt += 1
        
        return S_Matrix(s_spec_dict = s_spec_dict, t_spec_dict = t_spec_dict, interested_constraints = interested_constraint, u_channel = u_channel, lowest_s_power = lowest_s_power)



    def Spec_List_to_Spec_Dict(self):
        
        s_spec_dict = {}
        s_spec_list = [ {'m': self.s_Spec_List[i], 'cl': self.s_Spec_List[i + 1]} for i in range(0, len(self.s_Spec_List), 2) ]
        
        for spin in self.s_Spin_Dict:
            
            s_spec_dict[spin] = s_spec_list[0: self.s_Spin_Dict[spin]]
            del s_spec_list[0: self.s_Spin_Dict[spin]]  


        if self.s_is_t:
            t_spec_dict = s_spec_dict.copy()
        
        else:
            t_spec_dict = {}
            t_spec_list = [ {'m': self.t_Spec_List[i], 'cl': self.t_Spec_List[i + 1]} for i in range(0, len(self.t_Spec_List), 2) ]
            
            for spin in self.t_Spin_Dict:
                
                t_spec_dict[spin] = t_spec_list[0: self.t_Spin_Dict[spin]]
                del t_spec_list[0: self.t_Spin_Dict[spin]]

        return s_spec_dict, t_spec_dict
    
    
    def data_frame_form(self):

        String_Spec_1_df_dict = {}

        for spin in self.s_spec_dict:

            String_Spec_1_df_dict[spin] = {}

            for i in self.s_spec_dict[spin]:

                String_Spec_1_df_dict[spin]['m^2 = {}'.format(round(i['m']**2))] = i['cl']


        return pd.DataFrame(String_Spec_1_df_dict).sort_index().transpose()



###############################################################################################################################################
    def Update_Spec_with_new_spec(self, new_spec):

        self.Full_Spec = new_spec
        self.s_Spec_List = self.Full_Spec[:len(self.s_Spec_List)]
        self.t_Spec_List = self.Full_Spec[len(self.t_Spec_List):]

        new_s_spec, new_t_spec = self.Spec_List_to_Spec_Dict()

        self.__init__(new_s_spec, new_t_spec, self.constraints, self.u_ch)
        
        

    def a_single_coeff(self, p, q, l):
        
        def Delta(n1, n2):

            if n1 == n2:
                return 1

            else:
                return 0

        if self.u_ch is True:

            if (p ,q, l) in self.Computed_Coeff:

                return self.Computed_Coeff[(p, q, l)]
                
            else:

                
                tmp = [ ( pow(-1, p + r + 1) * int(binomial(p + r, r)) - Delta(r, 0) ) * int(binomial(l, q - r)) * int(binomial(l + q - r, l)) for r in range(0, q + 1) ]
                self.Computed_Coeff.update({(p, q, l): sum(tmp)})
                    
                return self.Computed_Coeff[(p, q, l)]
        
        else:
            
            if (p ,q, l) in self.Computed_Coeff:

                return self.Computed_Coeff[(p, q, l)]
                
            else:

                tmp =  - int(binomial(l, q)) * int(binomial(l + q, l)) 
                self.Computed_Coeff.update({(p, q, l): tmp})
                    
                return self.Computed_Coeff[(p, q, l)]


    def s_kq_Coeff(self, k, q):

        kq_Coeff = 0
        
        for spin_spec in self.s_spec_dict:

            for data in self.s_spec_dict[spin_spec]:                
                kq_Coeff += data['cl'] * self.a_single_coeff(k - q, q, spin_spec) * pow(data['m'], -2*(k + 1))
                
        return kq_Coeff
    
    
    def t_kq_Coeff(self, k, q):
        
        kq_Coeff = 0
        
        for spin_spec in self.t_spec_dict:

            for data in self.t_spec_dict[spin_spec]:                
                kq_Coeff += data['cl'] * self.a_single_coeff(k - q, q, spin_spec) * pow(data['m'], -2*(k + 1))

        return kq_Coeff


    def s_Vector(self):
        
        s_Vec = {}
        
        for constraint in self.s_constraints:

            p = constraint[0] - constraint[1]
            q = constraint[1]
            s_Vec[f's^{p} t^{q}'] = (self.s_kq_Coeff(constraint[0], constraint[1]))
            
        return s_Vec

    
    
    def t_Vector(self):
        
        t_Vec = {}
        
        for constraint in self.t_constraints:

            p = constraint[0] - constraint[1]
            q = constraint[1]
            t_Vec[f's^{q} t^{p}'] = (self.t_kq_Coeff(constraint[0], constraint[1]))
        
        return t_Vec

    

    def Crossing(self):

        Constraint_Vec = []

        s_Channel = self.s_Vector()
        t_Channel = self.t_Vector()
        s_set = set(s_Channel)
        t_set = set(t_Channel)
        Intersection = s_set.intersection(t_set)

        for el in Intersection:
            Constraint_Vec.append(s_Channel[el] - t_Channel[el])
        
        return array(Constraint_Vec)
    

    def Reward(self, punish_0 = 0, punish_wrong_sign = 0):

        crossing_vec = self.Crossing()

        if punish_wrong_sign != 0:
            abs_vec = array(list(map(abs, crossing_vec)))
            sgn_vec = crossing_vec - abs_vec

            return - norm(crossing_vec) - punish_0 * list(self.Full_Spec).count(0) - punish_wrong_sign * norm(sgn_vec)
        
        else:
            return - norm(crossing_vec) - punish_0 * list(self.Full_Spec).count(0) 


if __name__ == '__main__':
    
    best_spec = [
       1.        , 0.40131994, 0.84689925, 0.5417466 , 0.81143928,
       0.20064561, 0.16144549, 0.463935  , 0.49447583, 0.06316576,
       0.0093239 , 0.27085383, 0.01036254, 0.02546961, 0.00495387
       ]

    mass_spec = [
        sqrt(1), sqrt(2), sqrt(3), sqrt(4), sqrt(5),
        sqrt(2), sqrt(3), sqrt(4), sqrt(5),
        sqrt(3), sqrt(4), sqrt(5),
        sqrt(4), sqrt(5),
        sqrt(5),
     ]

    s_spin_num = {0: 5, 2: 4, 4: 3, 6: 2, 8: 1}

    constrained_k = [5 + i for i in range(0, 6)]

    spec = S_Matrix.Spectrum_Gen(
        s_spin_num = s_spin_num, s_mass_spec = mass_spec, s_cl_spec = best_spec,
        t_spin_num = s_spin_num, t_mass_spec = mass_spec, t_cl_spec = best_spec,
        interested_constraint = constrained_k,
        u_channel = True
        )
    
    print(spec.s_spec_dict)
    print(spec.Crossing())
    print(spec.Reward())