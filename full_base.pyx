#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
cimport numpy as np
import itertools
import logging
import hashlib
from JMP import NERD_SCALE
from JMP.full.forward import Forward


class FullBase(Forward):
    def __init__(self, a, ctt_e, health, parameter, tax=0., limit=5, concurrent=True):
        Forward.__init__(self, limit)

        transed_p = tuple(np.around(np.asarray(parameter) * NERD_SCALE, decimals=4))
        self._gamma, self._theta, self._phi, \
        self._kappa1, self._kappa2, self._utility_treated, \
        self._discount_factor, self._joe1, self._lamda, self._z1, self._z2,\
        self._gov_health, self._choice_c_edu_1yr_theta, self._comp, self._e_scale_v1, self._e_scale_v2, self._c_c = transed_p # 11 params and 3 confact

        tax = np.around(tax, decimals=3)
        self._concurrent = concurrent
        self._a_1yr = a * (1 if a < 3500 else 1 - tax)
        self._ctt_e = ctt_e
        self._health = health
        self._tax = tax
        assert ctt_e in (0, 1) # grandpa first make decision
        assert self._comp in (2, 3) # t of compulsory education
        self._init_cache()
        self._init_para()
        self._UID = '{0}, {1}'.format((a, ctt_e, health, tax), self._list_format2(transed_p)) # cache for params
        logging.info('Input Key       : \t%s', parameter)
        logging.info('Transed Key     : \t%s', self._UID)

        self._output_dir, self._check_sum = self._check_hash('full_base.pyx')
        self._output_dir = self._output_dir / self._check_sum
        self._output_dir.mkdir(exist_ok=True) # cache for current file version

        self._key = hashlib.md5(self._UID.encode('utf-8')).hexdigest()
        logging.info('Key Hash        : \t%s', self._key)
        self._output_file = self._output_dir / self._key # cache for params

    def _init_cache(self):
        self._make_choice_cache = {}
        self._utility_fyr_cache = {}
        self._utility_final_cache = {}
        self._utility_final_e_cache = {}
        self._ue_cache = {}
        self._ua_cache = {}
        self._u1_cache = {}
        self._u3_cache = {}
        self._isoelastic_cache = {}
        self._uhealth_cache = {}

    def _init_para(self):
        # length of each time period, in years. t=0: 6 years, t=1: 6 years, t=2: 3 years, t=3: 3 years, t=4: Terminal condition.
        self._choice_length_t = (6, 6, 3, 3, 2) # Ling is sure
        assert len(self._choice_length_t) == self._choice_len # Ling is sure

        # new edu length
        self._state_edu_length_t = (0, 0, 6, 9, 12, 16)  # [0] + list(np.add.accumulate(self._length_t).tolist()) # Ling is sure
        assert len(self._state_edu_length_t) == self._state_len # Ling is sure

        # annual wage in the urban area
        self._w_1yr = (23226 + 32083) * (1 - self._tax) # Ling is sure
        # self._w_1yr = 25297, 24366, 25107, 24773, 22100
        # assert len(self._w_1yr) == self._choice_len # Ling is sure

        self._sa_choice = (0, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000,
                           36000, 40000, 44000, 48000, 52000, 56000, 60000, 64000,
                           68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000,
                           100000, 104000, 108000, 112000, 116000, 120000, 124000,
                           128000, 132000, 136000, 140000, 144000, 148000, 152000,
                           156000, 168000, 182000, 196000, 210000, 224000, 238000,
                           252000, 266000, 280000, 294000, 300000
                           )
        self._se_choice = (0,
                           2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000,
                           20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000,
                           38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000,
                           56000, 58000, 60000, 65000, 70000, 75000, 80000, 85000, 90000,
                           95000, 100000, 105000, 110000
                           )
        self._c_grid = int((self._se_choice[1] - self._se_choice[0]) / 3) # Ling is sure

        # private cost of the education, indexed by [t][m_c]
        self._choice_c_edu_1yr = [[275 * 12, 373 * 12], [969, 3833], [1834, 4153], [4390, 7078], [15000] * 2]
        if self._comp == 3:
            self._choice_c_edu_1yr[3][0] = 3125
        for idx in self._choice_c_edu_1yr: # Eq 14
            sub = idx[1] - idx[0]
            idx[1] = idx[0] + self._choice_c_edu_1yr_theta * sub
        assert len(self._choice_c_edu_1yr) == self._choice_len # Ling is sure

        # borrowing limit: should be 0 or negative. Represents the maximum amount of debt.
        self._loan = 0 # Ling is sure

        # lower bound for % of medical cost paid by adult in contract to "commit to the contract".
        self._contract_medical = 0.8 # Ling is sure
        self._c_new_death = 8209 # Ling is sure

        self._comp = {1, 2, self._comp} # Ling is sure

        # total medical expenditure by length of sickness
        # what I want now is cost = 3305RMB/yr, and after the elderly's death, there's an extra 8209RMB one time cost.
        self._c_h_1yr = 3305  # + 8209 / 3 # Ling is sure
        table_c_h_total_1yr = (np.asarray([self._c_h_1yr] * self._choice_len, dtype=np.float64) * (1 - self._gov_health)).tolist() # Eq 13

        # out of pocket medical expenditure by length of sickness
        self._table_c_h_private_1yr = [int(i / 10) * 10 for i in table_c_h_total_1yr] # Ling is sure

        # amount of leisure time, depend on location, taking care of child, health:
        #   [[rural no child, rural with child],[urban no child, urban with child]].
        #   Endowment: 12 hours per day. The leisure is hours per week.
        leisure = np.asarray([[34, 22], [18, 6 + 12 * self._c_c]], dtype=np.float64) # Ling is sure
        self._table_leisure = leisure.tolist() # Ling is sure
        self._babysit = 8

        length_t = np.column_stack([self._choice_length_t] * 2) # Ling is sure
        length_t = np.asarray(length_t, dtype=np.float64) / 3. # Ling is sure

        # probability of getting sick at the end of this period, by [t]
        pr_s_base_3yr = [0.15] * self._choice_len # Ling is sure
        pr_s_base_3yr = np.asarray(pr_s_base_3yr, dtype=np.float64) # Ling is sure
        self._table_pr_s_fyr =  1 - np.power(1 - pr_s_base_3yr, length_t[:,0]) # Ling is sure

        # probability of dying at the end of this period, by [0 if no migration when sick, 1 if migrated when sick]
        pr_d_base_3yr = 0.64683343 # Ling is sure
        rho_migrate_sick = 1.67111
        pr_d_base_3yr = [[pr_d_base_3yr, min(1., pr_d_base_3yr * rho_migrate_sick)]] * self._choice_len # Ling is sure
        pr_d_base_3yr = np.asarray(pr_d_base_3yr, dtype=np.float64) # Ling is sure
        self._table_pr_d_fyr = 1 - np.power(1 - pr_d_base_3yr, length_t) # Ling is sure

        # Pr(not in school at the beginning of the next period | in school now, and the adult pays c_edu), by [t][ma==mc, ma!=mc]
        # 落榜=0.7, confirmed from data
        self._table_pr_e_fyr = [[0., 0., 0.], [0.0002, 0.0007, 0.], [0.0471, 0.0957, 0.], [0.0474, 0.0519, 0.], [0.1455, 0.1389, 0.7]] # Ling is sure
        assert len(self._table_pr_e_fyr) == self._choice_len # Ling is sure

        # annual interest rate on savings
        ir_1yr_base = 0.0275 # Ling is sure
        self._ir_1yr = np.power(1 + ir_1yr_base, self._choice_length_t) # Ling is sure

        # self._e_scale = 0.5 # Ling is sure
        self._w_scale = 4.14 # Ling is sure
        self._disutility_sick = 0.814393 # Ling is sure
        self._old_c = 2000. + self._contract_medical * self._c_h_1yr * (1 - self._gov_health) # Eq 11

        disc_fa = np.asarray(list(range(21)))
        self._disc_fa2 = np.power(self._discount_factor, disc_fa)
        self._disc_fa = (1 - self._disc_fa2) / (1 - self._discount_factor)
    
    def _translate(self, ret_next, t, state):
        v_a, v_e, future, choice = ret_next
        if future is None:
            state = *state[:3], -2, *state[4:]
            return (((v_a, v_e, t, state, choice, np.float64(1.0)),),)
        ret = []
        for prob, params in future:
            ret_future, _ = self._cache[params]
            for idx in self._translate(ret_future, params[0], params[1]):
                idx_0 = idx[0]
                updated_pr = prob * idx_0[5]
                idx_0 = *idx_0[:5], updated_pr
                ret.append(((v_a, v_e, t, state, choice, updated_pr), idx_0, *idx[1:]))
        return tuple(ret)

    # return [[(v_a, v_e, t, s, choice, p)]]]
    def _recursion(self, t, state: tuple, new_death=False):
        """
        :type t: int
        :type state: tuple[float]
        :type new_death: bool
        :rtype: tuple[list[list[tuple]], int]
        """
        self._acc_count_np[t] += 1
        cache_key = t, state, new_death
        ret = self._cache.get(cache_key)
        if ret is not None:
            return ret

        savings_a, edu, savings_e, health, ctt_a, migrate_sick, fame, loc = state # Ling is sure
        if self._limit == t:
            if health != -2: # Ling is sure
                health = -2
                new_death = True
                bequest = (1 - migrate_sick) * savings_e - self._c_new_death # new death to savings_a
                # bequest = min(0, bequest)
                savings_a = max(0, savings_a + bequest)
            assert health == -2
            ret = (self._utility_final_a(savings_a, edu, fame), # Ling is sure
                        self._utility_final_e(edu, new_death),
                        None,
                        None)
            self._cache[cache_key] = ret, 1
            return ret, 1

        choice = self._make_choice(t, edu == t, savings_a, savings_e, health, ctt_a, loc)
        self._add(len(choice), t * 2)

        all_futures = {}
        my_future_count = 0
        for cur_choice in choice:
            tmp_v_a, tmp_v_e, ret_next, \
            m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level, \
            _, _, _, f_count = self._test_one_choice(
                *cur_choice, t, edu, migrate_sick, health, ctt_a, new_death, fame)
            my_future_count += f_count

            key = m_a, m_c, c_a_o, c_edu, transfer # group key
            if key not in all_futures or tmp_v_e > all_futures[key][1]: # grandpa second make choice inside each group
                all_futures[key] = (tmp_v_a, tmp_v_e, ret_next, \
                                   (m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level)) # Eq 18

        max_v = (float('-inf'),)
        for save_state in all_futures.values():
            if save_state[0] > max_v[0]: # parent first make choice of groups
                max_v = save_state # Eq 16

        try:
            assert len(max_v) == 4
        except AssertionError:
            logging.critical('all_futures %s', [all_futures, t, state, self._UID])
            exit(1)

        self._cache[cache_key] = max_v, my_future_count
        return max_v, my_future_count

    def _test_one_choice(self,
                         m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level,
                         t, edu, migrate_sick, health, ctt_a, new_death, fame):
        # migrate_sick is 1 if the adult migrated when the elderly is sick.
        if migrate_sick == 0 and health >= 0 and m_a == 1: # Eq 8
            migrate_sick = 1

        penalty = self._penalty(ctt_a, health, m_a, m_c, transfer, fame) # Eq 12

        task = self._judge_edu(m_a, m_c, c_edu, c_h,
                               savings_a, savings_e, t, edu, migrate_sick, health, ctt_a, penalty)
        self._add(len(task), t * 2 + 1)

        all_pr = 0.
        avg_a = 0.
        avg_e = 0.
        ret_next = []
        my_future_count = 0
        for prob, state, new_death in task:
            if prob <= 1e-15:
                continue
            general_ret, f_count = self._recursion(t + 1, state, new_death=new_death)
            v_a, v_e, _, _ = general_ret
            ret_next.append((prob, (t + 1, state, new_death)))
            all_pr += prob # Ling is sure
            avg_a += prob * v_a # Ling is sure
            avg_e += prob * v_e # Ling is sure
            my_future_count += f_count
        assert len(ret_next) > 0
        # assert 1.0e-8 >= abs(all_pr - 1.0)

        if health != -2: # Ling is sure
            _ue = self._ue(health, m_a, m_c, c_e_o, c_edu, c_h, t)
            tmp_v_e = self._utility_fyr(_ue, avg_e, self._choice_length_t[t])
        else: # Ling is sure
            # _ue = 0
            tmp_v_e = self._utility_final_e(edu, new_death)

        _ua = self._ua(c_a_o, m_a, m_c, c_edu, t, penalty, health >= 0, ctt_a) # Eq 2
        tmp_v_a = self._utility_fyr(_ua, avg_a, self._choice_length_t[t]) # Ling is sure

        return tmp_v_a, tmp_v_e, ret_next, \
               m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level, \
               self._acc_count_np, self._appro_amt, len(self._cache), my_future_count

    def _judge_edu(self,
                   m_a, m_c, c_edu, c_h,
                   savings_a, savings_e, t, edu, migrate_sick, health, ctt_a, penalty):
        if c_edu > 0 or t == 0: # kindergarten
            assert t == edu # last enroll == 1
            pr_e = self._table_pr_e_fyr[t][m_a - m_c] # depend on relation and Eq 9
            return (
                *self._judge_health(
                    m_a, m_c, c_h,
                    savings_a, savings_e, t, edu + 1, migrate_sick, health, ctt_a, penalty,
                                             (1.0 - pr_e) * (1 - self._table_pr_e_fyr[t][2])), # enroll and updated
                *self._judge_health(
                    m_a, m_c, c_h,
                    savings_a, savings_e, t, edu, migrate_sick, health, ctt_a, penalty,
                    pr_e), # enroll but not update
                *self._judge_health(
                    m_a, m_c, c_h,
                    savings_a + c_edu, savings_e, t, edu, migrate_sick, health, ctt_a, penalty,
                    (1.0 - pr_e) * self._table_pr_e_fyr[t][2]) # enroll but rejected
            )
        else: # last enroll == 0
            return self._judge_health(
                m_a, m_c, c_h,
                savings_a, savings_e, t, edu, migrate_sick, health, ctt_a, penalty,
                1.) # not enroll

    def _judge_health(self,
                      m_a, m_c, c_h,
                      savings_a, savings_e, t, edu, migrate_sick, health, ctt_a, penalty,
                      pr_e: float):
        if abs(pr_e) < 1.0e-15:
            return []

        if t == self._limit - 1:
            pr_d = 1.0 # Ling is sure
            pr_s = 0.0
        elif health == -1:
            pr_d = 0.0
            pr_s = self._table_pr_s_fyr[t] # Ling is sure
        else:
            pr_d = self._table_pr_d_fyr[t, migrate_sick] # Eq 8
            pr_s = 0.0

        if health == -1:
            return (
                (pr_e * pr_s, (savings_a, edu, savings_e, t + 1, max(ctt_a, m_a - m_c), 0, penalty, m_a), False), # Eq 10
                (pr_e * (1 - pr_s), (savings_a, edu, savings_e, -1, max(ctt_a, m_a - m_c), 0, penalty, m_a), False) # Eq 10
            )
        elif health == -2:
            return (
                (pr_e, (savings_a, edu, 0, -2, ctt_a, migrate_sick, penalty, m_a), False),
            )
        else:
            bequest = (1 - migrate_sick) * savings_e - self._c_new_death # new death to savings_a
            # bequest = min(0, bequest)
            if c_h == 0:
                return (
                    (pr_e, (max(0, savings_a + bequest), edu, 0, -2, ctt_a, migrate_sick, penalty, m_a), True),
                )
            else:
                return (
                    (pr_e * pr_d, (max(0, savings_a + bequest), edu, 0, -2, ctt_a, migrate_sick, penalty, m_a), True),
                    (pr_e * (1 - pr_d), (savings_a, edu, savings_e, health, ctt_a, migrate_sick, penalty, m_a), False)
                )

    def _make_choice(self, int t, in_school, double savings_a, double savings_e, int health, int ctt_a, int loc):
        cache_key = t, in_school, savings_a, savings_e, health, ctt_a, loc
        ret = self._make_choice_cache.get(cache_key)
        if ret is not None:
            return ret

        cdef int choice_length_t
        choice_length_t = self._choice_length_t[t]
        assert savings_a >= 0
        assert savings_e >= 0

        ret = []
        min_c = 1000. # Ling is sure
        max_c = 8000., 21000. # Ling is sure
        subc = 1100, 8000 # subsistence consumption

        cdef double s_w_a_fyr
        cdef double s_w_e_fyr
        # m_a, m_c = 1, 1
        s_w_a_fyr = savings_a * self._ir_1yr[t] + self._w_1yr * choice_length_t # when m_a == 1, 2 cases
        s_w_e_fyr = savings_e * self._ir_1yr[t] # for all cases
        if health == -1:
            s_w_e_fyr += self._a_1yr * choice_length_t
        if t == 4:
            s_w_a_fyr -= self._joe1 * self._w_1yr * choice_length_t # joe1
        # adult choice
        if in_school: # for m_c == 1
            if t == 0:
                c_edu_choice = (self._choice_c_edu_1yr[t][1],) # kindergarten
            else:
                c_edu_choice = (0, self._choice_c_edu_1yr[t][1]) # Ling is sure
        else:
            c_edu_choice = (0,) # Ling is sure
        if health == -2:
            transfer_choice = (0.,) # Ling is sure
        elif health == -1:
            transfer_choice = (0.,) # Ling is sure
        else:
            transfer_choice = (2000., self._old_c) # Eq 11
        # elder choice
        c_h_choice = (0,) # Ling is sure
        if health >= 0:
            c_h_choice = (0, self._table_c_h_private_1yr[t]) # Ling is sure
        # adult choose
        if not (health == 0 and t == 0):
            for adult_choice_1yr in itertools.product(c_edu_choice, transfer_choice, self._sa_choice):
                c_edu_1yr, transfer_1yr, sa_fyr = adult_choice_1yr
                depend_t = 2.0 + self._e_scale_v1 if (not ((c_edu_1yr < 1e-10 and t == 3) or t == 4)) else 2.0 # OECD
                if sa_fyr > s_w_a_fyr:
                    continue
                if t < 2 and sa_fyr < 80000 and sa_fyr % 4000 != 0: # Ling is sure
                    continue
                c_edu_fyr = c_edu_1yr * choice_length_t
                transfer_fyr = transfer_1yr * choice_length_t
                c_tot_1yr = (s_w_a_fyr - c_edu_fyr - transfer_fyr - sa_fyr) / choice_length_t # Eq 3
                c_a_o_1yr = c_tot_1yr / depend_t # OECD
                if c_a_o_1yr < subc[1] or (c_a_o_1yr > max_c[1] and sa_fyr != self._sa_choice[-1]): # Ling is sure
                    continue
                if health == -2:
                    # append: m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e,level
                    ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, 0, 0, 0, 0))
                    continue
                # elder choose
                for elder_choice_1yr in itertools.product(c_h_choice, self._se_choice):
                    c_h_1yr, se_fyr = elder_choice_1yr
                    if se_fyr > s_w_e_fyr:
                        continue
                    if t < 2 and se_fyr < 60000 and se_fyr % 4000 != 0: # Ling is sure
                        continue
                    c_h_fyr = c_h_1yr * choice_length_t
                    c_e_o_1yr = (s_w_e_fyr - c_h_fyr + transfer_fyr - se_fyr) / choice_length_t # Eq 6
                    if c_e_o_1yr >= subc[0] and (max_c[0] > c_e_o_1yr or se_fyr == self._se_choice[-1]): # Ling is sure
                        # 正常
                        ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, c_e_o_1yr, c_h_1yr, se_fyr, 0))
                        continue
                    if c_h_1yr > 0 or se_fyr / choice_length_t + c_e_o_1yr >= subc[0]: # Ling is sure
                        continue
                    # 低保:
                    # (adult, elderly).
                    # (正常,正常)=0,
                    # (正常,低保)=1,
                    # (低保,正常)=2,
                    # (低保,低保)=3
                    # (0, 1, 2, 3)
                    ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, min_c, 0, 0, 1))

        # m_a, m_c = 1, 0
        if in_school: # for m_c == 0, 2 cases
            c_edu_choice = (0, self._choice_c_edu_1yr[t][0])
        else:
            c_edu_choice = (0,)
        if self._ctt_e == 1 and health == -1: # Eq 10
            transfer_choice = (4000., 6000.)
            # adult choose
            for adult_choice_1yr in itertools.product(c_edu_choice, transfer_choice, self._sa_choice):
                c_edu_1yr, transfer_1yr, sa_fyr = adult_choice_1yr
                depend_t = 1 + self._e_scale_v2 if (not ((c_edu_1yr < 1e-10 and t == 3) or t == 4)) else 1 # OECD
                if sa_fyr > s_w_a_fyr:
                    continue
                if t < 2 and sa_fyr < 80000 and sa_fyr % 4000 != 0: # Ling is sure
                    continue
                c_edu_fyr = c_edu_1yr * choice_length_t
                transfer_fyr = transfer_1yr * choice_length_t
                c_a_o_1yr = (s_w_a_fyr - c_edu_fyr - sa_fyr - transfer_fyr) / choice_length_t / 2.0 # Eq 3
                if c_a_o_1yr < subc[1] or (c_a_o_1yr > max_c[1] and sa_fyr != self._sa_choice[-1]): # Ling is sure
                    continue
                for elder_choice_1yr in itertools.product(c_h_choice, self._se_choice):
                    c_h_1yr, se_fyr = elder_choice_1yr
                    if se_fyr > s_w_e_fyr:
                        continue
                    if t < 2 and se_fyr < 60000 and se_fyr % 4000 != 0: # Ling is sure
                        continue
                    c_h_fyr = c_h_1yr * choice_length_t
                    c_tot_1yr = (s_w_e_fyr - c_h_fyr - se_fyr + transfer_fyr) / choice_length_t # Eq 6
                    c_e_o_1yr = c_tot_1yr / depend_t # OECD
                    if c_tot_1yr - c_e_o_1yr > transfer_1yr: # Eq 11
                        continue
                    if c_e_o_1yr < subc[0] or (c_e_o_1yr > max_c[0] and se_fyr != self._se_choice[-1]): # Ling is sure
                        continue
                    # 正常
                    ret.append((1, 0, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, c_e_o_1yr, c_h_1yr, se_fyr, 0))

        # m_a, m_c = 0, 0
        s_w_a_fyr = savings_a * self._ir_1yr[t] + self._a_1yr * 2 * choice_length_t # when m_a == 0, 1 case
        if health == -2:
            transfer_choice = (0.,)
        elif health == -1:
            transfer_choice = (0.,)
        else:
            transfer_choice = (0. if ctt_a == 0 else 3000., self._old_c) # Eq 11 consider 低保
        # adult choose
        for adult_choice_1yr in itertools.product(c_edu_choice, transfer_choice, self._sa_choice):
            c_edu_1yr, transfer_1yr, sa_fyr = adult_choice_1yr
            depend_t = 2.0 + self._e_scale_v2 if (not ((c_edu_1yr < 1e-10 and t == 3) or t == 4)) else 2.0 # OECD
            if sa_fyr > s_w_a_fyr:
                continue
            if t < 2 and sa_fyr < 80000 and sa_fyr % 4000 != 0: # Ling is sure
                continue
            c_edu_fyr = c_edu_1yr * choice_length_t
            transfer_fyr = transfer_1yr * choice_length_t
            c_tot_1yr = (s_w_a_fyr - c_edu_fyr - sa_fyr - transfer_fyr) / choice_length_t # Eq 3
            c_a_o_1yr = c_tot_1yr / depend_t # OECD
            if c_a_o_1yr > subc[0]:
                # 正常
                level_a = 0
            elif c_edu_1yr > 0: # Ling is sure
                continue
            elif transfer_1yr > 0:
                if not (t == 4 and self._joe1 + self._tax >= 0.55 and sa_fyr < 1e-10 and transfer_1yr == transfer_choice[0]): # Ling is sure
                    continue
                c_a_o_1yr += transfer_1yr / depend_t
                transfer_1yr = 0
                transfer_fyr = 0
                if c_a_o_1yr > subc[0]: # Ling is sure
                    # 正常
                    level_a = 0
                else:
                    # 低保
                    level_a = 1
                    c_a_o_1yr = min_c
            else:
                # 低保
                level_a = 1
                sa_fyr = 0
                c_a_o_1yr = min_c
            if c_a_o_1yr > max_c[0] and sa_fyr != self._sa_choice[-1]: # Ling is sure
                continue
            if health == -2:
                # append: m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level
                ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, 0, 0, 0, level_a * 2))
                continue
            # elder choose
            for elder_choice_1yr in itertools.product(c_h_choice, self._se_choice):
                c_h_1yr, se_fyr = elder_choice_1yr
                if se_fyr > s_w_e_fyr:
                    continue
                if t < 2 and se_fyr < 60000 and se_fyr % 4000 != 0: # Ling is sure
                    continue
                c_h_fyr = c_h_1yr * choice_length_t
                c_e_o_1yr = (s_w_e_fyr - c_h_fyr - se_fyr + transfer_fyr) / choice_length_t # Eq 6
                if c_e_o_1yr >= subc[0] and (max_c[0] > c_e_o_1yr or se_fyr == self._se_choice[-1]): # Ling is sure
                    # 正常
                    ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, c_e_o_1yr, c_h_1yr, se_fyr, level_a * 2))
                    continue
                if c_h_1yr > 0 or se_fyr / choice_length_t + c_e_o_1yr >= subc[0]: # Ling is sure
                    continue
                # 低保
                ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, min_c, 0, 0, level_a * 2 + 1))

        if len(ret) == 0:
            logging.critical('choices %s', [cache_key, self._UID])
            exit(1)

        # now we work with the full list and clean some cases.
        ret_new = []
        for choice in ret:
            assert len(choice) == 10
            m_a, m_c, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, c_e_o_1yr, c_h_1yr, se_fyr, level = choice
            c_a_o_1yr = round(c_a_o_1yr / self._c_grid) * self._c_grid # Ling is sure
            c_e_o_1yr = round(c_e_o_1yr / self._c_grid) * self._c_grid # Ling is sure
            ret_new.append((m_a, m_c, c_a_o_1yr, c_edu_1yr, sa_fyr, transfer_1yr, c_e_o_1yr, c_h_1yr, se_fyr, level))

        # we also drop duplicates.
        # ret_new = sorted(set(ret_new), key=ret_new.index)
        # ret_new = {}.fromkeys(ret_new).keys()
        ret_new = tuple(set(ret_new))
        self._make_choice_cache[cache_key] = ret_new
        return ret_new

    def _utility_fyr(self, _u, avg, choice_length_t): # Eq 16 and 18
        cache_key = _u, avg, choice_length_t
        ret = self._utility_fyr_cache.get(cache_key)
        if ret is not None:
            return ret

        ret = _u * self._disc_fa[choice_length_t] + self._disc_fa2[choice_length_t] * avg

        self._utility_fyr_cache[cache_key] = ret
        return ret

    def _utility_final_a(self, saving, edu, fame):
        cache_key = saving, edu, fame
        ret = self._utility_final_cache.get(cache_key)
        if ret is not None:
            return ret

        penalty = 1 + min(0, fame) # Eq 12
        saving = round(saving / 15.646 / 100) * 100 # Eq 15
        _u1 = self._u1(saving + self._a_1yr, self._table_leisure[0][0], self._theta) # Eq 7
        _u3 = self._u3(self._state_edu_length_t[edu]) # Eq 15
        ret = _u1 * _u3 * penalty * self._disc_fa[20] # Eq 15

        self._utility_final_cache[cache_key] = ret
        return ret

    def _utility_final_e(self, edu, new_death): # Ling is sure
        cache_key = edu, new_death
        ret = self._utility_final_e_cache.get(cache_key)
        if ret is not None:
            return ret

        if not new_death:
            return 0
        _u3 = max(1, self._state_edu_length_t[edu])
        ret = np.power(_u3 * 1., 1 - self._phi) / (1 - self._phi) * self._disc_fa[3]

        self._utility_final_e_cache[cache_key] = ret
        return ret

    def _ue(self, health, m_a, m_c, c_e_o, c_e, c_h, t): # Eq 4
        cache_key = health, m_a, m_c, c_e_o, c_e, c_h, t
        ret = self._ue_cache.get(cache_key)
        if ret is not None:
            return ret

        assert health >= -1
        leisure = self._table_leisure[0][m_a - m_c] # childcare leisure and Eq 7
        if t == 0 and leisure == self._table_leisure[0][1]:
            leisure -= self._babysit
        ret = self._u1(c_e_o, leisure, self._theta) * self._uhealth(health >= 0, c_h > 0) # childcare leisure and c_e_o and Eq 7 and Eq 5
        if t == 3 and c_e > 0:
            ret *= self._lamda
        ret += self._z2*self._uhealth(health >= 0, c_h > 0) * (self._lamda if t == 3 and c_e > 0 else 1)

        self._ue_cache[cache_key] = ret
        return ret

    def _ua(self, c_a_o, m_a, m_c, c_e, t, fame, is_sick, ctt_a): # Eq 2
        cache_key = c_a_o, m_a, m_c, c_e, t, fame, is_sick, ctt_a
        ret = self._ua_cache.get(cache_key)
        if ret is not None:
            return ret

        penalty = 1 + min(0, fame) # Eq 12
        leisure = self._table_leisure[m_a][1 - (m_a - m_c) if t < 4 else 0] # childcare leisure and Eq 7
        if t == 0 and leisure == self._table_leisure[0][1]:
            leisure -= self._babysit
        if m_a == 0 and is_sick and ctt_a == 1: # Eq 7
            leisure -= 12
        if m_a == 1:
            c_a_o *= 0.154320988 # price ratio
        ret = self._u1(c_a_o, leisure, self._theta) * penalty # Eq 2
        if t == 3 and c_e > 0:
            ret *= self._lamda
        ret += self._z1*penalty*(self._lamda if t == 3 and c_e > 0 else 1)

        self._ua_cache[cache_key] = ret
        return ret

    def _u1(self, c, l, theta): # Eq 2 and 4
        cache_key = c, l, theta
        ret = self._u1_cache.get(cache_key)
        if ret is not None:
            return ret

        cl = np.power(c, theta) * np.power(l, 1 - theta)
        cl = cl / 1.
        ret = self._isoelastic(cl, self._gamma)
        # assert ret >= 0

        self._u1_cache[cache_key] = ret
        return ret

    def _u3(self, edu): # Eq 15
        cache_key = edu
        ret = self._u3_cache.get(cache_key)
        if ret is not None:
            return ret

        edu = max(1, edu)
        edu = edu * 1.
        ret = self._isoelastic(edu, self._phi) + 1

        self._u3_cache[cache_key] = ret
        return ret

    def _isoelastic(self, c, phi): # Ling is sure
        cache_key = c, phi
        ret = self._isoelastic_cache.get(cache_key)
        if ret is not None:
            return ret

        ret = (np.power(c, 1 - phi) - 1) / (1 - phi)

        self._isoelastic_cache[cache_key] = ret
        return ret

    def _penalty(self, ctt_a, health, m_a, m_c, transfer, fame): # Eq 12
        # no guilt if no contract or if elderly is healthy.
        # when contract holds and elderly is sick, penalty is imposed when (not coming back to rural area) + (not sending enough money)
        if health == -2:
            return fame
        if health == -1:
            return 0

        if self._ctt_e == 0: # Eq 10
            return 0
        elif ctt_a == 0 and m_a == m_c:
            return 0

        rules_breaking = 0
        if transfer < self._old_c: # Eq 11
            rules_breaking -= self._kappa1
        if m_a == 1:
            rules_breaking -= self._kappa2
        rules_breaking = min(rules_breaking, fame)

        if rules_breaking == 0:
            rules_breaking = 1
        return rules_breaking

    def _uhealth(self, is_sick, private_medical_cost): # Eq 5
        cache_key = is_sick, private_medical_cost
        ret = self._uhealth_cache.get(cache_key)
        if ret is not None:
            return ret

        # have a fixed disutility of being sick
        # also have a fixed utility of being treated when sick
        if not is_sick:
            return 1
        uh = 1 - self._disutility_sick + (self._utility_treated if private_medical_cost else 0.)
        ret = max(uh, 0)

        self._uhealth_cache[cache_key] = ret
        return ret