#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
cimport numpy as np
import pickle
from multiprocessing import Pool as PPool
import sys
import itertools
import time
import os
import logging
import hashlib
from pathlib import Path


class Forward(object):
    def __init__(self, limit):
        self._route = []
        self._cache = {}
        self._limit = limit  # T 0 ->
        self._state_len = limit + 1
        self._choice_len = limit
        self._acc_count = 0
        self._appro_amt = [1] * (limit * 2)
        self._sum = 1
        self._start = time.time()
        self._concurrent = False
        self._runtime = 0
        self._my_future_count = 0
        self._cache_len = 0

    @staticmethod
    def _check_hash(raw_name):
        name = Path(raw_name)
        if not name.exists():
            name = name.with_suffix('.py')
            assert name.exists()
        with name.open('rb') as f:
            content = f.read()
        check = hashlib.md5(content).hexdigest()
        logging.info('RunFile         : \t%s', str(name))
        logging.info('RunFile Hash    : \t%s', check)
        return Path(os.path.dirname(raw_name)) / 'temp_output', check

    def _add(self, times, t):
        if times > self._appro_amt[t]:
            # print((t, times))
            self._appro_amt[t] = times
            s = 1
            for i in self._appro_amt[:-1]:
                s *= i
            self._sum = s

    @staticmethod
    def _list_format(l):
        return '[' + ', '.join(['{:.1f}'.format(i) for i in l]) + ']'

    @staticmethod
    def _list_format2(l):
        return '[' + ', '.join(['{:.5f}'.format(i) for i in l]) + ']'

    @staticmethod
    def _split(lower, upper, grid=3):  # 3 -> 2s, 10 -> 20000000s
        if upper < lower:
            return []
        elif upper == lower:
            return [lower]
        step = (upper - lower) * 1.0 / (grid - 1)
        step = max(step, 0.1)
        return set([min(lower + i * step, upper) for i in range(grid)])

    def output(self):
        raise RuntimeError('NOT USE')

    def run(self):
        raise RuntimeError('NOT USE')

    def get_result(self):
        return self._route


class Full(Forward):
    def __init__(self, a, ctt_e, health, parameter, limit=5, concurrent=True):
        Forward.__init__(self, limit)

        self._gamma, self._theta1, self._theta2, self._lamda, self._phi, \
        self._kappa1, self._kappa2, self._rho_migrate_sick, self._utility_treated, \
        self._disutility_sick, self._discount_factor = parameter

        self._init_para()
        self._init_cache()
        self._concurrent = concurrent
        self._a_1yr = a
        self._ctt_e = ctt_e
        self._health = health
        assert ctt_e in (0, 1)
        self._key = '{0}, {1}'.format((a, ctt_e, health), self._list_format2(parameter))

        self._output_dir, self._check_sum = self._check_hash(os.path.dirname(os.path.realpath(__file__)) + os.sep + 'model.pyx')
        self._output_dir = self._output_dir / self._check_sum
        self._output_dir.mkdir(exist_ok=True)

        logging.info('Key             : \t%s', self._key)
        self._key = hashlib.md5(self._key.encode('utf-8')).hexdigest()
        self._output_file = self._output_dir / self._key
        logging.info('Key Hash        : \t%s', self._key)

    def _init_cache(self):
        self._make_choice_cache = {}
        self._utility_a_cache = {}
        self._utility_e_cache = {}
        self._disc_fa_cache = {}
        self._disc_fa2_cache = {}
        self._utility_final_cache = {}
        self._utility_final_e_cache = {}
        self._ue_cache = {}
        self._ua_cache = {}
        self._u1a_cache = {}
        self._u1e_cache = {}
        self._u3_cache = {}
        self._isoelastic_cache = {}
        self._uhealth_cache = {}

    def _init_para(self):
        # length of each time period, in years. t=0: 6 years, t=1: 6 years, t=2: 3 years, t=3: 3 years, t=4: Terminal condition.
        self._choice_length_t = (6, 6, 3, 3, 2)
        assert len(self._choice_length_t) == self._choice_len
        # new edu length
        self._state_edu_length_t = (0, 0, 6, 9, 12, 16)  # [0] + list(np.add.accumulate(self._length_t).tolist())
        assert len(self._state_edu_length_t) == self._state_len
        # annual wage in the urban area
        self._w_1yr = 24500
        # average of 3 years total income of a 15~18 years old worker
        # self._child_wage_1yr = 10000
        # lower bound of consumption by location, rural and urban
        # self._table_c_o = [[1000, 2000, 3000, 4000, 5000], [5000, 10000, 15000, 20000]]
        # self._table_c_o_1yr = [[2000, 4000], [11000, 21000]]
        self._table_c_o_1yr = [[2000, 4000], [11000, 21000]]
        # private cost of the education, indexed by [t][m_c]
        self._choice_c_edu_1yr = [[275 * 12, 373 * 12], [969, 3833], [1834, 4153], [4390, 7078], [15000] * 2]
        assert len(self._choice_c_edu_1yr) == self._choice_len
        # upper bound of the savings
        # self._savings_max = 50000
        # borrowing limit: should be 0 or negative. Represents the maximum amount of debt.
        self._loan = 0
        # number of grids of the savings
        # self._savings_grid = 20
        # upper bound of the transfer
        # self._transfer_max = 5000
        # number of grids of the transfer
        # self._transfer_grid = 2
        # lower bound for % of medical cost paid by adult in contract to "commit to the contract".
        self._contract_medical = 0.8
        self._c_new_death = 8209
        # government's share in paying for the medical cost
        self._gov_health = 0.34
        # total medical expenditure by length of sickness
        # what I want now is cost = 3305RMB/yr, and after the elderly's death, there's an extra 8209RMB one time cost.
        self._c_h_1yr = 3305  # + 8209 / 3
        table_c_h_total_1yr = (np.asarray([self._c_h_1yr] * self._choice_len, dtype=np.float64) * (1 - self._gov_health)).tolist()
        # out of pocket medical expenditure by length of sickness
        self._table_c_h_private_1yr = [int(i / 10) * 10 for i in table_c_h_total_1yr]
        # amount of leisure time, depend on location, taking care of child, health:
        #   [[rural no child, rural with child],[urban no child, urban with child]].
        #   Endowment: 12 hours per day. The leisure is hours per week.
        self._table_leisure = list((np.asarray([[34, 22], [18, 6]], dtype=np.float64) * 200).tolist())
        # probability of getting sick at the end of this period, by [t]
        # pr_s_base_3yr = [0.15183247, 0.30407796, 0.15683839, 0.08687458]
        # pr_s_base_3yr = [[pr_s_base_3yr[i], min(1., pr_s_base_3yr[i] * self._rho_childcare)] for i in range(self._choice_len - 1)]
        # pr_s_base_3yr = [[0., 0.]] + pr_s_base_3yr
        self._rho_childcare = 1
        pr_s_base_3yr = 0.15
        pr_s_base_3yr = [[pr_s_base_3yr, min(1., pr_s_base_3yr * self._rho_childcare)]] * self._choice_len
        # probability of getting sick at the end of this period, depending on childcare history, by [t][0 if no childcare, 1 if provided childcare]
        pr_s_base_3yr = 1 - np.asarray(pr_s_base_3yr, dtype=np.float64)
        length_t = np.column_stack([self._choice_length_t] * 2)
        length_t = np.asarray(length_t, dtype=np.float64) / 3.
        pr_s_base = 1 - np.asarray(np.power(pr_s_base_3yr, length_t), dtype=np.float64)
        self._table_pr_s_fullyr = pr_s_base
        # probability of dying at the end of this period, by [0 if no migration when sick, 1 if migrated when sick]
        # self._pr_d_base = 0.84683343
        pr_d_base_3yr = 0.64683343
        pr_d_base_3yr = [[pr_d_base_3yr, min(1., pr_d_base_3yr * self._rho_migrate_sick)]] * self._choice_len
        pr_d_base_3yr = 1 - np.asarray(pr_d_base_3yr, dtype=np.float64)
        pr_d_base = 1 - np.asarray(np.power(pr_d_base_3yr, length_t), dtype=np.float64)
        self._table_pr_d_fullyr = pr_d_base
        # Pr(not in school at the beginning of the next period | in school now, and the adult pays c_edu), by [t][ma==mc, ma!=mc]
        # 落榜=0.7, confirmed from data
        self._table_pr_e_fullyr = [[0., 0., 0.], [0.0002, 0.0007, 0.], [0.0471, 0.0957, 0.], [0.0474, 0.0519, 0.], [0.1455, 0.1389, 0.7]]
        assert len(self._table_pr_e_fullyr) == self._choice_len
        # 1-beta, annual value
        # self._discount_factor = 1 / 1.0316  # NOTE: when using this variable, the function does not have t now.
        # annual interest rate on savings
        ir_1yr_base = 0.0275
        self._ir_1yr = np.power(1 + ir_1yr_base, self._choice_length_t)
        self._e_scale = 0.5
        self._w_scale = 4.14

    def output(self):
        tmp = 'T: {0}\tState: {1}\tP: {2:.5f}\tV_A: {4:.5f}\tV_E: {5:.5f}\tChoice: {3}'
        for i in self._route:
            for j in i:
                s = self._list_format(j[3])
                c = self._list_format([] if j[4] is None else j[4])
                logging.debug(tmp.format(j[2], s, j[5], c, j[0], j[1]))
            logging.debug('*' * 60)

    def run(self, show=True):
        if len(self._route) == 0:
            if self._output_file.exists():
                with self._output_file.open('rb') as f:
                    self._route = pickle.load(f)
                    self._acc_count = pickle.load(f)
                    self._sum = pickle.load(f)
                    self._appro_amt = pickle.load(f)
                    self._my_future_count = pickle.load(f)
                    self._cache_len = pickle.load(f)
            else:
                # the vector below is: (t, (savings_a, edu, savings_e, health, ctt_a, migrate_sick, childcare, fame, loc))
                tmp_next, self._my_future_count = self._recursion(0, (0.0, 0, self._a_1yr * self._w_scale, self._health, 0, 0, 0, 0, 0))
                self._route.extend(tmp_next)
                with self._output_file.open('wb') as f:
                    pickle.dump(self._route, f)
                    pickle.dump(self._acc_count, f)
                    pickle.dump(self._sum, f)
                    pickle.dump(self._appro_amt, f)
                    pickle.dump(self._my_future_count, f)
                    self._cache_len = len(self._cache)
                    pickle.dump(self._cache_len, f)
            self._runtime = time.time() - self._start

        if not show:
            return
        logging.info('Seconds         : \t%.4f', self._runtime)
        logging.info('Route Count     : \t%d', len(self._route))
        logging.info('Future Count    : \t%d', self._my_future_count)
        logging.info('Recursion Count : \t%d', self._acc_count)
        logging.info('Cache Count     : \t%d', self._cache_len)

        logging.info('Adoption Rate   : \t%.4f %%', len(self._route) * 100 / self._my_future_count)
        logging.info('Coincidence Rate: \t%.4f %%', (self._my_future_count - self._acc_count) * 100 / self._my_future_count)
        logging.info('Cache Rate      : \t%.4f %%', (self._acc_count - self._cache_len) * 100 / self._acc_count)
        logging.info('Speed           : \t%.4f Req/Sec', self._my_future_count / self._runtime)

        logging.info('Guess Count     : \t%d', self._sum)
        logging.info('Guess Rate      : \t%.4f %%', self._my_future_count * 100 / self._sum)
        logging.info('Guess Appro     : \t%s', self._appro_amt)
        logging.info('*' * 60)

    # return [[(v_a, v_e, t, s, choice, p)]]]
    def _recursion(self, t, state: tuple, new_death=False):
        self._acc_count += 1
        cache_key = t, state, new_death
        ret = self._cache.get(cache_key)
        if ret is not None:
            return ret

        savings_a, edu, savings_e, health, ctt_a, migrate_sick, childcare, fame, loc = state
        if self._limit == t:
            if health != -2:
                health = -2
                new_death = True
                bequest = (1 - migrate_sick) * savings_e - self._c_new_death
                # bequest = min(0, bequest)
                savings_a = max(0, savings_a + bequest)
            assert health == -2
            ret = (((
                        self._utility_final_a(savings_a, edu, fame),
                        self._utility_final_e(edu, new_death, 3),
                        t,
                        (*state[:3], health, *state[4:]),
                        None,
                        np.float64(1.0)
                    ),),)
            self._cache[cache_key] = ret, 1
            return ret, 1

        choice = self._make_choice(t, edu == t, savings_a, savings_e, health, ctt_a, loc)
        self._add(len(choice), t * 2)

        if t == 0 and self._concurrent and self._ctt_e != 0 and self._health == -1:
            logging.info('Concurrent Pool : \t%d', len(choice))
            pool = PPool(8)
            save_state_list = []
            for cur_choice in choice:
                save_state = pool.apply_async(self._test_one_choice,
                                              (*cur_choice, t, edu, migrate_sick, health, ctt_a, childcare, new_death, fame),
                                              error_callback=logging.warning)
                save_state_list.append(save_state)
            pool.close()
            choice = save_state_list

        all_futures = {}
        my_future_count = 0
        for cur_choice in choice:
            if isinstance(cur_choice, tuple):
                tmp_v_a, tmp_v_e, ret_next, \
                m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level, \
                _, _, _, f_count = self._test_one_choice(
                    *cur_choice, t, edu, migrate_sick, health, ctt_a, childcare, new_death, fame)
            else:
                tmp_v_a, tmp_v_e, ret_next, \
                m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level, \
                acc, appro, up_cache, f_count = cur_choice.get()
                self._acc_count += acc
                for approi, approv in enumerate(appro):
                    self._add(approv, approi)
                self._cache.update(up_cache)
            my_future_count += f_count

            key = (m_a, m_c, savings_a, c_edu, transfer)
            if key not in all_futures or tmp_v_e > all_futures[key][0]:
                all_futures[key] = tmp_v_a, tmp_v_e, ret_next, \
                                   m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level

                # logging.warning(str((tmp_v_a, tmp_v_e, t, new_death, np.infty,
                #                      savings_a, edu, savings_e, health, ctt_a, migrate_sick, childcare, fame, np.infty,
                #                      m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level))[1:-1])
        max_v = (float('-inf'),)
        for save_state in all_futures.values():
            if save_state[0] > max_v[0]:
                max_v = save_state

        try:
            assert len(max_v) == 13
        except AssertionError:
            logging.critical('all_futures: %s', all_futures)
            logging.critical('t: %s', t)
            logging.critical('choice len: %s', len(choice))
            exit(1)

        tmp_v_a, tmp_v_e, ret_next, \
        m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level = max_v
        ret = []
        for i in ret_next:
            ret.append(((tmp_v_a, tmp_v_e, t, state,
                         (m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level),
                         i[0][5]), *i))
        ret = tuple(ret)

        self._cache[cache_key] = ret, my_future_count
        return ret, my_future_count

    def _test_one_choice(self,
                         m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level,
                         t, edu, migrate_sick, health, ctt_a, childcare, new_death, fame):
        # childcare is 1 when the child had been taken care of by the elderly (when the elderly was healthy)
        # migrate_sick is 1 if the adult migrated when the elderly is sick.
        tmp_childcare = childcare
        tmp_childcare2 = m_a - m_c if health == -1 else 0
        if tmp_childcare < tmp_childcare2:
            tmp_childcare = tmp_childcare2

        migrate_sick2 = 1 if health >= 0 and m_a == 1 else 0
        if migrate_sick < migrate_sick2:
            migrate_sick = migrate_sick2

        contract_hold = ctt_a if ctt_a > m_a else m_a
        if contract_hold > self._ctt_e:
            contract_hold = self._ctt_e

        penalty = self._penalty(contract_hold == 1, health, m_a, transfer, fame)

        task = self._judge_edu(m_a, m_c, c_edu, c_h,
                               savings_a, savings_e, t, edu, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                               childcare)
        self._add(len(task), t * 2 + 1)

        all_pr = 0.
        avg_a = 0.
        avg_e = 0.
        ret_next = []
        my_future_count = 0
        for prob, state, new_death in task:
            if prob <= 1e-15:
                continue
            ret_iter, one_pr, one_a, one_e, f_count = self._update_pr(self._recursion(t + 1, state, new_death=new_death), prob)
            ret_next.extend(ret_iter)
            all_pr += one_pr
            avg_a += one_a
            avg_e += one_e
            my_future_count += f_count
        assert len(ret_next) > 0
        assert 1.0e-8 >= abs(all_pr - 1.0)

        if health != -2:
            _ue = self._ue(health, m_a, m_c, c_e_o, c_edu, c_h, t)
            tmp_v_e = self._utility_e(_ue, avg_e, self._choice_length_t[t])
        else:
            # _ue = 0
            tmp_v_e = self._utility_final_e(edu, new_death, self._choice_length_t[t])

        _ua = self._ua(c_a_o, m_a, m_c, c_edu, t, penalty)
        tmp_v_a = self._utility_a(avg_a, t, _ua)

        # logging.warning(str((t, savings_a, edu, savings_e, health, ctt_a, migrate_sick, childcare, fame, 1 if new_death else 0,
        #                     _ua, _ue,
        #                     m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level))[1:-1])

        return tmp_v_a, tmp_v_e, ret_next, \
               m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level, \
               self._acc_count, self._appro_amt, self._cache, my_future_count

    def _judge_edu(self,
                   m_a, m_c, c_edu, c_h,
                   savings_a, savings_e, t, edu, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                   childcare):
        if c_edu > 0 or t == 0:
            assert t == edu
            pr_e = self._table_pr_e_fullyr[t][m_a - m_c]
            return (
                *self._judge_health(
                    m_a, m_c, c_h, childcare,
                    savings_a, savings_e, t, edu + 1, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                                             (1.0 - pr_e) * (1 - self._table_pr_e_fullyr[t][2])),
                *self._judge_health(
                    m_a, m_c, c_h, childcare,
                    savings_a, savings_e, t, edu, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                    pr_e),
                *self._judge_health(
                    m_a, m_c, c_h, childcare,
                    savings_a + c_edu, savings_e, t, edu, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                    (1.0 - pr_e) * self._table_pr_e_fullyr[t][2])
            )
        else:
            return self._judge_health(
                m_a, m_c, c_h, childcare,
                savings_a, savings_e, t, edu, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                1.)

    def _judge_health(self,
                      m_a, m_c, c_h, childcare,
                      savings_a, savings_e, t, edu, tmp_childcare, migrate_sick, health, ctt_a, penalty,
                      pr_e: float):
        if abs(pr_e) < 1.0e-15:
            return []

        if t == self._limit - 1:
            pr_d = 1.0
            pr_s = 0.0
        elif health == -1:
            pr_d = 0.0
            pr_s = self._table_pr_s_fullyr[t, childcare]
        else:
            pr_d = self._table_pr_d_fullyr[t, migrate_sick]
            pr_s = 0.0

        if health == -1:
            return (
                (pr_e * pr_s, (savings_a, edu, savings_e, t + 1, max(ctt_a, m_a - m_c), 0, tmp_childcare, penalty, m_a), False),
                (pr_e * (1 - pr_s), (savings_a, edu, savings_e, -1, max(ctt_a, m_a - m_c), 0, tmp_childcare, penalty, m_a), False)
            )
        elif health == -2:
            return (
                (pr_e, (savings_a, edu, 0, -2, ctt_a, migrate_sick, tmp_childcare, penalty, m_a), False),
            )
        else:
            bequest = (1 - migrate_sick) * savings_e - self._c_new_death
            # bequest = min(0, bequest)
            if c_h == 0:
                return (
                    (pr_e, (max(0, savings_a + bequest), edu, 0, -2, ctt_a, migrate_sick, tmp_childcare, penalty, m_a), True),
                )
            else:
                return (
                    (pr_e * pr_d, (max(0, savings_a + bequest), edu, 0, -2, ctt_a, migrate_sick, tmp_childcare, penalty, m_a), True),
                    (pr_e * (1 - pr_d), (savings_a, edu, savings_e, health, ctt_a, migrate_sick, tmp_childcare, penalty, m_a), False)
                )

    @staticmethod
    def _update_pr(recursion_ret: tuple, pr: float):
        assert abs(pr) > 1e-15
        tmp_next, my_future_count = recursion_ret
        all_pr = 0.
        avg_a = 0.
        avg_e = 0.
        ret = []
        for tmp_p in tmp_next:
            v_a, v_e, t, s, choice, p = tmp_p[0]
            p *= pr
            all_pr += p
            avg_a += p * v_a
            avg_e += p * v_e
            ret.append(((v_a, v_e, t, s, choice, p), *tmp_p[1:]))
        return ret, all_pr, avg_a, avg_e, my_future_count

    def _make_choice(self, t, in_school, savings_a: float, savings_e: float, health, ctt_a, loc):
        cache_key = t, in_school, savings_a, savings_e, health, ctt_a, loc
        ret = self._make_choice_cache.get(cache_key)
        if ret is not None:
            return ret

        choice_length_t = self._choice_length_t[t]
        # trans_med = self._table_c_h_private_1yr[t] * self._contract_medical
        assert savings_a >= 0
        assert savings_e >= 0

        ret = []
        depend_t = not (not in_school and t >= 3)
        min_c = 1000

        # m_a, m_c = 1, 1
        s_w_a = savings_a * self._ir_1yr[t] + self._w_1yr * choice_length_t
        s_w_e = savings_e * self._ir_1yr[t] + (self._a_1yr * choice_length_t if health == -1 else 0)
        # adult choice
        if in_school:
            c_edu_choice = (0, self._choice_c_edu_1yr[t][1])
        else:
            c_edu_choice = (0,)
        c_a_o_choice = tuple(self._table_c_o_1yr[1])
        if health == -2:
            transfer_choice = (0.,)
        elif health == -1:
            transfer_choice = (0.,)
        else:
            transfer_choice = (2000., 2000. + self._contract_medical * self._c_h_1yr * (1 - self._gov_health))
        # elder choice
        c_h_choice = (0,)
        if health >= 0:
            c_h_choice = (0, self._table_c_h_private_1yr[t])
        c_e_o_choice = tuple(self._table_c_o_1yr[0])
        # adult choose
        if not (health == 0 and t == 0) and not (t == 4 and loc == 0):
            for adult_choice_1yr in itertools.product(c_edu_choice, c_a_o_choice, transfer_choice):
                c_edu_1yr, c_a_o_1yr, transfer_1yr = adult_choice_1yr
                c_edu_fyr = c_edu_1yr * choice_length_t
                c_a_o_fyr = c_a_o_1yr * choice_length_t
                transfer_fyr = transfer_1yr * choice_length_t
                c_c_o_fyr = c_a_o_fyr * self._e_scale if depend_t else 0
                sa = s_w_a - c_edu_fyr - c_a_o_fyr - c_c_o_fyr - transfer_fyr
                if sa < 0:
                    continue
                if health == -2:
                    # append: m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e,level
                    ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, 0, 0, 0, 0))
                    continue
                # elder choose
                for elder_choice_1yr in itertools.product(c_e_o_choice, c_h_choice):
                    c_e_o_1yr, c_h_1yr = elder_choice_1yr
                    c_e_o_fyr = c_e_o_1yr * choice_length_t
                    c_h_fyr = c_h_1yr * choice_length_t
                    se = s_w_e - c_h_fyr - c_e_o_fyr + transfer_fyr
                    if se >= 0:
                        # 正常
                        ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, c_h_1yr, se, 0))
                        continue
                    if c_e_o_1yr > self._table_c_o_1yr[0][0] or c_h_1yr > 0:
                        continue
                    c_e_o_1yr = min_c
                    c_e_o_fyr = c_e_o_1yr * choice_length_t
                    se = s_w_e - c_h_fyr - c_e_o_fyr + transfer_fyr
                    if se < 0:
                        # 低保:
                        # (adult, elderly).
                        # (正常,正常)=0,
                        # (正常,低消费)=1,
                        # (正常,低保)=2,
                        # (低消费,正常)=3,
                        # (低消费,低消费)=4,
                        # (低消费,低保)=5,
                        # (低保,正常)=6,
                        # (低保,低消费)=7,
                        # (低保,低保)=8
                        # (0, 0, 1, 0, 0, 1, 2, 2, 3)
                        ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, 0, 0, 2))
                        continue
                    # 低消费
                    ret.append((1, 1, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, c_h_1yr, se, 1))
                    continue

        # m_a, m_c = 1, 0
        if in_school:
            c_edu_choice = (0, self._choice_c_edu_1yr[t][0])
        else:
            c_edu_choice = (0,)
        if self._ctt_e == 1 and health == -1 and depend_t and not (t == 4 and loc == 0):
            transfer_choice = (4000., 6000.)
            # adult choose
            for adult_choice_1yr in itertools.product(c_edu_choice, c_a_o_choice, transfer_choice):
                c_edu_1yr, c_a_o_1yr, transfer_1yr = adult_choice_1yr
                c_edu_fyr = c_edu_1yr * choice_length_t
                c_a_o_fyr = c_a_o_1yr * choice_length_t
                transfer_fyr = transfer_1yr * choice_length_t
                sa = s_w_a - c_edu_fyr - c_a_o_fyr - transfer_fyr
                if sa < 0:
                    continue
                for elder_choice_1yr in itertools.product(c_e_o_choice, c_h_choice):
                    c_e_o_1yr, c_h_1yr = elder_choice_1yr
                    c_e_o_fyr = c_e_o_1yr * choice_length_t
                    c_h_fyr = c_h_1yr * choice_length_t
                    c_c_o_fyr = c_e_o_fyr * self._e_scale if depend_t else 0
                    if c_c_o_fyr > transfer_fyr:
                        continue
                    se = s_w_e - c_h_fyr - c_c_o_fyr - c_e_o_fyr + transfer_fyr
                    if se >= 0:
                        # 正常
                        ret.append((1, 0, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, c_h_1yr, se, 0))

        # m_a, m_c = 0, 0
        s_w_a = savings_a * self._ir_1yr[t] + self._a_1yr * choice_length_t
        c_a_o_choice = c_e_o_choice
        if health == -2:
            transfer_choice = (0.,)
        elif health == -1:
            transfer_choice = (0.,)
        else:
            transfer_choice = (0. if ctt_a == 0 else 3000., 2000. + self._contract_medical * self._c_h_1yr * (1 - self._gov_health))
        # adult choose
        for adult_choice_1yr in itertools.product(c_edu_choice, c_a_o_choice, transfer_choice):
            c_edu_1yr, c_a_o_1yr, transfer_1yr = adult_choice_1yr
            c_edu_fyr = c_edu_1yr * choice_length_t
            c_a_o_fyr = c_a_o_1yr * choice_length_t
            transfer_fyr = transfer_1yr * choice_length_t
            c_c_o_fyr = c_a_o_fyr * self._e_scale if depend_t else 0
            sa = s_w_a - c_edu_fyr - c_a_o_fyr - c_c_o_fyr - transfer_fyr
            # 正常
            level_a = 0
            if sa < 0:
                if c_a_o_1yr > self._table_c_o_1yr[0][0] or c_edu_1yr > 0:
                    continue
                transfer_fyr = transfer_1yr = 0.
                sa = s_w_a - c_edu_fyr - c_a_o_fyr - c_c_o_fyr - transfer_fyr
                # 低消费
                level_a = 1
                if sa < 0:
                    c_a_o_1yr = min_c
                    c_a_o_fyr = c_a_o_1yr * choice_length_t
                    c_c_o_fyr = c_a_o_fyr * self._e_scale if depend_t else 0
                    sa = s_w_a - c_edu_fyr - c_a_o_fyr - c_c_o_fyr - transfer_fyr
                    # 低消费
                    level_a = 1
                    if sa < 0:
                        # 低保
                        level_a = 2
                        sa = 0
            if health == -2:
                # append: m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level
                ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, 0, 0, 0, level_a * 3))
                continue
            # elder choose
            for elder_choice_1yr in itertools.product(c_e_o_choice, c_h_choice):
                c_e_o_1yr, c_h_1yr = elder_choice_1yr
                c_e_o_fyr = c_e_o_1yr * choice_length_t
                c_h_fyr = c_h_1yr * choice_length_t
                se = s_w_e - c_h_fyr - c_e_o_fyr + transfer_fyr
                if se >= 0:
                    # 正常
                    ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, c_h_1yr, se, level_a * 3 + 0))
                    continue
                if c_e_o_1yr > self._table_c_o_1yr[0][0] or c_h_1yr > 0:
                    continue
                c_e_o_1yr = min_c
                c_e_o_fyr = c_e_o_1yr * choice_length_t
                se = s_w_e - c_h_fyr - c_e_o_fyr + transfer_fyr
                if se < 0:
                    # 低保
                    ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, 0, 0, level_a * 3 + 2))
                    continue
                # 低消费
                ret.append((0, 0, c_a_o_1yr, c_edu_1yr, sa, transfer_1yr, c_e_o_1yr, c_h_1yr, se, level_a * 3 + 1))
                continue

        # now we work with the full list and clean some cases.
        ret_new = []
        for choice in ret:
            assert len(choice) == 10
            m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level = choice
            assert m_a in (0, 1)
            assert m_c in (0, 1)
            assert c_a_o >= min_c
            assert c_edu >= 0
            assert savings_a >= 0
            assert transfer >= 0
            assert health == -2 or c_e_o >= min_c
            assert c_h >= 0
            assert savings_e >= 0
            # drop if saving < borrowing limit
            if savings_a < self._loan or savings_e < self._loan:
                continue
            # logging.debug('C %s', (t, in_school, health, savings_a, savings_e, choice))
            savings_a = self._grid_savings_a(savings_a)
            savings_e = self._grid_savings_e(savings_e)

            ret_new.append((m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level))
        # we also drop duplicates.
        # ret_new = sorted(set(ret_new), key=ret_new.index)
        # ret_new = {}.fromkeys(ret_new).keys()
        ret_new = tuple(set(ret_new))
        self._make_choice_cache[cache_key] = ret_new
        return ret_new

    @staticmethod
    def _grid_savings_a(savings_a):
        if savings_a <= 6000:
            return int(savings_a / 100 + 1) * 100
        elif savings_a <= 130000:
            return int(savings_a / 1500 + 1) * 1500
        elif savings_a <= 200000:
            return int(savings_a / 10000 + 1) * 10000
        else:
            return 200000

    @staticmethod
    def _grid_savings_e(savings_e):
        if savings_e <= 24000:
            return int(savings_e / 1500 + 1) * 1500
        elif savings_e <= 81000:
            return int(savings_e / 500 + 1) * 500
        elif savings_e <= 95000:
            return int(savings_e / 1500 + 1) * 1500
        elif savings_e <= 110000:
            return int(savings_e / 3000 + 1) * 3000
        else:
            return 110000

    def _utility_a(self, avg, t, _ua):
        cache_key = avg, t, _ua
        ret = self._utility_a_cache.get(cache_key)
        if ret is not None:
            return ret
        ret = _ua * self._disc_fa(self._choice_length_t[t]) + self._disc_fa2(self._choice_length_t[t]) * avg
        self._utility_a_cache[cache_key] = ret
        return ret

    def _utility_e(self, _ue, avg, choice_length_t):
        cache_key = _ue, avg, choice_length_t
        ret = self._utility_e_cache.get(cache_key)
        if ret is not None:
            return ret
        ret = _ue * self._disc_fa(choice_length_t) + self._disc_fa2(choice_length_t) * avg
        self._utility_e_cache[cache_key] = ret
        return ret

    def _disc_fa(self, t):
        cache_key = t
        ret = self._disc_fa_cache.get(cache_key)
        if ret is not None:
            return ret
        beta = self._discount_factor
        ret = (1 - beta ** t) / (1 - beta)
        self._disc_fa_cache[cache_key] = ret
        return ret

    def _disc_fa2(self, t):
        cache_key = t
        ret = self._disc_fa2_cache.get(cache_key)
        if ret is not None:
            return ret
        beta = self._discount_factor
        ret = beta ** t
        self._disc_fa2_cache[cache_key] = ret
        return ret

    def _utility_final_a(self, saving, edu, fame):
        cache_key = saving, edu, fame
        ret = self._utility_final_cache.get(cache_key)
        if ret is not None:
            return ret
        penalty = 1 + min(0, fame)
        saving = int(saving / 15.646 / 100) * 100
        _u1 = self._u1a(saving + self._a_1yr, self._table_leisure[0][0])
        _u3 = self._u3(self._state_edu_length_t[edu])
        ret = _u1 * _u3 * penalty * self._disc_fa(20)
        self._utility_final_cache[cache_key] = ret
        return ret

    def _utility_final_e(self, edu, new_death, choice_length_t):
        cache_key = edu, new_death, choice_length_t
        ret = self._utility_final_e_cache.get(cache_key)
        if ret is not None:
            return ret
        if not new_death:
            return 0
        if self._a_1yr >= 4000:
            wage = 4000
        elif self._a_1yr >= 2000:
            wage = 2000
        else:
            wage = 1000
        _u1 = self._u1e(wage, self._table_leisure[0][0])
        _u3 = self._u3(self._state_edu_length_t[edu])
        ret = _u1 * _u3 * self._disc_fa(choice_length_t)
        self._utility_final_e_cache[cache_key] = ret
        return ret

    def _ue(self, health, m_a, m_c, c_e_o, c_e, c_h, t):
        cache_key = health, m_a, m_c, c_e_o, c_e, c_h, t
        ret = self._ue_cache.get(cache_key)
        if ret is not None:
            return ret
        assert health >= -1
        if health > -1:
            health = 0
        u1 = self._u1e(c_e_o, self._table_leisure[0][m_a - m_c])
        u12 = u1 * self._u2(t, c_e)
        u123 = u12 * self._uhealth(health >= 0, c_h > 0)
        # assert u1 <= u12
        ret = u123
        self._ue_cache[cache_key] = ret
        return ret

    def _ua(self, c_a_o, m_a, m_c, c_e, t, fame):
        cache_key = c_a_o, m_a, m_c, c_e, t, fame
        ret = self._ua_cache.get(cache_key)
        if ret is not None:
            return ret
        penalty = 1 + min(0, fame)
        leisure = self._table_leisure[m_a][1 - (m_a - m_c) if t < 4 else 0]
        #        if m_a == 0 and ctt_a == 1:
        #            leisure -= 12 * 200
        if m_a == 1:
            c_a_o *= 0.154320988
        ret = self._u1a(c_a_o, leisure) * self._u2(t, c_e) * penalty
        self._ua_cache[cache_key] = ret
        return ret

    def _u1a(self, c, l):
        cache_key = c, l
        ret = self._u1a_cache.get(cache_key)
        if ret is not None:
            return ret
        cl = np.power(c, self._theta1) * np.power(l, 1 - self._theta1)
        cl = cl / 5000. + 1
        if self._gamma == 1:
            u1 = np.log(cl)
            assert u1 >= 0
            return u1
        u1 = self._isoelastic(cl, self._gamma)
        assert u1 >= 0
        ret = u1
        self._u1a_cache[cache_key] = ret
        return ret

    def _u1e(self, c, l):
        cache_key = c, l
        ret = self._u1e_cache.get(cache_key)
        if ret is not None:
            return ret
        cl = np.power(c, self._theta2) * np.power(l, 1 - self._theta2)
        cl = cl / 5000. + 1
        if self._gamma == 1:
            u1 = np.log(cl)
            assert u1 >= 0
            return u1
        u1 = self._isoelastic(cl, self._gamma)
        assert u1 >= 0
        ret = u1
        self._u1e_cache[cache_key] = ret
        return ret

    def _u2(self, t, c_edu):
        if c_edu == 0 and t in (1, 2):
            return self._lamda
        return 1

    def _u3(self, edu):
        cache_key = edu
        ret = self._u3_cache.get(cache_key)
        if ret is not None:
            return ret
        edu = max(1, edu)
        edu = edu * 1.
        ret = self._isoelastic(edu, self._phi) + 1
        self._u3_cache[cache_key] = ret
        return ret

    def _isoelastic(self, c, phi):
        cache_key = c, phi
        ret = self._isoelastic_cache.get(cache_key)
        if ret is not None:
            return ret
        ret = (np.power(c, 1 - phi) - 1) / (1 - phi)
        self._isoelastic_cache[cache_key] = ret
        return ret

    def _penalty(self, contract_hold, health, m_a, tr, fame):
        # no guilt if no contract or if elderly is healthy.
        # when contract holds and elderly is sick, penalty is imposed when (not coming back to rural area) + (not sending enough money)
        if health == -2:
            return fame
        if health == -1:
            return 0
        if not contract_hold:
            return 0

        rules_breaking = 0
        if tr < 2000 + self._contract_medical * self._c_h_1yr * (1 - self._gov_health):
            rules_breaking -= self._kappa1
        if m_a == 1:
            rules_breaking -= self._kappa2
        rules_breaking = min(rules_breaking, fame)

        if rules_breaking == 0:
            rules_breaking = 1
        return rules_breaking

    def _uhealth(self, is_sick, private_medical_cost):
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

    def simulation(self):
        ret = self.get_result()
        if len(ret) == 0:
            self.run(False)
            assert len(ret) != 0
        pr_t = np.asarray([6, 6, 3, 3, 2, 20], dtype=np.float64) / 20.
        assert abs(sum(pr_t) - 2) < 1.0e-8 and len(pr_t) == self._state_len
        pr_ret = []
        temp_pr_acc = 0.
        # [[(v_a, v_e, t, s, choice, p)]]]
        # s=(savings_a, edu, savings_e, health, ctt_a, migrate_sick, childcare, fame, loc)
        # choice=(m_a, m_c, c_a_o, c_edu, savings_a, transfer, c_e_o, c_h, savings_e, level)
        for i in ret:
            assert len(i) == self._state_len
            for j, k in enumerate(i):
                temp_pr = pr_t[j] * i[0][5]
                temp_pr_acc += temp_pr
                pr_ret.append((temp_pr, k[2], k[3], k[4]))

        logging.info('Simulation Pr   : \t' +
                     str([temp_pr_acc, np.infty,
                          self._gamma, self._theta1, self._theta2, self._lamda, self._phi, self._kappa1, self._kappa2,
                          self._rho_migrate_sick, self._utility_treated, self._disutility_sick, self._discount_factor, np.infty,
                          self._a_1yr, self._ctt_e, self._key])[1:-1])
        assert abs(temp_pr_acc - 2.) < 1.0e-8
        return pr_ret


def _test(parameters):
    import cProfile
    import io
    import pstats
    f = Full(*parameters, concurrent=False)
    pr = cProfile.Profile()
    pr.enable()
    f.run()
    pr.disable()
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    f.output()


def _test_concurrent(parameters):
    f = Full(*parameters)
    f.run()
    f.output()
    print(f.simulation())


def _test_concurrent2(parameters):
    f_list = []
    for i in range(16):
        f = Full(*parameters, concurrent=False)
        f_list.append(f)
    pool = PPool()
    for i in f_list:
        pool.apply_async(i.run)
    pool.close()
    pool.join()
    for i in f_list:
        i.run()
        i.output()


def _init_log():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root.addHandler(ch)


if __name__ == '__main__':
    'Do Not Use'
