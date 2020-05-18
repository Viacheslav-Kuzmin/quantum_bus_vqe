import importlib
import sim_lib_pylib
importlib.reload(sim_lib_pylib)
from sim_lib_pylib import *
class ResultMPS(object):
    
    def __init__( self, simulator, res ):
        self.simulator, self.res = simulator, res
#         self.E = self.simulator.model.H_cost_corr.get_energy()
        self.E = res.fun
    def print_results(self):
        print(self.get_message_results())
    
    def get_message_results(self):

        simulator = self.simulator
        opt_props = simulator.opt_props
        system = simulator.system
        model = simulator.model
        
        E_max_str = "%.2f" % model.E_max if model.E_max is not None else 'N/a'
        E_min_str = "%.2f" % model.E_min if model.E_min is not None else 'N/a'
     
        message =  "State:          " + model.TAG + '\n' +\
        "N sys., N aux.: " + str(system.N_sys) + ', ' + str(system.N_aux) + '\n' +\
        "N layers:       " + str(simulator.N_layers) + '\n' +\
        "N params:       " + str(simulator.N_free_params) + '\n' +\
        "Tol. rel.:      " + str(opt_props.tol_rel) + '\%\n' +\
        "Method:         " + str(opt_props.method) + '\n' +\
        "nfev, nit:      " + str(self.res.nfev) + ', '+ str(self.res.nit) + '\n' +\
        "E max.:         " + E_max_str +'\n' +\
        "E min:          " + E_min_str +'\n' +\
        "E:              " + "%.2f" % self.E
        
        return message
class ModelMPS(object):
    
    E_max, E_min = None, None
    
    def __init__( self, TAG, N_sys, args = None):
        self.TAG = TAG
        self.N_sys = N_sys
        self.args = args
        
        if TAG == TAG_SSH_P:
            t_p, t_m, B = 1, 0, 0.1
            if args is not None:
                t_p, t_m, B = args
            self.H_cost_corr = get_SSH_H_cor_3(N_sys, t_p=t_p, t_m=t_m, B = B)
        elif TAG == TAG_SCHWING:
            J, w, m, e0 = 1, 1, -0.5, 0
            if args is not None:
                J, w, m, e0 = args
#             self.H_cost_corr = get_schw_H_cor_3(N_sys, J, w, m, e0)
            self.H_cost_corr = get_schw_H_cor_3_revers(N_sys, J, w, m, e0)
        elif TAG == TAG_MG:
            self.H_cost_corr = get_MG_H_cor(N_sys)
        elif TAG == TAG_GHZ_ANTIFERR:
            self.H_cost_corr = get_GHZ_antifer_H_cor(N_sys)
        elif TAG == TAG_W:
            self.H_cost_corr = get_W_H_cor(N_sys)
        elif TAG == TAG_CLUST:
            b = 0
            if args is not None:
                b = args[0]
            self.H_cost_corr = get_clust_H_cor(N_sys, b)
            
        elif TAG == TAG_AKLT:
            self.H_cost_corr = get_squeez_H_corr(N_sys)
            l, b = 0, 0
            if args is not None:
                l, b = args
            self.H_cost_corr = get_AKLT_H_corr(N_sys, l, b)
            
        elif TAG == TAG_HALD_IONS:
#                 H_cost = get_H_Hald_qubits(N_sys, trans_map_ion)
            J, Delt, D2, l = 1, 1, 1, 0
            if args is not None:
                J, Delt, D2, l = args
            self.H_cost_corr = get_H_Hald_qub_cor(N_sys, J, Delt, D2, l)
            
        else:
            raise NotImplementedError
class SimulatorMPS(Simulator):
    '''
    Object for simulation by preparing a Kraus map and applying it N_iter to the initial state.
    
    Parameters
    ----------
    kraus_gen_list : List of KrausGenerator objects which are applied in the corresponded order. 
        After each KrausGenerator the ancillas are resetted to the initial state.
    N_iter : number of iterations of application of kraus_gen_list before the cost function is measured.
    N_layers : int or None.
        Do not partisipate in the simulation, but is shown in the print of results. 
        If None, -> total number of used gates.
    '''

    def __init__( self, system_mps, opt_props, model_mps, params_init, gates_seq_list, blocks_order_list, N_layers = None, params_map = None):
        
        self.system = system_mps
        self.opt_props = opt_props
        self.params_init = params_init
        self.model = model_mps
        self.params_map = params_map    
        
        self.set_gates_sequences(gates_seq_list, N_layers)
        self.set_params_init(params_init) 
        self.blocks_order_list = blocks_order_list
    
    def set_gates_sequences(self, gates_seq_list, N_layers = None):
        for gates_seq in gates_seq_list:
            gates_seq.prepare()            
        self.gates_seq_list = gates_seq_list
        
        # Parameters of all gates
        self.N_free_params = sum( seq.N_free_params for seq in gates_seq_list )
        self.N_params = sum( seq.N_params for seq in gates_seq_list )
        
        if N_layers is None: 
            N_layers = [seq.N_gates for seq in gates_seq_list]
        self.N_layers = N_layers
        
    def simulation(self, params):

        E_list, list_of_E_var_lists = self.obtain_average([self.model.H_cost_corr], params, self.opt_props.jac)

        E, dE_list = np.real(E_list[0]), np.real(list_of_E_var_lists[0])
        return SimResult(E, 0, [], [], [], [],0,dE_list)
    
    def simulation_test(self, param):
        return self.simulation([param]*self.N_params)
    
    def obtain_average(self, H_list, params, jac = False):
        
        if not isinstance(H_list, list):
            H_list = [H_list]
        
        self.set_params_to_gates_seqs(params)
        
        # Should be changed to logical space in the future
        aux_inds = [a[0] for a in self.system.aux_mode_inds]
        
        for kk, H in enumerate(H_list):
            if not isinstance(H,numbers.Number):
                H.reset()
                H.set_aux_inds(aux_inds, self.system.N_sys)
                if isinstance(H,Correlator):
                    H_list[kk] = CorrelatorsList([H])
            else:
                H_list[kk] = CorrelatorsList([H])
                
        
        state_contractor_rep = StateContractorRepository(self.system, H_list)
        state_contractor_rep_var_list = [None]*len(params)
        
        for block_ind, blocks_order in enumerate(self.blocks_order_list):
            gates_seq_ind, blocks_inds, is_dag = blocks_order
            # Indices of the parameters in 'params' which are used in the iteration
            seq_params_inds = self.seq_params_inds_list[gates_seq_ind]
            # Indices which are not participate in the evolution after the current iteration
            state_inds_ready = self.get_state_inds_ready(block_ind)
            
            gates_seq = self.gates_seq_list[gates_seq_ind]
            
            U = gates_seq.A
            U_dag = gates_seq.A_dag
            if is_dag: 
                U, U_dag = U_dag, U
                seq_params_inds = seq_params_inds[::-1]
            
            if jac:
                U_var_list = gates_seq.A_var_list
                U_dag_var_list = gates_seq.A_dag_var_list
                if is_dag: 
                    U_var_list, U_dag_var_list = U_dag_var_list[::-1], U_var_list[::-1]

                for i, ind in enumerate(seq_params_inds):
                    if state_contractor_rep_var_list[ind] is None:
                        state_contractor_rep_var_list[ind] = state_contractor_rep.get_copy()
                        
                    state_contractor_rep_var_list[ind].contract(U_var_list[i], U_dag_var_list[i], 
                                                                blocks_inds, state_inds_ready)
            
                # Proceed existed variations
                for kk, s in enumerate(state_contractor_rep_var_list):
                    if kk not in seq_params_inds and s is not None:
                        s.contract(U, U_dag, blocks_inds, state_inds_ready)
            
            state_contractor_rep.contract(U, U_dag, blocks_inds, state_inds_ready)

        E_list = [H.get_energy() for H in H_list]
        if jac:
            list_of_E_var_lists = []
            for kk in range(len(H_list)):
                H_cost_corr_var_list = [s.list_of_correlators_list[kk] for s in state_contractor_rep_var_list]
                E_var_list = [H.get_energy() for H in H_cost_corr_var_list]
                E = E_list[kk]
                dE_list = np.array([(E_var-E)/eps for E_var in E_var_list])
                list_of_E_var_lists += [dE_list]
        else:
            list_of_E_var_lists = [0]

        return E_list, list_of_E_var_lists
            
    
    def trace_ready_modes(self, state, state_inds_G, modes_ready_inds):
        
        inds_to_leave = [i for i, ind in enumerate(state_inds_G) if ind not in modes_ready_inds]

        return state.ptrace(inds_to_leave), [state_inds_G[i] for i in inds_to_leave]
        
    
    def get_state_inds_ready(self, block_ind):
        modes_affected = []
        modes_remind = []
        
        for i in range(block_ind+1):
            gates_seq_ind, gates_inds_G, _ = self.blocks_order_list[i]
            gates_seq = self.gates_seq_list[gates_seq_ind]

            modes_affected += list(gates_inds_G)
            
        for i in range(block_ind+1, len(self.blocks_order_list) ):
            gates_seq_ind, gates_inds_G, _ = self.blocks_order_list[i]
            gates_seq = self.gates_seq_list[gates_seq_ind]
            
            modes_remind += list(gates_inds_G)
        
        modes_affected = list(set(modes_affected))
        modes_remind = list(set(modes_remind))
        
        modes_ready_inds = [m for m in modes_affected if m not in modes_remind]
        
        return modes_ready_inds
    
    def set_params_to_gates_seqs(self, params):

        # Indices of parameters of gates_seq-s
        seq_params_inds_list = []
        
        param_ind = 0
        for gates_seq in self.gates_seq_list:
            param_ind_new = param_ind + gates_seq.N_free_params
            params_to_set = params[param_ind : param_ind_new]
            seq_params_inds_list += [list(range(param_ind, param_ind_new))]
            gates_seq.gen_unitary_dens(params_to_set, self.opt_props.jac)
            
            param_ind = param_ind_new
        
        self.seq_params_inds_list = seq_params_inds_list
        
    def to_simulator(self):
        
        N_THRESHOLD = 16
        if self.system.N_sys + self.system.N_aux > N_THRESHOLD:
            raise Exception('To big dimension.')
        
        model = Model(self.model.TAG, self.model.N_sys, args = self.model.args)
        
        gates_list = []
        for blocks_order in self.blocks_order_list:
            gates_seq = self.gates_seq_list[blocks_order[0]]
            gates_list_0 = deepcopy(gates_seq.gates_list)
            gates_list += set_block_to_full_inds(blocks_order[1], gates_list_0)
        
        params_map = params_map_out_of_blocks(self.blocks_order_list, self.gates_seq_list)
        return SimulatorCoherent(self.system, self.opt_props, model, 
                                 GatesSequence(gates_list, params_map = params_map), params_init = self.params_init)
    
    def optimize_params(self):
        res = optimize_params(self)
        return ResultMPS( self, res )
# class SimulatorMPS(Simulator):
#     '''
#     Object for simulation by preparing a Kraus map and applying it N_iter to the initial state.
    
#     Parameters
#     ----------
#     kraus_gen_list : List of KrausGenerator objects which are applied in the corresponded order. 
#         After each KrausGenerator the ancillas are resetted to the initial state.
#     N_iter : number of iterations of application of kraus_gen_list before the cost function is measured.
#     N_layers : int or None.
#         Do not partisipate in the simulation, but is shown in the print of results. 
#         If None, -> total number of used gates.
#     '''
    
#     def __init__( self, system_mps, opt_props, model_mps, params_init, gates_seq_list, blocks_order_list, N_layers = None, params_map = None):
        
#         self.system = system_mps
#         self.opt_props = opt_props
#         self.params_init = params_init
#         self.model = model_mps
#         self.params_map = params_map    
        
#         self.set_gates_sequences(gates_seq_list, N_layers)
#         self.set_params_init(params_init) 
#         self.blocks_order_list = blocks_order_list
    
#     def set_gates_sequences(self, gates_seq_list, N_layers = None):
#         for gates_seq in gates_seq_list:
#             gates_seq.prepare()            
#         self.gates_seq_list = gates_seq_list
        
#         # Parameters of all gates
#         self.N_free_params = sum( seq.N_free_params for seq in gates_seq_list )
#         self.N_params = sum( seq.N_params for seq in gates_seq_list )
        
#         if N_layers is None: 
#             N_layers = [seq.N_gates for seq in gates_seq_list]
#         self.N_layers = N_layers
        
#     def simulation(self, params):

#         E_list, list_of_E_var_lists = self.obtain_average([self.model.H_cost_corr], params, self.opt_props.jac)

#         E, dE_list = np.real(E_list[0]), np.real(list_of_E_var_lists[0])
#         return SimResult(E, 0, [], [], [], [],0,dE_list)
    
#     def simulation_test(self, param):
#         return self.simulation([param]*self.N_params)
    
#     def obtain_average(self, H_list, params, jac = False):
        
#         if not isinstance(H_list, list):
#             H_list = [H_list]
        
#         self.set_params_to_gates_seqs(params)
        
#         # Should be changed to logical space in the future
#         aux_inds = [a[0] for a in self.system.aux_mode_inds]
        
#         for kk, H in enumerate(H_list):
#             if not isinstance(H,numbers.Number):
#                 H.reset()
#                 H.set_aux_inds(aux_inds)
#                 if isinstance(H,Correlator):
#                     H_list[kk] = CorrelatorsList([H])
#             else:
#                 H_list[kk] = CorrelatorsList([H])
                
        
#         state_contractor_rep = StateContractorRepository(self.system, H_list)
        
#         for block_ind, blocks_order in enumerate(self.blocks_order_list):
#             gates_seq_ind, blocks_inds, is_dag = blocks_order
#             # Indices of the parameters in 'params' which are used in the iteration
#             seq_params_inds = self.seq_params_inds_list[gates_seq_ind]
#             # Indices which are not participate in the evolution after the current iteration
#             state_inds_ready = self.get_state_inds_ready(block_ind)
            
#             gates_seq = self.gates_seq_list[gates_seq_ind]
            
#             U = gates_seq.A
#             U_dag = gates_seq.A_dag
#             if is_dag: 
#                 U, U_dag = U_dag, U
#                 seq_params_inds = seq_params_inds[::-1]
            
#             if jac:
#                 U_var_list = gates_seq.A_var_list
#                 U_dag_var_list = gates_seq.A_dag_var_list
#                 if is_dag: 
#                     U_var_list, U_dag_var_list = U_dag_var_list[::-1], U_var_list[::-1]

#                 for i, ind in enumerate(seq_params_inds):
#                     state_contractor_rep.contract(U_var_list[i], U_dag_var_list[i], 
#                                                                 blocks_inds, state_inds_ready, ind+1)
            
#             # Proceed existed variations and the original cost function
#             state_contractor_rep.contract(U, U_dag, 
#                                           blocks_inds, state_inds_ready)

#         E_list = [H.get_energy() for H in H_list]
#         if jac:
#             list_of_E_var_lists = []
#             for kk in range(len(H_list)):
#                 H_cost_corr_var_list = [s.list_of_correlators_list[kk] for s in state_contractor_rep_var_list]
#                 E_var_list = [H.get_energy() for H in H_cost_corr_var_list]
#                 E = E_list[kk]
#                 dE_list = np.array([(E_var-E)/eps for E_var in E_var_list])
#                 list_of_E_var_lists += [dE_list]
#         else:
#             list_of_E_var_lists = [0]

#         return E_list, list_of_E_var_lists
            
    
#     def trace_ready_modes(self, state, state_inds_G, modes_ready_inds):
        
#         inds_to_leave = [i for i, ind in enumerate(state_inds_G) if ind not in modes_ready_inds]

#         return state.ptrace(inds_to_leave), [state_inds_G[i] for i in inds_to_leave]
        
    
#     def get_state_inds_ready(self, block_ind):
#         modes_affected = []
#         modes_remind = []
        
#         for i in range(block_ind+1):
#             gates_seq_ind, gates_inds_G, _ = self.blocks_order_list[i]
#             gates_seq = self.gates_seq_list[gates_seq_ind]

#             modes_affected += list(gates_inds_G)
            
#         for i in range(block_ind+1, len(self.blocks_order_list) ):
#             gates_seq_ind, gates_inds_G, _ = self.blocks_order_list[i]
#             gates_seq = self.gates_seq_list[gates_seq_ind]
            
#             modes_remind += list(gates_inds_G)
        
#         modes_affected = list(set(modes_affected))
#         modes_remind = list(set(modes_remind))
        
#         modes_ready_inds = [m for m in modes_affected if m not in modes_remind]
        
#         return modes_ready_inds
    
#     def set_params_to_gates_seqs(self, params):

#         # Indices of parameters of gates_seq-s
#         seq_params_inds_list = []
        
#         param_ind = 0
#         for gates_seq in self.gates_seq_list:
#             param_ind_new = param_ind + gates_seq.N_free_params
#             params_to_set = params[param_ind : param_ind_new]
#             seq_params_inds_list += [list(range(param_ind, param_ind_new))]
#             gates_seq.gen_unitary_dens(params_to_set, self.opt_props.jac)
            
#             param_ind = param_ind_new
        
#         self.seq_params_inds_list = seq_params_inds_list
        
#     def to_simulator(self):
        
#         N_THRESHOLD = 16
#         if self.system.N_sys + self.system.N_aux > N_THRESHOLD:
#             raise Exception('To big dimension.')
        
#         model = Model(self.model.TAG, self.model.N_sys, args = self.model.args)
        
#         gates_list = []
#         for blocks_order in self.blocks_order_list:
#             gates_seq = self.gates_seq_list[blocks_order[0]]
#             gates_list_0 = deepcopy(gates_seq.gates_list)
#             gates_list += set_block_to_full_inds(blocks_order[1], gates_list_0)
        
#         params_map = params_map_out_of_blocks(self.blocks_order_list, self.gates_seq_list)
#         return SimulatorCoherent(self.system, self.opt_props, model, 
#                                  GatesSequence(gates_list, params_map = params_map), params_init = self.params_init)
def params_map_out_of_blocks(blocks_order_list, gates_seq_list):
    
    func = lambda x: x[0]
    map_list = []
    seq_params_inds = []
    unique_block_inds = []
    n_params = 0
    for seq_ind, gate_inds, is_dag in blocks_order_list:
        if seq_ind not in unique_block_inds:
            n_seq_params = gates_seq_list[seq_ind].N_free_params
            unique_block_inds += [seq_ind]
            seq_params_inds += [list(range(n_params, n_params + n_seq_params))]
            n_params += n_seq_params
        else:
            inds = seq_params_inds[unique_block_inds.index(seq_ind)]
            map_list += [[n_params + i, [p], func] for i, p in enumerate(inds)]
            n_params += len(inds)

#     unique_block_inds_check = deepcopy(unique_block_inds)
#     for seq_ind, gate_inds in blocks_order_list:
#         if seq_ind in unique_block_inds_check:
#             unique_block_inds_check.remove(seq_ind)
#         else:
#             inds = seq_params_inds[unique_block_inds.index(seq_ind)]
#             map_list += [[n_params + i, [p], func] for i, p in enumerate(inds)]
#             n_params += len(inds)
    return ParamsMap(map_list)

def set_block_to_full_inds(block_inds, gates_list):
    inds_all = []
    for gate in gates_list:
        inds_to_act_list = gate.inds_to_act_list
        for inds_to_act in inds_to_act_list:
            inds_all += inds_to_act

    gates_inds = list(set(inds_all))
    
    for gate in gates_list:
        inds_to_act_list = gate.inds_to_act_list
        inds_to_act_list_new = []
        for inds_to_act in inds_to_act_list:
            inds_to_act_list_new += [[block_inds[gates_inds.index(i)] for i in inds_to_act]]
        gate.inds_to_act_list = inds_to_act_list_new
    
    return gates_list


class StateContractor(object):

    # Qobj
    data = None
    # Indises of the full state represented by data
    state_inds = []
    # Contracted indices
    contract_inds = []
    # Objects with which the states indices where contracted
    contract_obj_names = []
    evolved = False
    
    def __init__( self, data = None, state_inds = [], contract_obj_names = [], contract_inds = []):
        self.data, self.contract_obj_names, self.contract_inds, self.state_inds = \
        data, contract_obj_names, contract_inds, state_inds
    
    def evolve(self, system, U, U_dag, U_inds):
        
        self.data, self.state_inds = system.complete_state_mq(self.data, self.state_inds, U_inds, dense = True)
#         U_dag = U.dag()
        data = U * self.data
        self.data = data * U_dag
        self.evolved = True
        
#     def is_approp(self, names_list):
#         for k, name in enumerate(self.contract_obj_names):
#             ind = self.contract_inds[k]
#             if ind < len(names_list) and name is not names_list[ind]:
#                 return False
#         return True
    
    def is_approp(self, names_list):
        
        for k, name in enumerate(self.contract_obj_names):
            ind = self.contract_inds[k]
            if ind < len(names_list): 
                if name is not names_list[ind]:
                    return False
            else:
                if name is not I_name:
                    return False
                
        return True
        
    def contract(self, correlator, state_inds_ready):
     
        inds_to_contract = [i for i in state_inds_ready if i in self.state_inds]
        is_purity, oper_list, names, inds_to_contract, is_enough = correlator.get_correlator_part(inds_to_contract) 

        if is_enough:
            inds_to_contract = copy(self.state_inds)
            is_purity, oper_list, names, inds_to_contract, is_enough = correlator.get_correlator_part(inds_to_contract) 

        C_list = [o.data for o in oper_list]
        # Trace where names == 'I'
        ind_I =[ind for i, ind in enumerate(inds_to_contract) if names[i]==I_name]
        ind_C =[ind for i, ind in enumerate(inds_to_contract) if names[i]!=I_name]
        pos_I = [i for i, ind in enumerate(self.state_inds) if ind in ind_I]
        pos_C = [i for i, ind in enumerate(self.state_inds) if ind in ind_C]
        
        pos_I_2 = [i for i, ind in enumerate(self.state_inds) if ind in ind_I]
        pos_C_2 = [i for i, ind in enumerate(self.state_inds) if ind in ind_C]
        
        data, dims_B = contract(self.data, pos_I, C_list, pos_C )

        
        if len(data) == 1:
            state_inds = []
        else:
            state_inds = [ind for i, ind in enumerate(self.state_inds) if ind not in inds_to_contract]
        
        if is_enough:
            if is_purity:
                correlator.value = data
#                 a = destroy(Qobj(data_g).dims[0][0])
#                 n = a.dag()*a
#                 nn = n*n
#                 rho = Qobj(data)
#                 n_val = (n*rho).tr()
#                 nn_val = (nn*rho).tr()
#                 var = nn_val - n_val**2
#                 correlator.value = [n_val, var]
            else:
                correlator.value = data[0,0]
            return None
        else:
            data = MyQobj(data, dims = dims_B)
            # Add contracted operators
            contract_obj_names = self.contract_obj_names + names        
            contract_inds = self.contract_inds + inds_to_contract
            return StateContractor(data, state_inds, contract_obj_names, contract_inds)
        
        
        
        
        

#     def contract(self, correlator, state_inds_ready):
        
#         inds_to_contract = [i for i in state_inds_ready if i in self.state_inds]

#         oper, names, is_enough = correlator.get_correlator_part(inds_to_contract) 
# #         if ID == id(correlator):
# #             print(correlator)
# #             print(self.contract_obj_names)            
#         # Trace where names == 'I'
#         inds_to_contract_I = [ind for i, ind in enumerate(inds_to_contract) if i<len(names) and names[i]=='I']
#         inds_to_contract_oper = [ind for i, ind in enumerate(inds_to_contract) if i<len(names) and names[i]!='I']
        
#         if is_enough:
#             pos_to_remain_I = [i for i, ind in enumerate(self.state_inds) if ind in inds_to_contract_oper]
#             state_inds = inds_to_contract_oper
#         else:
#             pos_to_remain_I = [i for i, ind in enumerate(self.state_inds) if ind not in inds_to_contract_I]
#             state_inds = [ind for i, ind in enumerate(self.state_inds) if ind not in inds_to_contract_I]

#         data = ptrace_mq(self.data, pos_to_remain_I)
        
        
#         # Contract with oper
#         pos_to_act = [ i for i, ind in enumerate(state_inds) if ind in inds_to_contract_oper]
#         oper = make_multimode_H_mq(oper, pos_to_act, data.dims)
#         pos_to_remain = [i for i, ind in enumerate(state_inds) if ind not in inds_to_contract_oper]

#         if len(pos_to_remain) == 0:
#             data = (oper * data).tr()
#             state_inds = []
#         else:
#             data = ptrace_mq(oper * data, pos_to_remain)
#             state_inds = [ind for i, ind in enumerate(state_inds) if ind not in inds_to_contract_oper]
        
#         if is_enough:
#             correlator.value = data
#             return None
#         else:
#             # Add contracted operators
#             contract_obj_names = self.contract_obj_names + names        
#             contract_inds = self.contract_inds + inds_to_contract
#             return StateContractor(data, state_inds, contract_obj_names, contract_inds)


def contract(A, pos_I, C_list, pos_C ):
    
    if isinstance(pos_I, list):
        pos_I = np.array(pos_I, dtype = np.int32)
    if isinstance(pos_C, list):
        pos_C = np.array(pos_C, dtype = np.int32)

    dims_A = np.array(A.dims, dtype = np.int32)
    A_data = A.data
    
    N_C = len(C_list)
    if N_C == 0:
        C_list_arr = np.array([[[]]], dtype = complex)
    else:
        C_shape = [N_C]+list(C_list[0].shape)
        C_list_arr = np.empty(C_shape, dtype = complex, order='C')
        for i in range(N_C):
            C_list_arr[i] = C_list[i].data

    data, dims_B = cy_contract(A_data, dims_A, pos_I, C_list_arr , pos_C)
    dims_B = list(dims_B)
        
    return data, dims_B
class StateContractorRepository(object):
    
    # List of StateContracor objects
    states_contr_list = []
    
    def __init__( self, system, list_of_correlators_list, states_contr_list = None):
        
        if not isinstance(list_of_correlators_list, list):
            list_of_correlators_list = [list_of_correlators_list]
        
        if states_contr_list is None:
            self.states_contr_list = [StateContractor()]
        else:
            self.states_contr_list = states_contr_list
            
        self.system, self.list_of_correlators_list = system, list_of_correlators_list
        
        # collection of all correlators in list_of_correlators_list
        correlators_list = []
        for c in list_of_correlators_list:
            correlators_list += c.correlators_list
            
        self.correlators_list = correlators_list
    
    def contract(self, U, U_dag, U_inds, state_inds_ready):
        system, correlators_list = self.system, self.correlators_list
        
        states_contr_list = self.states_contr_list
        new_states_contr_list = []
        old_to_new_contract_map = [[] for i in range(len(states_contr_list))]
        
        for correlator in correlators_list:
            
            if isinstance(correlator, numbers.Number) or correlator.value is not None:
                continue
                
            # Chech if there is already appropriate in the list of new contracted states
            contract_ind = correlator.contract_ind
            
            ind_of_new = None
            new_contr_inds = old_to_new_contract_map[contract_ind]
            if len(new_contr_inds) > 0:
                new_contr_list_to_check = [new_states_contr_list[i] for i in new_contr_inds]
                ind_of_new = self.get_approp_contractor(new_contr_list_to_check, correlator) 

            if ind_of_new is None:
                state_contr = states_contr_list[contract_ind]
                # Evolve if it is not evolved
                if not state_contr.evolved:
                    state_contr.evolve(system, U, U_dag, U_inds)

                state_contr_new = state_contr.contract(correlator, state_inds_ready)
                if state_contr_new is not None:
                    new_states_contr_list += [state_contr_new]
                    new_ind = len(new_states_contr_list)-1
                    old_to_new_contract_map[contract_ind] += [new_ind]
                    correlator.contract_ind = new_ind
            else:
                correlator.contract_ind = new_contr_inds[ind_of_new]
        
#         if len(old_to_new_contract_map[-1])>1e3:
#             save_to_g(old_to_new_contract_map)
#             sdsds
        self.states_contr_list = new_states_contr_list
                
    def get_approp_contractor(self, states_contr_list, correlator):
#         print(len(states_contr_list))
        for ind, state_contr in enumerate(states_contr_list):
            if state_contr.is_approp(correlator.state_names_list):
                return ind
            
        return None
    
    def get_copy(self):
        list_of_correlators_list_copy = [l.get_copy() for l in self.list_of_correlators_list]
        states_contr_list_copy = deepcopy(self.states_contr_list)
        return StateContractorRepository( self.system, list_of_correlators_list_copy, states_contr_list_copy )
class CorrelatorsList(object):
    
    def get_copy(self):
        correlators_list = [c if isinstance(c, numbers.Number) else c.get_copy() for c in self.correlators_list]
        return CorrelatorsList(correlators_list)
    
    def __init__( self, correlators_list = [] ):
        if not isinstance(correlators_list, list):
            correlators_list = [correlators_list]

            
        self.correlators_list = correlators_list
    

    def full(self, N_modes = 0):
        dims = 2
        I = identity(dims)
        correlators_list = self.correlators_list
        
        N_modes_max = max([len(correlator.names_list) for correlator in correlators_list if not isinstance(correlator, numbers.Number)])
        N_modes_max = max(N_modes_max, N_modes)
        op_full = 0
        for correlator in correlators_list:
            if isinstance(correlator, numbers.Number):
                op_full += correlator
            else:
                op_full += correlator.full(N_modes_max)
#                 op_list = []
#                 for name in correlator.names_list:
#                     if name != I_name:
#                         op_list += [correlator.name2data(name, is_qobj = True)]
#                     else:
#                         op_list += [I]
#                 op_full += correlator.coef * tensor(op_list + [I]*(N_modes_max - len(correlator.names_list)))
        return op_full
        
    def __radd__(self, B):
        return self.__add__(B)
        
    def __add__(self, B):
        
        if B is 0:
            return self
        
        if isinstance(B, Correlator):
            return B + self
        if isinstance(B, numbers.Number):
            
            C_list = []
            is_added = False
            for c in self.correlators_list:
                if not is_added and isinstance(c, numbers.Number):
                    C_list += [c + B]
                    is_added = True
                else:
                    if c!= 0 and c.coef != 0:
                        C_list += [c]
            
            if not is_added:
                C_list = [B] + C_list
            return(CorrelatorsList(C_list))
                
        else:
            B_list = B.correlators_list
            C_out = self
            for b in B_list:
                C_out = C_out + b
            return C_out
    
    def __rmul__(self, B):
        if isinstance(B, numbers.Number):
            return self.__mul__( B)
        else:
            raise NotImplementedError
            
        
    def __mul__(self, B):
            
        if isinstance(B, Correlator) or isinstance(B, numbers.Number):
            c_list_out = sum(c * B for c in self.correlators_list)
        else:
            c_list_out = sum(c * B for c in self.correlators_list)
            
        if c_list_out == 0:
            return 0
        elif isinstance(c_list_out, Correlator):
            return c_list_out
        elif len(c_list_out.correlators_list) == 1:
            return c_list_out.correlators_list[0]
        else:
            return c_list_out
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        string_out = '['
        for c in self.correlators_list:
            string_out +='\n'
            string_out += str(c) + '\n,'
        string_out = string_out[:-1]
        string_out += ']'
        return string_out
    
    def __pow__(self, other):
        if other is 0:
            return 1
        
        res = self
        for i in range(1,other):
            res = res*self
        return res
    
    def get_energy(self):
        
        E = 0
        for correlator in self.correlators_list:
            if isinstance(correlator, numbers.Number):
                E += correlator
            else:
                if correlator.value is None:
                    raise Exception('Not all correlators are calculated')
                E += correlator.coef * correlator.value

        return np.real(E)
    
    def reset(self):
        self.active_inds = None
        
        for correlator in self.correlators_list:
            if not isinstance(correlator, numbers.Number):
                correlator.reset()
    
    def set_aux_inds(self, aux_inds, N_sys):
        for correlator in self.correlators_list:
            if not isinstance(correlator, numbers.Number):
                correlator.set_aux_inds(aux_inds, N_sys)
    
    def dag(self):
        return CorrelatorsList([c if isinstance(c, numbers.Number) else c.dag() for c in self.correlators_list])
    
    
    
    def get_full_H(self):
        H = 0
        H_dims = self.get_H_dims()
        for correlator in self.correlators_list:
            H_list = correlator.get_H_list()
            coef = correlator.coef
            inds_to_act = correlator.sys_inds_to_act
            H += make_multimode_H(coef*tensor(H_list), inds_to_act, H_dims)

        return H
    
    def get_H_dims(self):
        all_inds = []
        all_dims = []
        for correlator in self.correlators_list:
            if isinstance(correlator, numbers.Number):
                continue
            
            all_inds += correlator.sys_inds_to_act
            H_list = correlator.get_H_list()
            for H in H_list:
                all_dims += get_q_dim_list(H)
        
        H_inds = sorted(list(set(all_inds)))
        H_dims = [all_dims[all_inds.index(ind)] for ind in H_inds]
#         N_sys = len(H_dims)
        return H_dims
class Correlator(object):
    
    value = None
    multiplicator = None
    aux_inds = []
    contract_ind = 0
#     state_names_list = []
#     inds_list_state = []
    
    def get_copy(self):
#         return deepcopy(self)
        cor_copy = Correlator(self.coef, self.op_names_list, self.sys_inds_to_act, 
                          self.name2data_interpretator, self.multiplicator, self.dag_map)
        cor_copy.state_names_list = self.state_names_list
        cor_copy.inds_list_state = self.inds_list_state
        cor_copy.aux_inds = self.aux_inds
#         cor_copy.set_aux_inds(self.aux_inds)
        cor_copy.contract_ind = copy(self.contract_ind)
        cor_copy.value = copy(self.value)
        return cor_copy
    
    def __init__( self, coef, op_names_list, sys_inds_to_act, name2data_interpretator, multiplicator = None, dag_map = None, aux_inds_to_act = [], aux_op_names_list = [] ):
        
        if len(op_names_list) != len(sys_inds_to_act):
            raise ValueError('len(op_names_list) != len(sys_inds_to_act)')
            
        self.coef = coef
        self.multiplicator = multiplicator
        self.name2data_interpretator = name2data_interpretator
        self.dag_map = dag_map
        
        self.aux_inds_to_act = aux_inds_to_act
        self.aux_op_names_list = aux_op_names_list
        if len(aux_inds_to_act)!=0:
            sys_inds_to_act = aux_inds_to_act
            op_names_list = aux_op_names_list
        
        op_names_list = [x for _,x in sorted(zip(
            sys_inds_to_act, op_names_list ))
                    ]
        sys_inds_to_act = sorted(sys_inds_to_act)
        
        self.op_names_list = op_names_list
        self.sys_inds_to_act = sys_inds_to_act
        
        # prepare list of the operators names
        names_list = []
        ind_last = 0
        for k, ind in enumerate(sys_inds_to_act):
            names_list += [I_name] * (ind - ind_last) + [op_names_list[k]]
            ind_last = ind+1
            
        self.names_list = names_list
        
    def __radd__ (self, B):
        return self.__add__(B)
    
    def __add__(self, B):
        
        if B is 0:
            return self
        
        if isinstance(B, CorrelatorsList):
            B_list = B.correlators_list
        else:
            B_list = [B]

        C_list = []
        is_added = False
        for B in B_list:
            if not is_added:
                is_added, C_out = self.add_singl( B)
            else:
                C_out = B
               
            if isinstance(C_out, numbers.Number) and C_out!= 0:
                C_list += [C_out]
            elif C_out!= 0 and C_out.coef != 0:
                C_list += [C_out]
        
        if not is_added and self.coef != 0:
            C_list = [self] + C_list
        
        if len(C_list) == 0:
            return 0
        elif len(C_list) == 1:
            return C_list[0]
        else:
            return CorrelatorsList(C_list)
        
    
    def add_singl(self, B):
        if isinstance(B, numbers.Number):
            return False, B
        
        if self.sys_inds_to_act != B.sys_inds_to_act:
            return False, B
        
        if self.op_names_list == B.op_names_list:
            coef_out = self.coef + B.coef
            if coef_out == 0:
                return True, 0
            else:
                return True, Correlator( coef_out, B.op_names_list, B.sys_inds_to_act, 
                                    B.name2data_interpretator, B.multiplicator, B.dag_map)
        
        return False, B
        
    
    def __rmul__(self, B):
        if isinstance(B, numbers.Number):
            return self.__mul__( B)
        else:
            raise NotImplementedError
            
    def __mul__(self, B):
        
        if isinstance(B, CorrelatorsList):
            c_list_out = CorrelatorsList()
            for b in B.correlators_list:
                c_list_out += self.mul_singl(b)
                
            if c_list_out == 0:
                res =  0
            elif isinstance(c_list_out, Correlator):
                res = c_list_out
            elif len(c_list_out.correlators_list) == 1:
                res =  c_list_out.correlators_list[0]
            else:
                res =  c_list_out
            
        else:
            res =  self.mul_singl(B)

        return res
    
    def mul_singl(self, B):
        if isinstance(B, numbers.Number):
            if B is 0:
                return 0
            else:
                return Correlator(self.coef * B, self.op_names_list, self.sys_inds_to_act, 
                             self.name2data_interpretator, self.multiplicator, self.dag_map)
            
        A = self
        multiplicator = self.multiplicator
        if multiplicator is None:
            raise ValueError('No multiplicator')
        
        A_names_list = A.names_list
        B_names_list = B.names_list
        N_A = len(A_names_list)
        N_B = len(B_names_list)
        
        if N_A < N_B:
            A_names_list = A_names_list + [I_name] * (N_B - N_A)
        else:
            B_names_list = B_names_list + [I_name] * (N_A - N_B)
        
        C_names_list = []
        C_coef = A.coef * B.coef
        for i in range(len(A_names_list)):
            name, coef = multiplicator(A_names_list[i], B_names_list[i])
            C_names_list += [name]
            C_coef *= coef
            
        op_names_list = [name for name in C_names_list if name is not I_name]
        sys_inds_to_act = [i for i, name in enumerate(C_names_list) if name is not I_name]
        
        if len(op_names_list) == 0:
            return C_coef
        return Correlator(C_coef, op_names_list, sys_inds_to_act, self.name2data_interpretator, self.multiplicator, self.dag_map)
        
    def name2data(self, name, is_qobj = False):
        return self.name2data_interpretator(name, is_qobj)
        
    def set_multiplicator(self, multiplicator):
        self.multiplicator = multiplicator
        
    def set_aux_inds(self, aux_inds, N_sys):
        self.aux_inds = aux_inds
        
        if len(self.aux_inds_to_act)!=0:
            self.state_names_list = copy(self.names_list) + ['I']*N_sys
            self.inds_list_state = self.aux_inds_to_act
            return
            
        self.inds_list_state = sys_to_state_inds(self.sys_inds_to_act, aux_inds)
        self.state_names_list = copy(self.names_list)
        for ind in sorted(aux_inds):
            self.state_names_list.insert(ind,'I')
    
    def full(self, N_modes, with_aux = False):
        I = identity(2)
        op_list = [self.name2data(name, is_qobj = True) for name in self.names_list]
        return self.coef * tensor(op_list + [I]*(N_modes-len(op_list)))
    
    def get_H_list(self):
        return [self.name2data(name, is_qobj = True) for name in self.op_names_list]
#     def get_correlator_part(self, inds_to_contract):
        
#         oper_list = [self.name2data(self.op_names_list[self.inds_list_state.index(ind)])
#                      for ind in inds_to_contract if ind in self.inds_list_state]
        
# #         oper_list = [self.opers_list[self.inds_list_state.index(ind)].data 
# #                      for ind in inds_to_contract if ind in self.inds_list_state]
#         names_out = [self.state_names_list[ind] for ind in inds_to_contract if ind < len(self.state_names_list)]

#         is_enough = len(inds_to_contract) != 0 and max(inds_to_contract) >= len(self.state_names_list)-1
            
#         if len(oper_list) is 0:
#             oper = 1
#         else:
#             oper = tensor_mq(oper_list)
#         return oper, names_out, is_enough
        
             
    def get_correlator_part(self, inds_to_contract):
        
        oper_list = [self.name2data(self.op_names_list[self.inds_list_state.index(ind)])
                     for ind in inds_to_contract if ind in self.inds_list_state]
        
#         oper_list = [self.opers_list[self.inds_list_state.index(ind)].data 
#                      for ind in inds_to_contract if ind in self.inds_list_state]
        names_out = [self.state_names_list[ind] if ind < len(self.state_names_list) else I_name 
                     for ind in inds_to_contract]

        is_enough = len(inds_to_contract) != 0 and max(inds_to_contract) >= len(self.state_names_list)-1
            
#         if len(oper_list) is 0:
#             oper = 1
#         else:
#             oper = tensor_mq(oper_list)

        is_purity = False
        if Purity_name in names_out:
            is_purity = True
            ind = names_out.index(Purity_name)
            del names_out[ind]
            del oper_list[ind]
            del inds_to_contract[ind]
        
        return is_purity, oper_list, names_out, inds_to_contract, is_enough

    def reset(self):
        self.value = None
        self.contract_ind = 0
        
    def get_opts_tensor(self):
        return tensor([self.name2data_interpretator(name) for name in self.op_names_list])

    def dag(self):
        coef = np.conj(self.coef)
        op_names_list = []
        for name in self.op_names_list:
            name_dag, c = self.dag_map(name)
            op_names_list += [name_dag]
            coef *= c
        
        return Correlator( coef, op_names_list, self.sys_inds_to_act, 
                            name2data_interpretator = self.name2data_interpretator, multiplicator = self.multiplicator, dag_map = dag_map)
        
        
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        
        return 'Correlator' + '\n' +\
    'op_names_list = ' + str(self.op_names_list) + '\n'+\
    'sys_inds_to_act = ' + str(self.sys_inds_to_act) + '\n'+\
    'coef = ' + str(self.coef)
Px_name = 'Sx'
Py_name = 'Sy'
Pz_name = 'Sz'
Pr_name = 'Sp'
Pl_name = 'Sm'
P0_name = 'proj_0'
P1_name = 'proj_1'
I_name = 'I'
Purity_name = 'purity'
PAULI_NAMES = [Px_name, Py_name, Pz_name, Pr_name, Pl_name, P0_name, P1_name, I_name, Purity_name]
PAULI_DATA_Q = [S_x, S_y, S_z, S_p, S_m, S_m*S_p, S_p*S_m, identity(2), identity(2) ]
PAULI_DATA_Q_normal = [d/d.data.data[0] for d in PAULI_DATA_Q]

PAULI_DATA = [MyQobj(d.full()) for d in PAULI_DATA_Q]
# PAULI_DATA_DENSE = [d.full() for d in PAULI_DATA_Q]

def name2data_pauli(name, is_qobj = False):
    if name not in PAULI_NAMES:
        raise ValueError('Not a Pauli')
    data_space = PAULI_DATA_Q if is_qobj else PAULI_DATA

    return data_space[PAULI_NAMES.index(name)]

def data2name_pauli(data):
    
    if len(data.data.data) is 0:
        return None, 0
    
    data_norm = data/data.data.data[0]
    for kk, p_n in enumerate(PAULI_DATA_Q_normal):
        if (p_n - data_norm).norm() == 0:
            p = PAULI_DATA_Q[kk]
            coef = data.data.data[0] / p.data.data[0]
            return PAULI_NAMES[kk], coef
    
    raise ValueError('Not a Pauli')

def multiplicator_pauli_prepare(A_name, B_name):

    A_data = name2data_pauli(A_name, is_qobj = True)
    B_data = name2data_pauli(B_name, is_qobj = True)
    
    return data2name_pauli(A_data * B_data)

MULT_MAP = {}
for n1 in PAULI_NAMES:
    MULT_MAP[n1] = {}
    for n2 in PAULI_NAMES:
        MULT_MAP[n1][n2] = multiplicator_pauli_prepare(n1, n2)
        

def multiplicator_pauli(A_name, B_name):

    return MULT_MAP[A_name][B_name]


def dag_map_pauli_prepare(name):
    data = name2data_pauli(name, is_qobj = True)
    return data2name_pauli(data.dag())

DAG_MAP = {}
for n1 in PAULI_NAMES:
    DAG_MAP[n1] = dag_map_pauli_prepare(n1)

def dag_map(name):
    return DAG_MAP[name]
def get_SSH_H_cor_3(N_mode, t_p, t_m, B=0):
    '''
    Return 'N_mode' SSH Hamiltonian
    '''
    cor_list = sum(Correlator(t_p+(-1)**n*t_m, [Px_name, Px_name], [n,n+1], name2data_pauli, multiplicator_pauli )
                   for n in range(N_mode-1))
    cor_list += sum(Correlator(t_p+(-1)**n*t_m, [Py_name, Py_name], [n,n+1], name2data_pauli, multiplicator_pauli )
                    for n in range(N_mode-1) )
    cor_list += Correlator(B, [Pz_name], [0], name2data_pauli, multiplicator_pauli)
    cor_list += Correlator(-B, [Pz_name], [N_mode-1], name2data_pauli, multiplicator_pauli )
    
    return cor_list
def get_schw_H_cor_3(N_mode, J = 1, w = 1, m = -0.5, e0 = 0):
    '''
    Return 'N_mode' Schwinger Hamiltonian
    '''
    n_f, m_f, d_f = name2data_pauli, multiplicator_pauli, dag_map
    
    Hs = sum(Correlator(w, [Pr_name, Pl_name], [n,n+1], n_f, m_f, d_f )
                   for n in range(N_mode-1))
    
    Hs += sum(Correlator(w, [Pl_name, Pr_name], [n,n+1], n_f, m_f, d_f )
                   for n in range(N_mode-1))

    Ln_array = [ e0 + -1./2*sum( [ ( Correlator(1, [Pz_name], [l], n_f, m_f, d_f )
                                    +(-1)**(l+1) ) for l in range(n+1) ]) 
                for n in range(N_mode-1) ]
    
    HJ = J * sum( [ L * L for L in Ln_array] )
    Hz_m = -m/2 * sum( [ Correlator((-1)**n, [Pz_name], [n], n_f, m_f, d_f ) for n in range(N_mode) ])
#     return Hz_m
    return Hs + HJ + Hz_m

def get_schw_H_cor_3_revers(N_mode, J = 1, w = 1, m = -0.5, e0 = 0):
    '''
    Return 'N_mode' Schwinger Hamiltonian
    '''
    
    Sz_list = []
    Sx_list = []
    Sy_list = []
    for ii in range(N_mode)[::-1]:
        Sz_list += [Correlator(-1, [Pz_name], [ii], name2data_pauli, multiplicator_pauli )]
        Sx_list += [Correlator(1, [Px_name], [ii], name2data_pauli, multiplicator_pauli )]
        Sy_list += [Correlator(1, [Py_name], [ii], name2data_pauli, multiplicator_pauli )]
    
    Hs = w * sum( [ 0.5*(Sx_list[n] * Sx_list[n+1] + Sy_list[n] * Sy_list[n+1]) for n in range(N_mode-1) ] )
   
    Ln_array = [ e0 + -1./2 * sum( [ ( Sz_list[l] + (-1)**(l+1) ) for l in range(n+1) ]) for n in range(N_mode-1) ]
    
    HJ = J * sum( [ L * L for L in Ln_array] )
    
    Hz_m = -m/2 * sum( [ (-1)**n * Sz_list[n] for n in range(N_mode) ])
    
    return Hs + HJ + Hz_m
def get_W_H_cor(N_mode):
    '''
    Return 'N_mode' parent Hamiltonian for W state
    '''
    cor_list = 0
    for P_name in [Px_name, Py_name]:
        cor_list += sum(Correlator(-1, [P_name, P_name], [n,n+1], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode-1))
        cor_list += Correlator(-1, [P_name, P_name], [N_mode-1, 0], name2data_pauli, multiplicator_pauli )
        
    cor_list += (sum(Correlator(1, [Pz_name], [n], name2data_pauli, multiplicator_pauli )
                    for n in range(N_mode) ) + (N_mode -2))**2
    
    return cor_list
def get_GHZ_antifer_H_cor(N_mode):
    '''
    Cost H for |0101...>+|1010...>
    '''
    cor_list = Correlator(-1, [Px_name]*N_mode, list(range(N_mode)), name2data_pauli, multiplicator_pauli )
    
    cor_list += sum(Correlator(1, [Pz_name, Pz_name], [n,n+1], name2data_pauli, multiplicator_pauli )
                   for n in range(N_mode-1))
    
    return cor_list
def get_MG_H_cor(N_mode):
    '''
    Return 'N_mode' Hamiltonian for Majumdar-Gosh model
    '''
    cor_list = 0
    for P_name in [Px_name, Py_name, Pz_name]:
        cor_list += 2*sum(Correlator(1, [P_name, P_name], [n,n+1], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode-1))
        
    for P_name in [Px_name, Py_name, Pz_name]:
        cor_list += sum(Correlator(1, [P_name, P_name], [n,n+2], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode-2))
    
    return cor_list
def get_clust_H_cor(N_mode, b = 0):
    '''
    Return 'N_mode' parent Hamiltonian wor a cluster state
    '''
    
    cor_list = sum(Correlator(-1, [Px_name, Pz_name, Px_name], [n,n+1,n+2], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode-2))
    
    cor_list += (Correlator(b, [Pz_name], [0], name2data_pauli, multiplicator_pauli )+
                  Correlator(b, [Pz_name], [N_mode-1], name2data_pauli, multiplicator_pauli ))
    
    return cor_list
def get_squeez_H_corr(N_mode):
    '''
    Return 'N_mode' parent Hamiltonian wor a squeesed state
    '''
    cor_list = -2*sum(Correlator(1, [Px_name], [n], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode))**2
    
    cor_list += -2*sum(Correlator(1, [Py_name], [n], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode))**2
    
    cor_list += sum(Correlator(-1, [Pz_name], [n], name2data_pauli, multiplicator_pauli )
                       for n in range(N_mode))**2
    
    return cor_list
def get_AKLT_H_corr(N_mode, l = 0, b = 0):
    '''
    Return 'N_mode' AKLT Hamiltonian
    '''
    N_mode_3 = N_mode//2
    
    S3_xyz_list = []
    for P_name in [Px_name, Py_name, Pz_name]:
        S3_list = [
            Correlator(1/2, [P_name], [2*n], name2data_pauli, multiplicator_pauli )+
            Correlator(1/2, [P_name], [2*n+1], name2data_pauli, multiplicator_pauli )
            for n in range(N_mode_3)
                    ]
        S3_xyz_list += [S3_list]
        
    S3_x_list, S3_y_list, S3_z_list = S3_xyz_list
    
    H = 0
    for n in range(N_mode_3-1):
        SS = sum(S_list[n]*S_list[n+1] for S_list in [S3_x_list,S3_y_list,S3_z_list] )
        H += SS
        H += 1/3*SS**2
    
    
    Sz_list = [Correlator(1, [Pz_name], [ii], name2data_pauli, multiplicator_pauli ) for ii in range(N_mode) ]
    H +=b*(Sz_list[0] + Sz_list[1] + -1*(Sz_list[-1] + Sz_list[-2]))
    
    H += sum( 
        Correlator(-l, [P_name, P_name], [2*ii, 2*ii+1], name2data_pauli, multiplicator_pauli )
        for P_name in [Px_name, Pz_name, Px_name] for ii in range(N_mode_3)
    )
    H += l * N_mode_3
    return H
def get_H_Hald_qub_cor(N_mode, J = 1, Delt = 1, D2 = 1, l = 0):
    
    N_mode_3 = N_mode//2
    
    S3_xyz_list = []
    for P_name in [Px_name, Py_name, Pz_name]:
        S3_list = [
            Correlator(1/2, [P_name], [2*n], name2data_pauli, multiplicator_pauli )+
            Correlator(1/2, [P_name], [2*n+1], name2data_pauli, multiplicator_pauli )
            for n in range(N_mode_3)
                    ]
        S3_xyz_list += [S3_list]
        
    S3_x_list, S3_y_list, S3_z_list = S3_xyz_list
    
    H = 0
    for n in range(N_mode_3-1):
        H += sum(S_list[n]*S_list[n+1] for S_list in [S3_x_list,S3_y_list] )
        H += Delt*S3_z_list[n]*S3_z_list[n+1]
    
    H += D2*sum(S3_z*S3_z for S3_z in S3_z_list)
    H *= J
    
    H += sum( 
        Correlator(-l, [P_name, P_name], [2*ii, 2*ii+1], name2data_pauli, multiplicator_pauli )
        for P_name in [Px_name, Pz_name, Px_name] for ii in range(N_mode_3)
    )
    H += l * N_mode_3
    return H
def get_extended_space_schw(N, H): 
    n_f, m_f, d_f = name2data_pauli, multiplicator_pauli, dag_map
    op_list = [1]
    for j in range(N//2):
        op = Correlator(1, [Px_name, Px_name], [j, j+1], n_f, m_f, d_f)+\
            Correlator(1, [Px_name, Px_name], [N-2-j, N-1-j], n_f, m_f, d_f)+\
            Correlator(1, [Py_name, Py_name], [j, j+1], n_f, m_f, d_f)+\
            Correlator(1, [Py_name, Py_name], [N-2-j, N-1-j], n_f, m_f, d_f)
        
        op_list += [op]
    
    op_dag_list = [np.conj(op) if isinstance(op, numbers.Number) else op.dag() for op in op_list]

    OHO_list = []
    OO_list = []
    for op_i in op_dag_list:
        OH = op_i*H
        for op_j in op_list:
            OHO = OH*op_j
            OHO_list += [OHO]
            
            OO_list += [op_i*op_j]
    
    return OHO_list, OO_list


def get_extended_space_schw_full(N, H): 
    Sx_list = [];
    Sy_list = [];
    for ii in range(N):
        Sx_list += [make_multimode_oper(S_x, N, [ii])]
        Sy_list += [make_multimode_oper(S_y, N, [ii])]
        
    op_list = [1]
    for j in range(N//2):
        op = Sx_list[j] * Sx_list[j+1]+\
            Sy_list[j] * Sy_list[j+1]+\
            Sx_list[N-2-j] * Sx_list[N-1-j]+\
            Sy_list[N-2-j] * Sy_list[N-1-j]
        op_list += [op]
    
    op_dag_list = [np.conj(op) if isinstance(op, numbers.Number) else op.dag() for op in op_list]

    OHO_list = []
    OO_list = []
    for op_i in op_dag_list:
        OH = op_i*H
        for op_j in op_list:
            OHO = OH*op_j
            OHO_list += [OHO]
            
            OO_list += [op_i*op_j]
    
    return OHO_list, OO_list
def block_order(N_sys, mps_circ , bulk_offset = 0, with_aux = True):
    d_N, size_block, size_edge_l, size_edge_r, N_bulk_blocks = \
            mps_circ.d_N, mps_circ.size_block, mps_circ.size_edge_L, mps_circ.size_edge_R, mps_circ.N_bulk_blocks
    
    inds_block_bulk = [
        [0]*with_aux + list(range(with_aux + i*d_N ,with_aux + size_block + i*d_N))
        for i in range(bulk_offset, (N_sys - size_block-bulk_offset)//d_N + 1)
    ]
    
    blocks_order = []
    ind = 0
    if size_edge_l!= 0:
        blocks_order = [[0,[0]*with_aux+list(range(with_aux,size_edge_l+with_aux)), False]]
        ind += 1
        
    blocks_order += [[k%N_bulk_blocks+ind, inds, False] for k, inds in enumerate(inds_block_bulk)]
    
    ind += N_bulk_blocks
    
    if size_edge_r!= 0:
        blocks_order += [[ind,[0]*with_aux+list(range(N_sys-size_edge_r+with_aux,N_sys+with_aux)), False]]
    
    return blocks_order
class MPS_Circuit(object):
    '''
    Properties of the boxed defined MPS circuit.
    
    Parameters
    ----------
    size_edge_L: int
        Number of qubits in the first (left) box
    N_lays_edge_L: int
        Number of layers in the first box
    size_edge_R: int
        Number of qubits in the lasr (right) box
    N_lays_edge_R: int
        Number of layers in the last box
    size_block: int
        Number of qubits in the bulk boxes
    N_lays_block: int, minimum 1
        Number of layers in the bulk box. Not used if N_bulk_gates defined.
    N_bulk_blocks: int, minimum 1
        Number of different alternating bulk boxes.
    d_N: int, min 1
        Number of qubits to which the bulk boxes are shiftet after each other.
    N_bulk_gates: int
        Number of gates invested to all bulk boxes.
    '''
    
    def __init__( self, 
                 size_block, N_lays_block = 1, 
                 size_edge_L = None, N_lays_edge_L = 1, 
                 size_edge_R = None, N_lays_edge_R = 1, 
                 size_M = None, N_lays_M = 1, 
                 N_bulk_blocks = 1, d_N = 1, N_bulk_gates = None):
        
        self.size_edge_L = 0 if N_lays_edge_L is 0 else \
            size_block-1 if size_edge_L is None else size_edge_L
        
        self.size_edge_R = 0 if N_lays_edge_R is 0 else \
            size_block-1 if size_edge_R is None else size_edge_R
            
        self.size_block = size_block
        self.N_lays_block = N_lays_block
        self.N_lays_edge_L = N_lays_edge_L
        self.N_lays_edge_R = N_lays_edge_R
        self.N_bulk_blocks = N_bulk_blocks
        self.d_N = d_N
        self.N_bulk_gates = N_bulk_blocks * size_block * N_lays_block if N_bulk_gates is None else N_bulk_gates
    
        self.size_M = size_M
        self.N_lays_M = N_lays_M
        
    def set_N_lays_block(self, N_lays_block):
        self.N_lays_block = N_lays_block
        self.N_bulk_gates = self.N_bulk_blocks * self.size_block * N_lays_block
        
        
    def get_N_params_per_block(self):
        
        N_params = self.N_bulk_gates//self.N_bulk_blocks
        N_params_list = [N_params]*self.N_bulk_blocks
        
        for i in range(self.N_bulk_gates%self.N_bulk_blocks):
            N_params_list[i]+=1
        
        return N_params_list
    
    def conseq_block_constructor(self, layer):
    
        layer_edge_L = layer[:self.size_edge_L] * self.N_lays_edge_L
        layer_edge_R = layer[:self.size_edge_R] * self.N_lays_edge_R
        
        N_params_blocks_list = self.get_N_params_per_block()
        
        layer_block_list = []
        for N_params in N_params_blocks_list:
            
            layer_bulk = layer[:self.size_block]*(N_params//self.size_block)
            if N_params%self.size_block == self.d_N+1:
                layer_bulk += layer[:self.d_N]+[ layer[self.d_N+1] ]
            else:
                layer_bulk += layer[:N_params%self.size_block ]
                
            layer_block_list += [layer_bulk]
            
        return layer_edge_L, layer_edge_R, layer_block_list
    
    def last_ind_param_conseq(self):
        N_params_blocks_list = self.get_N_params_per_block()
        
        N_div = self.N_bulk_gates%self.N_bulk_blocks
        last_block_ind = self.N_bulk_blocks-1 if N_div == 0 else N_div-1
        ind_last = self.size_edge_L * self.N_lays_edge_L + sum(N_params_blocks_list[:last_block_ind+1])-1
        
        # according to conseq_block_constructor in order the gates in 
        # the block anre not repeated in a row
        x = N_params_blocks_list[last_block_ind]%self.size_block
        if x == self.d_N+2 or (x ==0 and self.d_N+2 == self.size_block):
            ind_last -= 1
            
        return ind_last
def get_MPS_simulator(N_sys, mps_circ, model_args, MODEL_TAG, opt_props, n_phon = 0, block_constructor = None, dim_aux = None, cnot_mode = False, n_phon_pur = None):
    
    if dim_aux is None:
        dim_aux_MAX = 15
        dim_aux_MIN = N_sys//2+1

        if n_phon > 0:
            dim_aux_MIN += 5

        dim_aux =min(dim_aux_MIN, dim_aux_MAX)
    
    if n_phon_pur is None:
        state_aux = [thermal_dm(dim_aux,n_phon)]
    else:
        state_aux = [fock_dm(dim_aux, n_phon_pur)]
        
    states_in_list = state_aux+[fock(2,i%2) for i in range(N_sys)]
#     x=0.005
#     states_in_list = state_aux+[(1-x)*fock_dm(2,i%2)+ x*fock_dm(2,(i+1)%2) for i in range(N_sys)]

    system = System( states_in_list, logic_mode_inds=[0], inverse_logic=True)
    model_mps = ModelMPS(MODEL_TAG, N_sys, args=model_args)

    g_blue, g_red = red_and_blue_gates(dim_aux)
    
    max_box_size = max(mps_circ.size_edge_L, 
                       mps_circ.size_edge_R, 
                       mps_circ.size_block)
    
    layer = [Gate(g_blue, inds_to_act_list = [0,i]) for i in range(1,max_box_size+1)]
    
    N_bulk_params_list = mps_circ.get_N_params_per_block()
    
    if mps_circ.N_bulk_blocks == 2:
        layer_edge_L, layer_edge_R, layer_block_list = construct_2_blocks(layer, mps_circ)
    else:
        layer_edge_L = layer[:mps_circ.size_edge_L] * mps_circ.N_lays_edge_L
        layer_edge_R = layer[:mps_circ.size_edge_R] * mps_circ.N_lays_edge_R
        layer_block_list = [layer[:mps_circ.size_block] * (n//mps_circ.size_block) + layer[:(n%mps_circ.size_block)]
                           for n in N_bulk_params_list]
    
        
#     layer_edge_L = [layer[i] for i in [0,1]]
#     layer_edge_R = [layer[i] for i in [0,1]]
#     layer_block_list = [[layer[i] for i in [2,1,0]], [layer[i] for i in [1,0,1,2]]]
#     layer_block_list = [[layer[i] for i in [2,1,0,1,2]], [layer[i] for i in [2,1,0,1,2]]]
#     layer_block_list = [[layer[i] for i in [2,1,0,1]], [layer[i] for i in [1,0,1,2]]]

    
    gates_seq_bulk_list = [GatesSequence( layer_block ) for layer_block in layer_block_list]
    gates_seq_L = [GatesSequence( layer_edge_L )] if mps_circ.size_edge_L is not 0 else []
    gates_seq_R = [GatesSequence( layer_edge_R )] if mps_circ.size_edge_R is not 0 else []
    
    gates_seq_list = gates_seq_L + gates_seq_bulk_list + gates_seq_R
    
    params_init = 0.01
    blocks_order = block_order(N_sys, mps_circ)
    simulator_mps = SimulatorMPS(system, opt_props, model_mps, params_init, gates_seq_list, blocks_order)
    
    return simulator_mps
def f_ind(size_block, l):
    
    return abs(size_block - 1 - l%(2*(size_block-1)))

def construct_2_blocks(layer, mps_circ):
    
    layer_edge_L = layer[:mps_circ.size_edge_L] * mps_circ.N_lays_edge_L
    layer_edge_R = layer[:mps_circ.size_edge_R] * mps_circ.N_lays_edge_R
    
    s = mps_circ.size_block
    
    n_bulk = mps_circ.N_bulk_gates
    n_1 = n_bulk//2
    n_2 = n_bulk//2
    
    if n_bulk%2 == 1: 
        x = ( n_bulk-2)/(2*(s-1))%2
#         print(x)
        if x <= 1 and x > 0: 
            n_1 += 1
        else:
            n_2 += 1
#     print(n_1, n_2)
    layer_inds_1 = [f_ind(s, l) for l in range(n_1)]
    layer_inds_2 = [f_ind(s, l) for l in range(n_2)][::-1]
#     print(layer_inds_1, layer_inds_2)
#     print(layer_inds_1)
#     print(np.array(layer_inds_2)+1)
#     print(layer_inds_1)
#     print(layer_inds_2)
    layer_block_1 = [layer[i] for i in layer_inds_1]
    layer_block_2 = [layer[i] for i in layer_inds_2]
    layer_block_list = [layer_block_1, layer_block_2]
    
    return layer_edge_L, layer_edge_R, layer_block_list
def get_MPS_with_MS_simulator(N_sys, mps_circ, opt_props, MODEL_TAG, model_args):
    
    states_in_list = [fock(2,i%2) for i in range(N_sys)]

    system = System( states_in_list )
    model_mps = ModelMPS(MODEL_TAG, N_sys, args=model_args)

    MS_edge = Gate(get_H_MS( S_x, mps_circ.size_edge_L))
    MS_bulk = Gate(get_H_MS( S_x, mps_circ.size_block))
    
    layer_z = [Gate(S_z, inds_to_act_list = [i]) for i in range(max(mps_circ.size_edge_L, mps_circ.size_block))]

    layer_edge_L, layer_edge_R = [], []
    for i in range(mps_circ.N_lays_edge_L):
        layer_edge_L += deepcopy([MS_edge] + layer_z[:mps_circ.size_edge_L])

    for i in range(mps_circ.N_lays_edge_R):
        layer_edge_R += deepcopy([MS_edge] + layer_z[:mps_circ.size_edge_R])

    N_bulk_params_list = mps_circ.get_N_params_per_block()

    layer_block_list = []
    for n in N_bulk_params_list:
        layer_block = []
        for i in range(n):
            layer_block += deepcopy([MS_bulk] + layer_z[:mps_circ.size_block])
        layer_block_list += [layer_block]


    gates_seq_bulk_list = [GatesSequence( layer_block ) for layer_block in layer_block_list]
    gates_seq_L = [GatesSequence( layer_edge_L )] if mps_circ.size_edge_L is not 0 else []
    gates_seq_R = [GatesSequence( layer_edge_R )] if mps_circ.size_edge_R is not 0 else []

    gates_seq_list = gates_seq_L + gates_seq_bulk_list + gates_seq_R

    params_init = 0.01
    blocks_order = block_order(N_sys, mps_circ, with_aux = False)
    simulator_mps = SimulatorMPS(system, opt_props, model_mps, params_init, gates_seq_list, blocks_order)
    
    return simulator_mps
def get_multiqub_simulator(N_modes, N_layers, opt_props, model_tag, model_args, edge_size =  None, alpha = 3):
    
    with_trans = True if edge_size is not None else False
    
    st_sys_in_q = tensor([fock(2,0),fock(2,1)]*(N_modes//2)).unit()
    system = System( st_sys_in_q )
    model = Model(model_tag, N_modes, args=model_args)
    
    H_z = [make_multimode_oper(S_z, N_modes, [i]) for i in range(N_modes)]
    H_JZ = get_my_Hij(N_modes, alpha = alpha, B = 20)

    if model_tag == TAG_SCHWING or model_tag == TAG_SSH_P:
        if with_trans:
            H_z_list = [H_z[i] - H_z[N_modes - i - 1] for i in range(edge_size)]
            if N_modes!=2*edge_size:
                H_z_list += [sum(H_z[k]*(-1)**k for k in range(edge_size, N_modes-edge_size))]
        else:
            H_z_list = [H_z[i] - H_z[N_modes - i - 1] for i in range(N_modes//2)]
    elif model_tag == TAG_SSH_P:
        if with_trans:
            H_z_list = [H_z[i] + H_z[N_modes - i - 1] for i in range(edge_size)]
            if N_modes!=2*edge_size:
                H_z_list += [sum(H_z[k] for k in range(edge_size, N_modes-edge_size))]
        else:
            H_z_list = [H_z[i] + H_z[N_modes - i - 1] for i in range(N_modes//2)]
    else:
        raise NotImplementedError(model_tag + ' is not implemented')
    
#     H_z_list = H_z
    
    gate_Z = [Gate(h) for h in H_z_list]
    gate_JZ = [Gate(H_JZ)]
    layer = gate_JZ + gate_Z
    
    gates_list = layer * N_layers
    
    gates_seq = GatesSequence( gates_list )
    
    simulator = SimulatorCoherent(system, opt_props, model, gates_seq, params_init = .01)
    
    return simulator
def get_correlators(N_modes, P_name):
    cor_list = []
    for i in range(N_modes):
        cor_list += [Correlator(1, [P_name], [i], name2data_pauli, multiplicator_pauli )]
            
    return cor_list

def get_correlators_2(N_modes, P_name):
    cor_list = []
    for i in range(N_modes):
        for j in range(i+1,N_modes):
            cor_list += [Correlator(1, [P_name, P_name], [i,j], name2data_pauli, multiplicator_pauli )]
            
    return cor_list

def get_correlators_xx_yy(N_modes):
    cor_list = []
    for i in range(N_modes):
        for j in range(i+1,N_modes):
            cor_list += [Correlator(1, [Px_name, Px_name], [i,j], name2data_pauli, multiplicator_pauli ) +\
            Correlator(1, [Py_name, Py_name], [i,j], name2data_pauli, multiplicator_pauli )]
            
#         for j in range(i+1,N_modes):
#             cor_list += [Correlator(1, [Pz_name, Pz_name], [i,j], name2data_pauli, multiplicator_pauli )]
            
    return cor_list

def get_correlators_xx_x(N_modes):
    cor_list_xx = []
    for i in range(N_modes):
        for j in range(i+1,N_modes):
            cor_list_xx += [Correlator(1, [Px_name, Px_name], [i,j], name2data_pauli, multiplicator_pauli )]

    cor_list_x = []
    for i in range(N_modes):
        cor_list_x += [Correlator(1, [Px_name], [i], name2data_pauli, multiplicator_pauli )]
            
    return cor_list_xx, cor_list_x

def get_SSH_correlators_old(N_modes, ards, cor_name):
    return get_SSH_correlators(N_modes, ards[0], ards[1], ards[2])


def get_correlators_DMRG(H_corr, cor_name_2, cor_name_1 = None):
    
    N_modes = len(H_corr.get_H_dims())
    
    args_list = []
    for c in H_corr.correlators_list:
        if isinstance(c, numbers.Number):
            continue
        n = len(c.op_names_list)
        args = [c.coef]
        for i in range(n):
            args += [c.op_names_list[i], c.sys_inds_to_act[i]+1]
        args_list += [tuple(args)]
    
    do_1 = 0 if cor_name_1 is None else 1
    do_2 = 0 if cor_name_2 is None else 1
    E_list_dmrg = np.array(get_correlators_c(cor_name_1, cor_name_2, N_modes, args_list, do_1, do_2))
    
    
    E_list_1 = E_list_dmrg[:N_modes] if do_1 else []
    E_list_2 = E_list_dmrg[do_1*N_modes:] if do_2 else []
        
    return E_list_1, E_list_2

def reshape_E_cors(E_cors):
    N_c = len(E_cors)
    N = int(((1+8*N_c)**0.5+1)/2)
    
    M = np.empty((N,N))
    M[:] = numpy.nan
    
    k = 0
    for i in range(N):
        for j in range(i+1,N):
            M[i, j] = np.real(E_cors[k])
            k+=1    
    return M


def reshape_E_cors_err(E_cors):
    N_c = len(E_cors)
    N = int(((1+8*N_c)**0.5+1)/2)
    
    M = np.empty((N,N))
    M[:] = 0
    
    k = 0
    for i in range(N):
        for j in range(i+1,N):
            if i != 0 and j!= N-1:
                M[i, j] = np.real(E_cors[k])
            k+=1    
    return M

# def err_func(E_cors, E_cors_dmrg):
    
#     M1 = reshape_E_cors_err(E_cors)
#     M2 = reshape_E_cors_err(E_cors_dmrg)
    
#     err = np.sum(np.abs(M1-M2))/np.sum(np.abs(M2))

#     return err

def err_func(E_cors, E_cors_dmrg):
    err = np.sum(np.abs(E_cors-E_cors_dmrg))/np.sum(np.abs(E_cors_dmrg))
#     err_list = [(E_cors[i]-E_cors_dmrg[i])/E_cors_dmrg[i] for i in range(len(E_cors_dmrg))]
#     err = np.sum(np.abs(err_list))/N_sys**2
#     print(err_list, err)
#     sds
    return err
def get_e1_and_e0_DMRG(H_corr):
    
    N_modes = len(H_corr.get_H_dims())

    args_list = []
    for c in H_corr.correlators_list:
        if isinstance(c, numbers.Number):
            continue
        n = len(c.op_names_list)
        args = [c.coef]
        for i in range(n):
            args += [c.op_names_list[i], c.sys_inds_to_act[i]+1]
        args_list += [tuple(args)]

    ens = get_e1_and_e0_c(N_modes, args_list)
    
    return ens
def double_correlators_MPS(simulator_mps, params_opt, P_name, with_aux_state = False):
    N_sys = simulator_mps.system.N_sys
    cor_list_z, cor_list_zz = get_correlators(N_sys, P_name), get_correlators_2(N_sys, P_name)
    
    if with_aux_state:
        pur_cor = Correlator(1, [], [], name2data_pauli, aux_inds_to_act = [0], aux_op_names_list = [Purity_name])
    else:
        pur_cor = []
        
    E_cors_out, _ = simulator_mps.obtain_average(cor_list_z + cor_list_zz + [pur_cor], params_opt)
    E_cors_z, E_cors_zz = E_cors_out[:N_sys], E_cors_out[N_sys:N_sys + len(cor_list_zz)]
    E_list = process_correlators(E_cors_z, E_cors_zz)
    
    data_aux = E_cors_out[-1] if with_aux_state else None
        
    return E_list, data_aux

def double_correlators_DMRG(H_corr, P_name):
    E_list_z_dmrg, E_list_zz_dmrg = get_correlators_DMRG(H_corr, cor_name_1 = P_name, cor_name_2 =P_name)
    E_list_dmrg = process_correlators(E_list_z_dmrg, E_list_zz_dmrg)
    
    return E_list_dmrg

def double_correlators_MQ(simulator, params_opt, P_name):
    
    N_aux = simulator.system.N_aux
    N_sys = simulator.system.N_sys
    
    sim_res = simulator.simulation(params_opt, return_F = True)
    rho = sim_res.rho_out.to_qobj().ptrace(range(N_aux, N_aux+N_sys))
    
    cor_list_z, cor_list_zz = get_correlators(N_sys, P_name), get_correlators_2(N_sys, P_name)
    cor_av_list_z = average_cors_with_full(cor_list_z, rho)
    cor_av_list_zz = average_cors_with_full(cor_list_zz, rho)
    
    E_list = process_correlators(cor_av_list_z, cor_av_list_zz)
    
    return E_list

def average_cors_with_full(cor_list, rho):
    N_sys = get_q_N_mode(rho)
    cor_av_list = []
    for c in cor_list:
        c_cor = c.full(N_sys)
        cor = np.real((c_cor * rho).tr())
        cor_av_list += [cor]
    return cor_av_list

def process_correlators(E_list_1, E_list_2):
    E_list = np.zeros(len(E_list_2))
    N_sys = len(E_list_1)
    k = 0
    for i in range(N_sys):
        for j in range(i+1,N_sys):
            E_list[k] = E_list_2[k] - E_list_1[i]*E_list_1[j]
            k+=1
            
    return E_list
def get_MPS_simulator_with_traps(N_sys, mps_circ, model_args, MODEL_TAG, opt_props, n_phon = 0, N_traps = 1, is_free=False, dim_aux = None):
    
    if dim_aux is None:
        dim_aux_MAX = 15
        dim_aux_MIN = N_sys//2+1

        if n_phon > 0:
            dim_aux_MIN += 5

        dim_aux =min(dim_aux_MIN, dim_aux_MAX)
    
    state_aux = [thermal_dm(dim_aux,n_phon)]*N_traps
    states_in_list = state_aux+[fock(2,i%2) for i in range(N_sys)]

    system = System( states_in_list, logic_mode_inds=list(range(N_traps)), inverse_logic=True)
    model_mps = ModelMPS(MODEL_TAG, N_sys, args=model_args)

    g_blue, g_red = red_and_blue_gates(dim_aux)
    
    
#     dim_aux_MAX = 20
#     dim_aux =min(N_sys//2+1, dim_aux_MAX)
    
#     state_aux = [thermal_dm(dim_aux,n_phon)]*N_traps
#     states_in_list = state_aux+[fock(2,i%2) for i in range(N_sys)]

#     system = System( states_in_list, logic_mode_inds=list(range(N_traps)), inverse_logic=True)
#     model_mps = ModelMPS(MODEL_TAG, N_sys, args=model_args)

#     g_blue, g_red = red_and_blue_gates(dim_aux)
    
    max_box_size = max(mps_circ.size_edge_L, 
                       mps_circ.size_edge_R, 
                       mps_circ.size_block)
    
    layer = [Gate(g_blue, inds_to_act_list = [0,i]) for i in range(1,max_box_size+1)]
    
    N_bulk_params_list = mps_circ.get_N_params_per_block()

#     layer_edge_L = layer[:mps_circ.size_edge_L] * mps_circ.N_lays_edge_L
#     layer_edge_R = layer[:mps_circ.size_edge_R] * mps_circ.N_lays_edge_R
#     layer_block_list = [layer[:mps_circ.size_block] * (n//mps_circ.size_block) + layer[:(n%mps_circ.size_block)]
#                        for n in N_bulk_params_list]
    
    if mps_circ.N_bulk_blocks == 2:
        layer_edge_L, layer_edge_R, layer_block_list = construct_2_blocks(layer, mps_circ)
    else:
        layer_edge_L = layer[:mps_circ.size_edge_L] * mps_circ.N_lays_edge_L
        layer_edge_R = layer[:mps_circ.size_edge_R] * mps_circ.N_lays_edge_R
        layer_block_list = [layer[:mps_circ.size_block] * (n//mps_circ.size_block) + layer[:(n%mps_circ.size_block)]
                           for n in N_bulk_params_list]
        
    
    gates_seq_bulk_list = [GatesSequence( layer_block ) for layer_block in layer_block_list]
    gates_seq_L = [GatesSequence( layer_edge_L )] if mps_circ.size_edge_L is not 0 else []
    gates_seq_R = [GatesSequence( layer_edge_R )] if mps_circ.size_edge_R is not 0 else []
    
    
    
    
    blocks_order = block_order_with_traps(N_sys, mps_circ, N_traps)
    gates_seq_list = gates_seq_L + gates_seq_bulk_list + gates_seq_R
    if is_free and N_traps>1:
        blocks_order = block_order_with_traps_free(N_sys, mps_circ, N_traps)
        layer_M = layer[:mps_circ.size_M] * mps_circ.N_lays_M
        gates_seq_list += [GatesSequence( layer_M )]
    
    params_init = 0.01
#     blocks_order = block_order(N_sys, mps_circ)
    simulator_mps = SimulatorMPS(system, opt_props, model_mps, params_init, gates_seq_list, blocks_order)
    
    return simulator_mps

def block_order_with_traps(N_sys, mps_circ, N_traps):
    d_N, size_block, size_edge_l, size_edge_r, N_bulk_blocks = \
            mps_circ.d_N, mps_circ.size_block, mps_circ.size_edge_R, mps_circ.size_edge_R, mps_circ.N_bulk_blocks
    
    N_sys_per_block = N_sys//N_traps
    N_sys = N_sys_per_block * N_traps
    
    inds_block_bulk = []
    inds_block_bulk += [
        [0] + list(range(N_traps + i*d_N ,N_traps + size_block + i*d_N))
        for i in range((N_sys - size_block)//d_N + 1)
    ]
        
    blocks_order = [[0,[0]+list(range(N_traps,N_traps + size_edge_l))]]
    
    ind = 1
    blocks_order += [[k%N_bulk_blocks+ind, inds] for k, inds in enumerate(inds_block_bulk)]
    
    ind = 1 + N_bulk_blocks
    blocks_order += [[ind,[0]+list(range(N_sys-size_edge_r+N_traps,N_sys+N_traps))]]
    
    for block in blocks_order:
        block += [False]
    
    last_block = deepcopy(blocks_order[-1])
    last_block_reverse = deepcopy(last_block)
    last_block_reverse[-1] = True
    blocks_to_insert = [last_block, last_block_reverse]
    
    inds_to_insert = list(range(1+N_sys_per_block - size_edge_l,len(blocks_order),N_sys_per_block)[:-1])
    
    n = len(blocks_order)
    for ind in inds_to_insert[::-1]:
        blocks = deepcopy(blocks_to_insert)
        for b in blocks:
            b[1][1:] = blocks_order[ind-1][1][2:]
            
        blocks_order = blocks_order[:ind] + blocks + blocks_order[ind:]
        for block in blocks_order[ind+1:]:
            block[1][0]+=1

    return blocks_order


def block_order_with_traps_free(N_sys, mps_circ, N_traps):
    d_N, size_block, size_edge_l, size_edge_r, N_bulk_blocks = \
            mps_circ.d_N, mps_circ.size_block, mps_circ.size_edge_R, mps_circ.size_edge_R, mps_circ.N_bulk_blocks
    
    N_sys_per_block = N_sys//N_traps
    N_sys = N_sys_per_block * N_traps
    
    inds_block_bulk = []
    inds_block_bulk += [
        [0] + list(range(N_traps + i*d_N ,N_traps + size_block + i*d_N))
        for i in range((N_sys - size_block)//d_N + 1)
    ]
        
    blocks_order = [[0,[0]+list(range(N_traps,N_traps + size_edge_l))]]
    
    ind = 1
    blocks_order += [[k%N_bulk_blocks+ind, inds] for k, inds in enumerate(inds_block_bulk)]
    
    ind = 1 + N_bulk_blocks
    blocks_order += [[ind,[0]+list(range(N_sys-size_edge_r+N_traps,N_sys+N_traps))]]
    
    for block in blocks_order:
        block += [False]
    
    last_block = deepcopy(blocks_order[-1])
    last_block[0] = 4
    last_block_reverse = deepcopy(last_block)
    last_block_reverse[-1] = True
    blocks_to_insert = [last_block, last_block_reverse]
    
    inds_to_insert = list(range(1+N_sys_per_block - size_edge_l,len(blocks_order),N_sys_per_block)[:-1])
    
    n = len(blocks_order)
    for ind in inds_to_insert[::-1]:
        blocks = deepcopy(blocks_to_insert)
        for b in blocks:
            b[1][1:] = blocks_order[ind-1][1][2:]
            
        blocks_order = blocks_order[:ind] + blocks + blocks_order[ind:]
        for block in blocks_order[ind+1:]:
            block[1][0]+=1

    return blocks_order

