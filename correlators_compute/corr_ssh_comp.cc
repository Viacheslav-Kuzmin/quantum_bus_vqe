#include <Python.h>
using namespace itensor;
#include "/scratch/x2241057/MPS_correlators/correlators_compute/itensor/all.h"
#include "xtensor/xarray.hpp"


using namespace xt;
using namespace std;


using array_type = xt::xarray<double, xt::layout_type::row_major>;
using shape_type = array_type::shape_type;


struct RetValH {
  SpinHalf sites;
  MPO H;
};
struct RetValDMRG {
  SpinHalf sites;
  MPS psi;
};

static RetValH get_H(int N_mode, PyObject *args)
{
    
    Complex coef;
    char* op1;
    char* op2;
    int ind1, ind2, i, seqlen;
    
    int itemlen;
    PyObject* seq;
    
    seq = PySequence_Fast(args, "argument must be iterable");
    seqlen = PySequence_Fast_GET_SIZE(seq);

    auto sites = SpinHalf(N_mode);
    auto ampo = AutoMPO(sites);
    for(i=0; i < seqlen; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        item = PySequence_Fast(item, "argument must be iterable");
        itemlen = PySequence_Fast_GET_SIZE(item);
        
        if(itemlen==3){
            PyArg_ParseTuple(item, "Dsi", &coef, &op1, &ind1);
            ampo += 2*coef,op1,ind1;
        }
        if(itemlen==5){
            PyArg_ParseTuple(item, "Dsisi", &coef, &op1, &ind1, &op2, &ind2);
            ampo += 4*coef,op1,ind1,op2,ind2;
        }
    }
    return {sites, MPO(ampo)};
}

RetValDMRG dmrg_H(int const& N_mode, PyObject *args)
{   
    auto ret_val_H = get_H(N_mode, args);
//     auto ret_val_H = get_H_SSH(N_mode, args);
    auto sites = ret_val_H.sites;

    auto H = ret_val_H.H;

    //Set up random initial wavefunction
    auto psi = MPS(sites);

    //Perform 5 sweeps of DMRG
    auto sweeps = Sweeps(60);
    //Specify max number of states kept each sweep
//     sweeps.maxm() = 1, 1;
    sweeps.maxm() = 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000;
//     sweeps.maxm() = 50, 50, 100, 100, 200, 200, 500, 500, 1000, 1000, 2000, 2000;

    dmrg(psi,H,sweeps,{"Silent",true});

    //Continue to analyze wavefunction afterward 
//     Real energy = overlap(psi,H,psi);

    return {sites, psi};
}



float *get_correlators_2(MPS psi, SiteSet sites, std::string op_name)
{   
    ITensor C, C_j;

    int N_mode = sites.N();
    
    size_t N_C = (N_mode*(N_mode-1))/2;
    float *correlators = new float[N_C];
//     static float correlators[N_C] = {};
    
    int k = 0;
    for(int i = 1; i < N_mode; ++i){

        psi.position(i);
        auto S_i = sites.op(op_name,i);


        // Calculate first correlator at site i
        C = psi.A(i);
        C *= S_i;
        auto ir = commonIndex(psi.A(i),psi.A(i+1),Link);
        C *= dag(prime(prime(psi.A(i),Site),ir));

        for(int j = i+1; j <= N_mode; ++j){

            // Common operation
            C *= psi.A(j);

            // Calculate second correlator at site j
            auto S_j = sites.op(op_name,j);
            C_j = C*S_j;
            auto il = commonIndex(psi.A(j),psi.A(j-1),Link);
            C_j *=  dag(prime(prime(psi.A(j),Site),il));

            correlators[k] = 4*C_j.real();
//             println(correlators[k]);
            k+=1;
            
            // Continue contraction
            C *= dag(prime(psi.A(j),Link));    
            }
        }
    return correlators;
}

float *get_correlators_1(MPS psi, SiteSet sites, std::string op_name)
{   
    ITensor C;

    int N_mode = sites.N();
    
    float *correlators = new float[N_mode];
    
    int k = 0;
    for(int i = 1; i <= N_mode; ++i){

        psi.position(i);
        auto S_i = sites.op(op_name,i);


        // Calculate first correlator at site i
        C = psi.A(i);
        C *= S_i;
        C *= dag(prime(psi.A(i),Site));
        correlators[k] = 2*C.real();
//         correlators[k] = 1;
        k+=1;
    }
    return correlators;
}

// float *get_correlators_1(MPS psi, SiteSet sites, std::string op_name)
// {   
//     ITensor C, C_j;

//     int N_mode = sites.N();
    
//     float *correlators = new float[N_mode];
// //     static float correlators[N_C] = {};
    
//     int k = 0;
//     for(int i = 1; i < N_mode; ++i){

//         psi.position(i);
//         auto S_i = sites.op("Sz",i);


//         // Calculate first correlator at site i
//         C = psi.A(i);
//         C *= S_i;
//         C *= dag(prime(psi.A(i),Site));
//         correlators[k] = 2*C.real();
//     }
//     return correlators;
// }

PyObject *makelist(array_type array, size_t size) {
    PyObject *l = PyList_New(size);
    for (size_t i = 0; i != size; ++i) {
        PyList_SET_ITEM(l, i, Py_BuildValue("f", array(i+1)));
    }
    return l;
}

static PyObject* get_correlators(PyObject* self, PyObject* py_args)
{   
    PyObject* args;
    int N_mode;
    char* cor_2_name;
    char* cor_1_name;
    int do_1;
    int do_2;
    
    PyArg_ParseTuple(py_args, "ssiOii", &cor_1_name, &cor_2_name, &N_mode, &args, &do_1, &do_2);
    auto ret_val_dmrg = dmrg_H(N_mode, args);
    
    MPS psi = ret_val_dmrg.psi;
    auto sites = ret_val_dmrg.sites;

    float *correlators_2;
    float *correlators_1;
    
    int K_1 = 0;
    int K_2 = 0;
    
    if(do_1 == 1){
        K_1 = N_mode;
    };
    if(do_2 == 1){
        K_2 = (N_mode*(N_mode-1))/2;
    }; 
    
    PyObject *l_2 = PyList_New(K_1 + K_2);
        
//     float *correlators_1 = get_correlators_1(psi, sites, cor_1_name);  
//     for(int i = 0; i !=(N_mode+1); ++i){
//         PyList_SET_ITEM(l_2, i, Py_BuildValue("f", *(correlators_1+i)));
//     };
    if(do_1 == 1){
        correlators_1 = get_correlators_1(psi, sites, cor_1_name);
        
        for(int i = 0; i !=K_1; ++i){
            PyList_SET_ITEM(l_2, i, Py_BuildValue("f", *(correlators_1+i)));
        };
    };
    
    if(do_2 == 1){
        correlators_2 = get_correlators_2(psi, sites, cor_2_name);
        
        for(int i = 0; i !=K_2; ++i){
            PyList_SET_ITEM(l_2, K_1+i, Py_BuildValue("f", *(correlators_2+i)));
        } ;  
    };
    
    
    return l_2;
}

static PyMethodDef myMethods[] = {
    { "get_correlators", get_correlators, METH_VARARGS, "Calculate correlators" },
    { NULL, NULL, 0, NULL }
};


// Our Module Definition struct
static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "corr_ssh_comp",
    "",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_corr_ssh_comp(void)
{
    return PyModule_Create(&myModule);
}