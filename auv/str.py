# Copyright (c) 2021 Perspecta Labs. All rights reserved.
import numpy as np
import os
import types

from .AUV_sim import DexcelInterface


def ff(x, params):
    # Set up the X in the correct form
    auvx = dict()
    oh = [0 for _ in range(len(params.onehots))]
    ohval = [None for _ in range(len(params.onehots))]
    allinf = [np.Inf] * params.ycount

    for xi in params.xonehots:
        ohs = xi.split('_')
        # Get which of the one-hots we're checking
        intoh = int(ohs[0])
        # Add to the overall count
        oh[intoh] += x[params.xonehots[xi]]
        # If we have more than one, that's too many, punt
        if oh[intoh] > 1:
            print(f'got too many one hots!  returning inf: {x}')
            return {"f": allinf}
        # Save the index of this one-hot
        if x[params.xonehots[xi]]:
            ohval[intoh] = int(ohs[1])

    for xnm in params.xinputs:
        # If all of the values are valid, just use them
        if len(params.bounds[xnm]) == 2:
            auvx[xnm] = x[params.xinputs[xnm]]
        # else, if we have a multiplier, do that
        elif len(params.bounds[xnm]) == 3:
            auvx[xnm] = x[params.xinputs[xnm]] * params.bounds[xnm][2]
        # else we have a list of valid values, select that
        else:
            auvx[xnm] = params.bounds[xnm][2 + x[params.xinputs[xnm]]]

    # Make sure we have one and only one
    if params.onehots:
        for o in oh:
            if o != 1:
                print(f'got not enough one hots! returning inf: {x}')
                return {"f": allinf}

    # The input takes an index, so set that to what the one-hot said
    if params.onehots:
        for i, _ in enumerate(oh):
            auvx[params.onehots[i]] = ohval[i] + params.one_hots[params.onehots[i]][2]

    # We only use the objectives and constraints
    ret = params.func.problem(auvx)

    # print(f'auvx {auvx}')
    # print(f"_f {ret}")
    return ret


def build_str_params(excelfile="AUVRangeSpeedV3.pkl"):
    fparams = types.SimpleNamespace()
    if not os.path.isabs(excelfile):
        pt = os.path.dirname(os.path.realpath(__file__))
        excelfile = os.path.join(pt, excelfile)
    assert os.path.exists(excelfile)

    fparams.func = DexcelInterface(filename=excelfile)

    fparams.bounds = {
        'Diameter': (.25, 1.00),
        'Length': (1.0, 5.0),
        'Depth_Rating': (200, 500),
        'Safety_Factor_for_Depth_Rating': (1.33, 2.0),
        'Battery_Specific_Energy': (100, 360),
        'Battery_Fraction': (0.4, 0.6),
        'Cd_Drag_coefficient': (0.0078, 0.0080),
        'Appendage_added_area': (0.1, 0.2),
        'Propulsion_efficiency': (0.45, 0.55),
        'Density_of_seawater': (1025, 1030),
        'CruiseSpeed': (0.5, 1.5),
        'Design_Hotel_Power_Draw': (20, 22)
    }
    fparams.one_hots = {
        'PV_Material_Choice': (0, 4, 1)
    }

    # Collect the inputs
    fparams.xinputs = dict()
    i = 0
    for nm in fparams.func.inputs:
        if nm not in fparams.bounds:
            if nm not in fparams.one_hots:
                print(f"Woopsie, {nm} doesn't have a bounds")
            continue
        fparams.xinputs[nm] = i
        i += 1

    # Deal with the one_hots (if you have any)
    fparams.onehots = list()
    fparams.xonehots = dict()
    onehotcount = len(fparams.xinputs)
    xcount = 0
    for i, nm in enumerate(fparams.one_hots):
        fparams.onehots.append(nm)
        for j in range(fparams.one_hots[nm][0], fparams.one_hots[nm][1] + 1):
            fparams.xonehots[f'{i}_{j}'] = onehotcount + xcount
            xcount += 1

    fparams.ycount = len(fparams.func.objectives) + len(fparams.func.constraints)

    # Get our counts and indices for later
    fparams.numobjs = len(fparams.func.objectives)
    fparams.numconsts = len(fparams.func.constraints)
    fparams.indconsts = len(fparams.func.objectives)
    return fparams


def run_p(fx, fy, fparams):
    const = fy[fparams.indconsts:fparams.indconsts + fparams.numconsts]
    ret = list()
    for i in range(len(const)):
        ret.append(const[i] <= 0)

    # Make sure that we only select 1 per one-hot variable
    for i in range(len(fparams.onehots)):
        xsum = 0
        for j in range(len(fparams.xonehots)):
            xi = f'{i}_{j}'
            if xi in fparams.xonehots:
                xsum += fx[fparams.xonehots[xi]]

        ret.append(xsum <= 1)
        ret.append(xsum >= 1)

    # print(f'_const {ret}')
    return False if False in ret else True


def run_phi(fx, fy, fparams):
    obj = fy[:fparams.numobjs]
    return obj[0]