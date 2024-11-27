def transform_knobs2vector(knobs_detail, knobs):
    keys = list(knobs.keys())
    ys = []
    for key in keys:
        if knobs_detail[key]['vartype'] == 'integer' or knobs_detail[key]['vartype'] == 'real':
            # minv, maxv = knobs_detail[key]['min_val'], knobs_detail[key]['max_val']
            # tmpv = (knobs[key] - minv) / (maxv - minv)
            # ys.append(tmpv)
            ys.append(knobs[key])
        elif knobs_detail[key]['vartype'] == 'enum' :
            enum_vs = knobs_detail[key]['enumvals']
            tmpv = enum_vs.index(knobs[key]) / (len(enum_vs) - 1)
            ys.append(tmpv)
        elif knobs_detail[key]['vartype'] == 'bool':
            if knobs[key].upper() == 'ON':
                ys.append(1)
            else:
                ys.append(0)
        else:
            pass
    return ys

def transform_vector2knobs(knobs_detail, target_knobs, vector):
    knobs = {}
    for i,key in enumerate(target_knobs):
        if knobs_detail[key]['vartype'] == 'integer' :
            # minv, maxv = knobs_detail[key]['min'], knobs_detail[key]['max']
            # tmpv = (maxv - minv) * float(vector[i]) + minv
            # knobs[key] = int(tmpv)
            knobs[key] = int(vector[i])
        elif knobs_detail[key]['vartype'] == 'real':
            knobs[key] = float(vector[i])
            
        elif knobs_detail[key]['vartype'] == 'enum':
            enum_vs = knobs_detail[key]['enum_values']
            tmpv = vector[i] * (len(enum_vs) - 1)
            knobs[key] = enum_vs[int(tmpv)]
        else:
            pass
    return knobs
