import numpy as np
import SimpleITK as sitk


def generate_vex_files(np_arr, extractor):

    mask = np.copy(np_arr)
    mask[np_arr > 0] = 1
    # print('Extraction parameters:\n\t', extractor.settings)
    # print('Enabled filters:\n\t', extractor.enabledImagetypes)
    # print('Enabled features:\n\t', extractor.enabledFeatures)

    data_x = sitk.GetImageFromArray(np_arr)
    data_y = sitk.GetImageFromArray(mask)

    features = extractor.execute(data_x, data_y, label=1)
    keys = sorted(features.keys())
    vs = []
    for key in keys:
        if type(features[key]) == tuple:
            for v in features[key]:
                vs.append(v)
        elif type(features[key]) ==np.ndarray:
            vs.append(float(features[key]))
        elif type(features[key]) in [int, float, np.float64]:
            vs.append(float(features[key]))
        elif type(features[key]) in [str, dict]:
            continue
        else:
            print(type(features[key]))

    # print(f"len={len(vs)}, {vs}")
    return vs
