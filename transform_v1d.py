import re


def layer_to_stage(match):
    type = match.group(1)
    stage = match.group(2)
    op = match.group(3)
    op = "conv" if op == "conv" else "bn"
    layer = int(match.group(4))
    unit = layer // 3 + 1
    idx = layer % 3 + 1
    if type == "layers":
        return r"stage%s_unit%d_%s%d" % (stage, unit, op, idx)
    elif type == "down":
        assert unit == 1, "downsample happens in unit %d" % unit
        if op == "conv":
            return r"stage%s_unit%d_sc" % (stage, unit)
        elif op == "bn":
            return r"stage%s_unit%d_sc_bn" % (stage, unit)


def gluon_to_mxnet(name):
    # remove arg: and aux:
    name = name.replace("arg:", "").replace("aux:", "")

    # remove prefix
    name = name.replace("resnetv1d_", "")

    if name.startswith("layers") or name.startswith("down"):
        p = re.compile(r"(layers|down)(\d+)_(conv|batchnorm)(\d+)")
        # layer to stage
        name = p.sub(layer_to_stage, name)

    # running to moving
    name = name.replace("running", "moving")

    # convert stem
    if name.startswith("conv0"):
        name = name.replace("conv0", "conv0_0")
    if name.startswith("batchnorm0"):
        name = name.replace("batchnorm0", "bn0_0")
    if name.startswith("conv1"):
        name = name.replace("conv1", "conv0_1")
    if name.startswith("batchnorm1"):
        name = name.replace("batchnorm1", "bn0_1")
    if name.startswith("conv2"):
        name = name.replace("conv2", "conv0_2")
    if name.startswith("batchnorm2"):
        name = name.replace("batchnorm2", "bn0_2")

    # add prefix
    if "moving" in name:
        return "aux:" + name
    else:
        return "arg:" + name


def test_suit():
    example = [
        'resnetv1d_conv0_weight',
        'resnetv1d_batchnorm0_gamma',
        'resnetv1d_batchnorm0_beta',
        'resnetv1d_conv1_weight',
        'resnetv1d_batchnorm1_gamma',
        'resnetv1d_batchnorm1_beta',
        'resnetv1d_conv2_weight',
        'resnetv1d_batchnorm2_gamma',
        'resnetv1d_batchnorm2_beta',
        'resnetv1d_layers1_conv0_weight',
        'resnetv1d_layers1_batchnorm0_gamma',
        'resnetv1d_layers1_batchnorm0_beta',
        'resnetv1d_layers1_conv1_weight',
        'resnetv1d_layers1_batchnorm1_gamma',
        'resnetv1d_layers1_batchnorm1_beta',
        'resnetv1d_layers1_conv2_weight',
        'resnetv1d_layers1_batchnorm2_gamma',
        'resnetv1d_layers1_batchnorm2_beta',
        'resnetv1d_down1_conv0_weight',
        'resnetv1d_down1_batchnorm0_gamma',
        'resnetv1d_down1_batchnorm0_beta',
        'resnetv1d_layers1_conv3_weight',
        'resnetv1d_layers1_batchnorm3_gamma',
        'resnetv1d_layers1_batchnorm3_beta',
        'resnetv1d_layers1_conv4_weight',
        'resnetv1d_layers1_batchnorm4_gamma',
        'resnetv1d_layers1_batchnorm4_beta',
        'resnetv1d_layers1_conv5_weight',
        'resnetv1d_layers1_batchnorm5_gamma',
        'resnetv1d_layers1_batchnorm5_beta',
        'resnetv1d_layers1_conv6_weight',
        'resnetv1d_layers1_batchnorm6_gamma',
        'resnetv1d_layers1_batchnorm6_beta',
        'resnetv1d_layers1_conv7_weight',
        'resnetv1d_layers1_batchnorm7_gamma',
        'resnetv1d_layers1_batchnorm7_beta',
        'resnetv1d_layers1_conv8_weight',
        'resnetv1d_layers1_batchnorm8_gamma',
        'resnetv1d_layers1_batchnorm8_beta',
        'resnetv1d_layers2_conv0_weight',
        'resnetv1d_layers2_batchnorm0_gamma',
        'resnetv1d_layers2_batchnorm0_beta',
        'resnetv1d_layers2_conv1_weight',
        'resnetv1d_layers2_batchnorm1_gamma',
        'resnetv1d_layers2_batchnorm1_beta',
        'resnetv1d_layers2_conv2_weight',
        'resnetv1d_layers2_batchnorm2_gamma',
        'resnetv1d_layers2_batchnorm2_beta',
        'resnetv1d_down2_conv0_weight',
        'resnetv1d_down2_batchnorm0_gamma',
        'resnetv1d_down2_batchnorm0_beta',
        'resnetv1d_layers2_conv3_weight',
        'resnetv1d_layers2_batchnorm3_gamma',
        'resnetv1d_layers2_batchnorm3_beta',
        'resnetv1d_layers2_conv4_weight',
        'resnetv1d_layers2_batchnorm4_gamma',
        'resnetv1d_layers2_batchnorm4_beta',
        'resnetv1d_layers2_conv5_weight',
        'resnetv1d_layers2_batchnorm5_gamma',
        'resnetv1d_layers2_batchnorm5_beta',
        'resnetv1d_layers2_conv6_weight',
        'resnetv1d_layers2_batchnorm6_gamma',
        'resnetv1d_layers2_batchnorm6_beta',
        'resnetv1d_layers2_conv7_weight',
        'resnetv1d_layers2_batchnorm7_gamma',
        'resnetv1d_layers2_batchnorm7_beta',
        'resnetv1d_layers2_conv8_weight',
        'resnetv1d_layers2_batchnorm8_gamma',
        'resnetv1d_layers2_batchnorm8_beta',
        'resnetv1d_layers2_conv9_weight',
        'resnetv1d_layers2_batchnorm9_gamma',
        'resnetv1d_layers2_batchnorm9_beta',
        'resnetv1d_layers2_conv10_weight',
        'resnetv1d_layers2_batchnorm10_gamma',
        'resnetv1d_layers2_batchnorm10_beta',
        'resnetv1d_layers2_conv11_weight',
        'resnetv1d_layers2_batchnorm11_gamma',
        'resnetv1d_layers2_batchnorm11_beta',
        'resnetv1d_layers3_conv0_weight',
        'resnetv1d_layers3_batchnorm0_gamma',
        'resnetv1d_layers3_batchnorm0_beta',
        'resnetv1d_layers3_conv1_weight',
        'resnetv1d_layers3_batchnorm1_gamma',
        'resnetv1d_layers3_batchnorm1_beta',
        'resnetv1d_layers3_conv2_weight',
        'resnetv1d_layers3_batchnorm2_gamma',
        'resnetv1d_layers3_batchnorm2_beta',
        'resnetv1d_down3_conv0_weight',
        'resnetv1d_down3_batchnorm0_gamma',
        'resnetv1d_down3_batchnorm0_beta',
        'resnetv1d_layers3_conv3_weight',
        'resnetv1d_layers3_batchnorm3_gamma',
        'resnetv1d_layers3_batchnorm3_beta',
        'resnetv1d_layers3_conv4_weight',
        'resnetv1d_layers3_batchnorm4_gamma',
        'resnetv1d_layers3_batchnorm4_beta',
        'resnetv1d_layers3_conv5_weight',
        'resnetv1d_layers3_batchnorm5_gamma',
        'resnetv1d_layers3_batchnorm5_beta',
        'resnetv1d_layers3_conv6_weight',
        'resnetv1d_layers3_batchnorm6_gamma',
        'resnetv1d_layers3_batchnorm6_beta',
        'resnetv1d_layers3_conv7_weight',
        'resnetv1d_layers3_batchnorm7_gamma',
        'resnetv1d_layers3_batchnorm7_beta',
        'resnetv1d_layers3_conv8_weight',
        'resnetv1d_layers3_batchnorm8_gamma',
        'resnetv1d_layers3_batchnorm8_beta',
        'resnetv1d_layers3_conv9_weight',
        'resnetv1d_layers3_batchnorm9_gamma',
        'resnetv1d_layers3_batchnorm9_beta',
        'resnetv1d_layers3_conv10_weight',
        'resnetv1d_layers3_batchnorm10_gamma',
        'resnetv1d_layers3_batchnorm10_beta',
        'resnetv1d_layers3_conv11_weight',
        'resnetv1d_layers3_batchnorm11_gamma',
        'resnetv1d_layers3_batchnorm11_beta',
        'resnetv1d_layers3_conv12_weight',
        'resnetv1d_layers3_batchnorm12_gamma',
        'resnetv1d_layers3_batchnorm12_beta',
        'resnetv1d_layers3_conv13_weight',
        'resnetv1d_layers3_batchnorm13_gamma',
        'resnetv1d_layers3_batchnorm13_beta',
        'resnetv1d_layers3_conv14_weight',
        'resnetv1d_layers3_batchnorm14_gamma',
        'resnetv1d_layers3_batchnorm14_beta',
        'resnetv1d_layers3_conv15_weight',
        'resnetv1d_layers3_batchnorm15_gamma',
        'resnetv1d_layers3_batchnorm15_beta',
        'resnetv1d_layers3_conv16_weight',
        'resnetv1d_layers3_batchnorm16_gamma',
        'resnetv1d_layers3_batchnorm16_beta',
        'resnetv1d_layers3_conv17_weight',
        'resnetv1d_layers3_batchnorm17_gamma',
        'resnetv1d_layers3_batchnorm17_beta',
        'resnetv1d_layers4_conv0_weight',
        'resnetv1d_layers4_batchnorm0_gamma',
        'resnetv1d_layers4_batchnorm0_beta',
        'resnetv1d_layers4_conv1_weight',
        'resnetv1d_layers4_batchnorm1_gamma',
        'resnetv1d_layers4_batchnorm1_beta',
        'resnetv1d_layers4_conv2_weight',
        'resnetv1d_layers4_batchnorm2_gamma',
        'resnetv1d_layers4_batchnorm2_beta',
        'resnetv1d_down4_conv0_weight',
        'resnetv1d_down4_batchnorm0_gamma',
        'resnetv1d_down4_batchnorm0_beta',
        'resnetv1d_layers4_conv3_weight',
        'resnetv1d_layers4_batchnorm3_gamma',
        'resnetv1d_layers4_batchnorm3_beta',
        'resnetv1d_layers4_conv4_weight',
        'resnetv1d_layers4_batchnorm4_gamma',
        'resnetv1d_layers4_batchnorm4_beta',
        'resnetv1d_layers4_conv5_weight',
        'resnetv1d_layers4_batchnorm5_gamma',
        'resnetv1d_layers4_batchnorm5_beta',
        'resnetv1d_layers4_conv6_weight',
        'resnetv1d_layers4_batchnorm6_gamma',
        'resnetv1d_layers4_batchnorm6_beta',
        'resnetv1d_layers4_conv7_weight',
        'resnetv1d_layers4_batchnorm7_gamma',
        'resnetv1d_layers4_batchnorm7_beta',
        'resnetv1d_layers4_conv8_weight',
        'resnetv1d_layers4_batchnorm8_gamma',
        'resnetv1d_layers4_batchnorm8_beta',
        'resnetv1d_dense0_weight',
        'resnetv1d_dense0_bias',
        'resnetv1d_batchnorm0_running_mean',
        'resnetv1d_batchnorm0_running_var',
        'resnetv1d_batchnorm1_running_mean',
        'resnetv1d_batchnorm1_running_var',
        'resnetv1d_batchnorm2_running_mean',
        'resnetv1d_batchnorm2_running_var',
        'resnetv1d_layers1_batchnorm0_running_mean',
        'resnetv1d_layers1_batchnorm0_running_var',
        'resnetv1d_layers1_batchnorm1_running_mean',
        'resnetv1d_layers1_batchnorm1_running_var',
        'resnetv1d_layers1_batchnorm2_running_mean',
        'resnetv1d_layers1_batchnorm2_running_var',
        'resnetv1d_down1_batchnorm0_running_mean',
        'resnetv1d_down1_batchnorm0_running_var',
        'resnetv1d_layers1_batchnorm3_running_mean',
        'resnetv1d_layers1_batchnorm3_running_var',
        'resnetv1d_layers1_batchnorm4_running_mean',
        'resnetv1d_layers1_batchnorm4_running_var',
        'resnetv1d_layers1_batchnorm5_running_mean',
        'resnetv1d_layers1_batchnorm5_running_var',
        'resnetv1d_layers1_batchnorm6_running_mean',
        'resnetv1d_layers1_batchnorm6_running_var',
        'resnetv1d_layers1_batchnorm7_running_mean',
        'resnetv1d_layers1_batchnorm7_running_var',
        'resnetv1d_layers1_batchnorm8_running_mean',
        'resnetv1d_layers1_batchnorm8_running_var',
        'resnetv1d_layers2_batchnorm0_running_mean',
        'resnetv1d_layers2_batchnorm0_running_var',
        'resnetv1d_layers2_batchnorm1_running_mean',
        'resnetv1d_layers2_batchnorm1_running_var',
        'resnetv1d_layers2_batchnorm2_running_mean',
        'resnetv1d_layers2_batchnorm2_running_var',
        'resnetv1d_down2_batchnorm0_running_mean',
        'resnetv1d_down2_batchnorm0_running_var',
        'resnetv1d_layers2_batchnorm3_running_mean',
        'resnetv1d_layers2_batchnorm3_running_var',
        'resnetv1d_layers2_batchnorm4_running_mean',
        'resnetv1d_layers2_batchnorm4_running_var',
        'resnetv1d_layers2_batchnorm5_running_mean',
        'resnetv1d_layers2_batchnorm5_running_var',
        'resnetv1d_layers2_batchnorm6_running_mean',
        'resnetv1d_layers2_batchnorm6_running_var',
        'resnetv1d_layers2_batchnorm7_running_mean',
        'resnetv1d_layers2_batchnorm7_running_var',
        'resnetv1d_layers2_batchnorm8_running_mean',
        'resnetv1d_layers2_batchnorm8_running_var',
        'resnetv1d_layers2_batchnorm9_running_mean',
        'resnetv1d_layers2_batchnorm9_running_var',
        'resnetv1d_layers2_batchnorm10_running_mean',
        'resnetv1d_layers2_batchnorm10_running_var',
        'resnetv1d_layers2_batchnorm11_running_mean',
        'resnetv1d_layers2_batchnorm11_running_var',
        'resnetv1d_layers3_batchnorm0_running_mean',
        'resnetv1d_layers3_batchnorm0_running_var',
        'resnetv1d_layers3_batchnorm1_running_mean',
        'resnetv1d_layers3_batchnorm1_running_var',
        'resnetv1d_layers3_batchnorm2_running_mean',
        'resnetv1d_layers3_batchnorm2_running_var',
        'resnetv1d_down3_batchnorm0_running_mean',
        'resnetv1d_down3_batchnorm0_running_var',
        'resnetv1d_layers3_batchnorm3_running_mean',
        'resnetv1d_layers3_batchnorm3_running_var',
        'resnetv1d_layers3_batchnorm4_running_mean',
        'resnetv1d_layers3_batchnorm4_running_var',
        'resnetv1d_layers3_batchnorm5_running_mean',
        'resnetv1d_layers3_batchnorm5_running_var',
        'resnetv1d_layers3_batchnorm6_running_mean',
        'resnetv1d_layers3_batchnorm6_running_var',
        'resnetv1d_layers3_batchnorm7_running_mean',
        'resnetv1d_layers3_batchnorm7_running_var',
        'resnetv1d_layers3_batchnorm8_running_mean',
        'resnetv1d_layers3_batchnorm8_running_var',
        'resnetv1d_layers3_batchnorm9_running_mean',
        'resnetv1d_layers3_batchnorm9_running_var',
        'resnetv1d_layers3_batchnorm10_running_mean',
        'resnetv1d_layers3_batchnorm10_running_var',
        'resnetv1d_layers3_batchnorm11_running_mean',
        'resnetv1d_layers3_batchnorm11_running_var',
        'resnetv1d_layers3_batchnorm12_running_mean',
        'resnetv1d_layers3_batchnorm12_running_var',
        'resnetv1d_layers3_batchnorm13_running_mean',
        'resnetv1d_layers3_batchnorm13_running_var',
        'resnetv1d_layers3_batchnorm14_running_mean',
        'resnetv1d_layers3_batchnorm14_running_var',
        'resnetv1d_layers3_batchnorm15_running_mean',
        'resnetv1d_layers3_batchnorm15_running_var',
        'resnetv1d_layers3_batchnorm16_running_mean',
        'resnetv1d_layers3_batchnorm16_running_var',
        'resnetv1d_layers3_batchnorm17_running_mean',
        'resnetv1d_layers3_batchnorm17_running_var',
        'resnetv1d_layers4_batchnorm0_running_mean',
        'resnetv1d_layers4_batchnorm0_running_var',
        'resnetv1d_layers4_batchnorm1_running_mean',
        'resnetv1d_layers4_batchnorm1_running_var',
        'resnetv1d_layers4_batchnorm2_running_mean',
        'resnetv1d_layers4_batchnorm2_running_var',
        'resnetv1d_down4_batchnorm0_running_mean',
        'resnetv1d_down4_batchnorm0_running_var',
        'resnetv1d_layers4_batchnorm3_running_mean',
        'resnetv1d_layers4_batchnorm3_running_var',
        'resnetv1d_layers4_batchnorm4_running_mean',
        'resnetv1d_layers4_batchnorm4_running_var',
        'resnetv1d_layers4_batchnorm5_running_mean',
        'resnetv1d_layers4_batchnorm5_running_var',
        'resnetv1d_layers4_batchnorm6_running_mean',
        'resnetv1d_layers4_batchnorm6_running_var',
        'resnetv1d_layers4_batchnorm7_running_mean',
        'resnetv1d_layers4_batchnorm7_running_var',
        'resnetv1d_layers4_batchnorm8_running_mean',
        'resnetv1d_layers4_batchnorm8_running_var'
    ]
    example = [gluon_to_mxnet(name) for name in example]
    print(sorted(example))


if __name__ == "__main__":
    import sys
    import mxnet as mx

    assert len(sys.argv) == 2

    model_path = sys.argv[1]
    converted_model_path = model_path + ".convert"
    params = mx.nd.load(model_path)
    new_params = dict()
    for k in sorted(params.keys()):
        new_k = gluon_to_mxnet(k)
        new_params[new_k] = params[k]
        print("convert %s to %s" % (k, new_k))
    mx.nd.save(converted_model_path, new_params)
