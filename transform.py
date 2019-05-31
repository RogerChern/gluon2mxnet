import re


def layer_to_stage(match):
    stage = match.group(1)
    unit = int(match.group(2)) + 1
    return r"stage%s_unit%s" % (stage, unit)


def gluon_to_mxnet(name):
    # dot to underscore
    name = name.replace(".", "_")
    if name.startswith("layer"):
        p = re.compile(r"layer(\d+)_(\d+)")
        # layer to stage
        name = p.sub(layer_to_stage, name)
    # downsample to sc
    name = name.replace("downsample_0", "sc")
    name = name.replace("downsample_1", "sc_bn")
    # running to moving
    name = name.replace("running", "moving")
    # conv1 to conv0
    if name.startswith("conv1"):
        name = name.replace("conv1", "conv0")
    # bn1 to bn0
    if name.startswith("bn1"):
        name = name.replace("bn1", "bn0")
    # add prefix
    if "moving" in name:
        return "aux:" + name
    else:
        return "arg:" + name


def test_suit():
    example = [
        'bn1.beta', 'bn1.gamma', 'bn1.running_mean', 'bn1.running_var',
        'conv1.weight', 'fc.bias', 'fc.weight', 'layer1.0.bn1.beta',
        'layer1.0.bn1.gamma', 'layer1.0.bn1.running_mean',
        'layer1.0.bn1.running_var', 'layer1.0.bn2.beta', 'layer1.0.bn2.gamma',
        'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var',
        'layer1.0.bn3.beta', 'layer1.0.bn3.gamma', 'layer1.0.bn3.running_mean',
        'layer1.0.bn3.running_var', 'layer1.0.conv1.weight',
        'layer1.0.conv2.weight', 'layer1.0.conv3.weight',
        'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.beta',
        'layer1.0.downsample.1.gamma', 'layer1.0.downsample.1.running_mean',
        'layer1.0.downsample.1.running_var', 'layer1.1.bn1.beta',
        'layer1.1.bn1.gamma', 'layer1.1.bn1.running_mean',
        'layer1.1.bn1.running_var', 'layer1.1.bn2.beta', 'layer1.1.bn2.gamma',
        'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
        'layer1.1.bn3.beta', 'layer1.1.bn3.gamma', 'layer1.1.bn3.running_mean',
        'layer1.1.bn3.running_var', 'layer1.1.conv1.weight',
        'layer1.1.conv2.weight', 'layer1.1.conv3.weight', 'layer1.2.bn1.beta',
        'layer1.2.bn1.gamma', 'layer1.2.bn1.running_mean',
        'layer1.2.bn1.running_var', 'layer1.2.bn2.beta', 'layer1.2.bn2.gamma',
        'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var',
        'layer1.2.bn3.beta', 'layer1.2.bn3.gamma', 'layer1.2.bn3.running_mean',
        'layer1.2.bn3.running_var', 'layer1.2.conv1.weight',
        'layer1.2.conv2.weight', 'layer1.2.conv3.weight', 'layer2.0.bn1.beta',
        'layer2.0.bn1.gamma', 'layer2.0.bn1.running_mean',
        'layer2.0.bn1.running_var', 'layer2.0.bn2.beta', 'layer2.0.bn2.gamma',
        'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
        'layer2.0.bn3.beta', 'layer2.0.bn3.gamma', 'layer2.0.bn3.running_mean',
        'layer2.0.bn3.running_var', 'layer2.0.conv1.weight',
        'layer2.0.conv2.weight', 'layer2.0.conv3.weight',
        'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.beta',
        'layer2.0.downsample.1.gamma', 'layer2.0.downsample.1.running_mean',
        'layer2.0.downsample.1.running_var', 'layer2.1.bn1.beta',
        'layer2.1.bn1.gamma', 'layer2.1.bn1.running_mean',
        'layer2.1.bn1.running_var', 'layer2.1.bn2.beta', 'layer2.1.bn2.gamma',
        'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
        'layer2.1.bn3.beta', 'layer2.1.bn3.gamma', 'layer2.1.bn3.running_mean',
        'layer2.1.bn3.running_var', 'layer2.1.conv1.weight',
        'layer2.1.conv2.weight', 'layer2.1.conv3.weight', 'layer2.2.bn1.beta',
        'layer2.2.bn1.gamma', 'layer2.2.bn1.running_mean',
        'layer2.2.bn1.running_var', 'layer2.2.bn2.beta', 'layer2.2.bn2.gamma',
        'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var',
        'layer2.2.bn3.beta', 'layer2.2.bn3.gamma', 'layer2.2.bn3.running_mean',
        'layer2.2.bn3.running_var', 'layer2.2.conv1.weight',
        'layer2.2.conv2.weight', 'layer2.2.conv3.weight', 'layer2.3.bn1.beta',
        'layer2.3.bn1.gamma', 'layer2.3.bn1.running_mean',
        'layer2.3.bn1.running_var', 'layer2.3.bn2.beta', 'layer2.3.bn2.gamma',
        'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var',
        'layer2.3.bn3.beta', 'layer2.3.bn3.gamma', 'layer2.3.bn3.running_mean',
        'layer2.3.bn3.running_var', 'layer2.3.conv1.weight',
        'layer2.3.conv2.weight', 'layer2.3.conv3.weight', 'layer3.0.bn1.beta',
        'layer3.0.bn1.gamma', 'layer3.0.bn1.running_mean',
        'layer3.0.bn1.running_var', 'layer3.0.bn2.beta', 'layer3.0.bn2.gamma',
        'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var',
        'layer3.0.bn3.beta', 'layer3.0.bn3.gamma', 'layer3.0.bn3.running_mean',
        'layer3.0.bn3.running_var', 'layer3.0.conv1.weight',
        'layer3.0.conv2.weight', 'layer3.0.conv3.weight',
        'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.beta',
        'layer3.0.downsample.1.gamma', 'layer3.0.downsample.1.running_mean',
        'layer3.0.downsample.1.running_var', 'layer3.1.bn1.beta',
        'layer3.1.bn1.gamma', 'layer3.1.bn1.running_mean',
        'layer3.1.bn1.running_var', 'layer3.1.bn2.beta', 'layer3.1.bn2.gamma',
        'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
        'layer3.1.bn3.beta', 'layer3.1.bn3.gamma', 'layer3.1.bn3.running_mean',
        'layer3.1.bn3.running_var', 'layer3.1.conv1.weight',
        'layer3.1.conv2.weight', 'layer3.1.conv3.weight', 'layer3.2.bn1.beta',
        'layer3.2.bn1.gamma', 'layer3.2.bn1.running_mean',
        'layer3.2.bn1.running_var', 'layer3.2.bn2.beta', 'layer3.2.bn2.gamma',
        'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var',
        'layer3.2.bn3.beta', 'layer3.2.bn3.gamma', 'layer3.2.bn3.running_mean',
        'layer3.2.bn3.running_var', 'layer3.2.conv1.weight',
        'layer3.2.conv2.weight', 'layer3.2.conv3.weight', 'layer3.3.bn1.beta',
        'layer3.3.bn1.gamma', 'layer3.3.bn1.running_mean',
        'layer3.3.bn1.running_var', 'layer3.3.bn2.beta', 'layer3.3.bn2.gamma',
        'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var',
        'layer3.3.bn3.beta', 'layer3.3.bn3.gamma', 'layer3.3.bn3.running_mean',
        'layer3.3.bn3.running_var', 'layer3.3.conv1.weight',
        'layer3.3.conv2.weight', 'layer3.3.conv3.weight', 'layer3.4.bn1.beta',
        'layer3.4.bn1.gamma', 'layer3.4.bn1.running_mean',
        'layer3.4.bn1.running_var', 'layer3.4.bn2.beta', 'layer3.4.bn2.gamma',
        'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var',
        'layer3.4.bn3.beta', 'layer3.4.bn3.gamma', 'layer3.4.bn3.running_mean',
        'layer3.4.bn3.running_var', 'layer3.4.conv1.weight',
        'layer3.4.conv2.weight', 'layer3.4.conv3.weight', 'layer3.5.bn1.beta',
        'layer3.5.bn1.gamma', 'layer3.5.bn1.running_mean',
        'layer3.5.bn1.running_var', 'layer3.5.bn2.beta', 'layer3.5.bn2.gamma',
        'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var',
        'layer3.5.bn3.beta', 'layer3.5.bn3.gamma', 'layer3.5.bn3.running_mean',
        'layer3.5.bn3.running_var', 'layer3.5.conv1.weight',
        'layer3.5.conv2.weight', 'layer3.5.conv3.weight', 'layer4.0.bn1.beta',
        'layer4.0.bn1.gamma', 'layer4.0.bn1.running_mean',
        'layer4.0.bn1.running_var', 'layer4.0.bn2.beta', 'layer4.0.bn2.gamma',
        'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var',
        'layer4.0.bn3.beta', 'layer4.0.bn3.gamma', 'layer4.0.bn3.running_mean',
        'layer4.0.bn3.running_var', 'layer4.0.conv1.weight',
        'layer4.0.conv2.weight', 'layer4.0.conv3.weight',
        'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.beta',
        'layer4.0.downsample.1.gamma', 'layer4.0.downsample.1.running_mean',
        'layer4.0.downsample.1.running_var', 'layer4.1.bn1.beta',
        'layer4.1.bn1.gamma', 'layer4.1.bn1.running_mean',
        'layer4.1.bn1.running_var', 'layer4.1.bn2.beta', 'layer4.1.bn2.gamma',
        'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var',
        'layer4.1.bn3.beta', 'layer4.1.bn3.gamma', 'layer4.1.bn3.running_mean',
        'layer4.1.bn3.running_var', 'layer4.1.conv1.weight',
        'layer4.1.conv2.weight', 'layer4.1.conv3.weight', 'layer4.2.bn1.beta',
        'layer4.2.bn1.gamma', 'layer4.2.bn1.running_mean',
        'layer4.2.bn1.running_var', 'layer4.2.bn2.beta', 'layer4.2.bn2.gamma',
        'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var',
        'layer4.2.bn3.beta', 'layer4.2.bn3.gamma', 'layer4.2.bn3.running_mean',
        'layer4.2.bn3.running_var', 'layer4.2.conv1.weight',
        'layer4.2.conv2.weight', 'layer4.2.conv3.weight'
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
        print(new_k)
    mx.nd.save(converted_model_path, new_params)
