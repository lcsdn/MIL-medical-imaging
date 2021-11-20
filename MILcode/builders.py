from torchvision.models import resnet18, resnet34, wide_resnet50_2

def build_pretrained_model(model_name, finetune_params):
    """
    Build a pretrained model based on name and finetuning parameters.

    Args:
        model_name: (str) name of pretrained model, usually same as in torchvision.models.
        finetune_params: depending on model, parameters to choose to apply and/or
            specify its finetuning.
    
    Output:
        pretrained_model: (nn.Module) the pretrained model.
        feature_dim: (int) the dimension of the features after average pooling
            and before the last fully connected layer.
    """
    if model_name == 'resnet18':
        pretrained_model = resnet18(pretrained=True)
        if finetune_params is True or finetune_params == 'lastconvblock':
            # only last conv block of last layer is trained
            for child in list(pretrained_model.children())[:-3]:
                for param in child.parameters():
                    param.requires_grad = False
            for param in pretrained_model.layer4[:-1].parameters():
                param.requires_grad = False
        elif finetune_params == 'lastconv':
            # only last convolution and batch norm are trained
            for param in pretrained_model.parameters():
                param.requires_grad = False
            for child in list(pretrained_model.layer4[-1].children())[-2:]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        return pretrained_model, 512
    elif model_name == 'resnet34':
        pretrained_model = resnet34(pretrained=True)
        if finetune_params is True or finetune_params == 'lastconvblock':
            # only last conv block of last layer is trained
            for child in list(pretrained_model.children())[:-3]:
                for param in child.parameters():
                    param.requires_grad = False
            for param in pretrained_model.layer4[:-1].parameters():
                param.requires_grad = False
        elif finetune_params == 'lastconv':
            # only last convolution and batch norm are trained
            for param in pretrained_model.parameters():
                param.requires_grad = False
            for child in list(pretrained_model.layer4[-1].children())[-2:]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        return pretrained_model, 512
    elif model_name == 'wide_resnet50_2':
        pretrained_model = wide_resnet50_2(pretrained=True)
        if finetune_params is False or finetune_params == 'lastconvblock':
            # only last conv block of last layer is trained
            for child in list(pretrained_model.children())[:-3]:
                for param in child.parameters():
                    param.requires_grad = False
            for param in pretrained_model.layer4[:-1].parameters():
                param.requires_grad = False
        elif finetune_params == 'lastconv':
            # only last convolution and batch norm are trained
            for param in pretrained_model.parameters():
                param.requires_grad = False
            for child in list(pretrained_model.layer4[-1].children())[-2:]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        return pretrained_model, 2048
    else:
        raise ValueError(f'Unknown model "{model_name}".')
