import logging

import torch
import torch.nn.functional as F


# Initiate Logger
logger = logging.getLogger(__name__)


class GradCAM():
    """Calculate GradCAM salinecy map.
    git repos: https://github.com/1Konny/gradcam_plus_plus-pytorch
    A simple example:
        # initialize a model, model_dict and gradcam
        model = torch.load(f'./models/{model_name}_ckpt_ep0020')
        model.eval()
        model_dict = {
            'type': model_name,
            'arch': model,
            'layer_name': target_layer_name,
            'input_size': (5000)
        }
        gradcam = GradCAM(model_dict, device)
        # Query for Targeted class
        items = session.query(ECGtoK).filter(ECGtoK.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))
        .filter(ECGtoK.potassium_gp1 == target_class).order_by(func.random()).limit(10).all()
        preprocess_leads(value)
        # get a GradCAM saliency map
        mask, logit = gradcam(normed_img)
        # make heatmap from mask and synthesize saliency map using heatmap and ecg
        visualize_cam_scatter(weights, one_lead_ecg, title)
        visualize_cam_line(weights, one_lead_ecg, title)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        device (torch.device): assign a device
    """

    def __init__(self, model_dict, device):
        # model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.input_size = model_dict['input_size']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):  # pylint: disable=unused-argument
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input_signal, output):  # pylint: disable=unused-argument
            self.activations['value'] = output

        # Select the target layer by layer name
        target_layer = self.model_arch._modules[layer_name]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        assert "input_size" in model_dict, "Need input size!"
        input_size = model_dict['input_size']
        self.model_arch(torch.zeros(1, 12, input_size, device=device))
        logger.info("saliency_map size : %s", self.activations['value'].shape[2:])

    def forward(self, input_signal, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 12, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, _, w = input_signal.size()

        logit = self.model_arch(input_signal)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, _ = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # Not sure whether relu should be retained
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=w)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input_signal, class_idx=None, retain_graph=False):
        return self.forward(input_signal, class_idx, retain_graph)


class GradCAMpp(GradCAM):  # pylint: disable=too-few-public-methods
    """Calculate GradCAM++ salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcam
        model = torch.load(f'./models/{model_name}_ckpt_ep0020')
        model.eval()
        model_dict = {
            'type': model_name,
            'arch': model,
            'layer_name': target_layer_name,
            'input_size': (5000)
        }
        gradcam = GradCAMpp(model_dict, device)
        # get a ecg signal from db and apply preprocess method
        items = session.query(ECGtoK).filter(ECGtoK.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))
        .filter(ECGtoK.potassium_gp1 == target_class).order_by(func.random()).limit(10).all()
        preprocess_leads(value)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(torch_ecg)
        # make heatmap from mask and synthesize saliency map using heatmap and ecg
        visualize_cam_scatter(weights, one_lead_ecg, title)
        visualize_cam_line(weights, one_lead_ecg, title)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        device (torch.device): assign a device
    """

    def __init__(self, model_dict, device):  # pylint: disable=useless-super-delegation
        super(GradCAMpp, self).__init__(model_dict, device)

    def forward(self, input_signal, class_idx=None, retain_graph=False):
        """
        Args:
            input_signal: input signal with shape of (1, 12, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, _, _ = input_signal.size()

        logit = self.model_arch(input_signal)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, v).sum(-1, keepdim=True).view(b, k, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, v).sum(-1).view(b, k, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=self.input_size)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit
