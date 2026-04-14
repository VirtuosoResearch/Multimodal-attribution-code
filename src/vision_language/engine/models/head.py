import torch
import torch.nn.functional as F
from tqdm import tqdm
from timm.models import create_model
from engine.clip import clip

def get_text_dataset_per_class(text_dataset):
    """Group text samples by class label."""
    per_class = {}
    for batch in tqdm(text_dataset):
        try:
            text_embds, label, eot_indices = batch
        except:
            text_embds, label = batch
            eot_indices = None
        label = int(label)
        per_class.setdefault(label, []).append((text_embds, eot_indices))
    
    return per_class


def get_zero_shot_weights(text_dataset, num_classes, in_features, device="cuda"):
    """
    Compute per-class text features by averaging over multiple templates,
    then normalize to obtain zero-shot weights.
    """
    with torch.no_grad():
        per_class = get_text_dataset_per_class(text_dataset)
        weights = torch.zeros(num_classes, in_features)
        for label in per_class.keys():
            texts_embds_list = []
            for text_embds, _ in per_class[label]:
                texts_embds_list.append(text_embds.unsqueeze(0).to(device))
            avg_features = torch.cat(texts_embds_list, dim=0).mean(dim=0)
            weights[label] = avg_features.cpu()
        weights = F.normalize(weights, dim=1)
    return weights

class UML(torch.nn.Module):
    def __init__(self, 
                 vision_model,
                 text_indim,
                 num_classes, 
                 bias=False, 
                 learnable_temp=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.img_proj = None  

        self.vision_model = create_model(vision_model, pretrained=True, img_size=224)        
        self.shared_dim = self.vision_model.num_features   
        if text_indim > 0:
            self.img_proj = torch.nn.Linear(self.vision_model.num_features, text_indim, bias=bias) 
            self.shared_dim = text_indim    
   
        self.head = torch.nn.Linear(self.shared_dim, num_classes, bias=bias)
        self.img_scale = torch.nn.Parameter(torch.tensor(1.0)) if learnable_temp else torch.tensor(1.0)
        self.txt_scale = torch.nn.Parameter(torch.tensor(1.0)) if learnable_temp else torch.tensor(1.0)
        
    def forward(self, images, text_features = None):
        images = self.vision_model.forward(images)
        images = self.img_proj(images) if self.img_proj is not None else images
        img_logits = self.head(images) * self.img_scale
        if text_features is not None:
            txt_logits = self.head(text_features) * self.txt_scale
            return img_logits, txt_logits
        return img_logits, None

    def zero_shot_init(self, zeroshot_dataset):
        print("=> Initializing head with zero-shot weights")
        self.head.weight.data = get_zero_shot_weights(zeroshot_dataset, self.num_classes, self.shared_dim)
 

class UMLClip(torch.nn.Module):
    def __init__(self, 
                 clip_encoder,
                 num_classes, 
                 logit_scale_init=torch.log(torch.tensor(1/0.07)),
                 bias=False, 
                 learnable_temp=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.img_proj = None  
        self.vision_model, _ = clip.load(clip_encoder, jit=False) 
        self.shared_dim = self.vision_model.embed_dim
        self.head = torch.nn.Linear(self.shared_dim, num_classes, bias=bias)
        self.logit_scale = torch.tensor(logit_scale_init)  # fixed scale
        self.vision_model.float()

    def forward(self, images, text_features = None):
        images = self.vision_model.encode_image(images)
        img_logits = self.head(images) * self.logit_scale.exp()
        if text_features is not None:
            txt_logits = self.head(text_features) * self.logit_scale.exp()
            return img_logits, txt_logits
        return img_logits, None

    def zero_shot_init(self, zeroshot_dataset):
        print("=> Initializing head with zero-shot weights")
        self.head.weight.data = get_zero_shot_weights(zeroshot_dataset, self.num_classes, self.shared_dim)
 
