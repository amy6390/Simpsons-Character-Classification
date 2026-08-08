import torch
import torchvision.models
from torchvision.models import efficientnet_b0
import os
from PIL import Image

def load_model(state_dict_path, num_classes):
    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model=efficientnet_b0(weights='DEFAULT')
    model.classifier[1]=torch.nn.Linear(in_features=1280, out_features=num_classes)

    #loading and applying state dict
    state_dict=torch.load(state_dict_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    img_transforms=weights.transforms()
    return model, img_transforms

def prediction(img, model, transforms, classes=['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'maggie_simpson', 
                                                'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover', 'moe_szyslak', 
                                                'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 
                                                'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'troy_mcclure', 'waylon_smithers']):

    #processing image and adding a batch dimension
    img_transformed=transforms(img)
    img_transformed=img_transformed.unsqueeze(0)
    with torch.no_grad():
        preds=model(img_transformed)

    #extract probabilities and class
    probs=torch.nn.functional.softmax(preds[0], dim=0)
    class_id=probs.argmax().item()
    class_name=classes[class_id]

    return class_name