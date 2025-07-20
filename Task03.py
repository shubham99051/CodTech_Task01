import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import copy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(img_path, max_size=512):
    image = Image.open(img_path).convert("RGB")
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Un-normalize and convert to PIL image
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

# Content and Style Loss modules
class ContentLoss(nn.Module):
    def _init_(self, target):
        super(ContentLoss, self)._init_()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def _init_(self, target_feature):
        super(StyleLoss, self)._init_()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Load images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

assert content_img.size() == style_img.size(), "Images must be of the same size"

# Use pretrained VGG19
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Layers to extract
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Build the model
def get_style_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential().to(device)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# Input image (start from content image)
input_img = content_img.clone()

# Model and losses
model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

# Optimizer
input_img.requires_grad_(True)
optimizer = optim.LBFGS([input_img])

# Style Transfer loop
num_steps = 300
print("Optimizing...")
run = [0]
while run[0] <= num_steps:

    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * 1000000 + content_score
        loss.backward()
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Step {run[0]}:")
            print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
        return loss

    optimizer.step(closure)

# Final output
input_img.data.clamp_(0, 1)
output_image = im_convert(input_img)
output_image.save("output.jpg")
output_image.show()
