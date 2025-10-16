import torch
from PIL import Image
from torchvision import transforms
from model import load_model
from utils.confidentiality import get_confidentiality_from_image
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def infer(image_path, checkpoint_path='models/best_model.pth'):
    model = load_model(checkpoint_path, num_classes=16, device=DEVICE)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        top_prob, pred = torch.max(prob, 1)

        # Perform PII-based confidentiality analysis
        confidentiality_level, pii_count, pii_entities = get_confidentiality_from_image(image_path)

        return {
            'predicted_class': pred.item(),
            'confidence_score': round(top_prob.item() * 100, 2),
            'confidentiality_level': confidentiality_level,
            'pii_count': pii_count,
            'pii_entities': pii_entities
        }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python inference.py <image_path>')
    else:
        result = infer(sys.argv[1])
        print(result)
