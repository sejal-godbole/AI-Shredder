import os
import torch
from torch.utils.data import DataLoader
from dataset import RVLCDIPDataset, get_transforms, collate_skip_none
from model import load_model
from utils.confidentiality import get_confidentiality_from_image
from utils.logger import log_to_csv
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# -----------------------
# Paths and settings
# -----------------------
TEST_CSV = 'data/test_labels.csv'
IMG_ROOT = 'data'  # folder where test images actually exist
CHECKPOINT_PATH = 'models/best_model.pth'
LOG_PATH = 'logs/eval_results.csv'
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------
# Evaluation
# -----------------------
def evaluate():
    # Create dataset and DataLoader
    test_dataset = RVLCDIPDataset(TEST_CSV, IMG_ROOT, transform=get_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_skip_none  # skip None samples
    )

    # Load model
    model = load_model(CHECKPOINT_PATH, num_classes=16, device=DEVICE)
    model.eval()

    results = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue  # skip empty batch
            images, labels, doc_ids = batch
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top_probs, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for doc_id, pred, prob in zip(doc_ids, preds.cpu().numpy(), top_probs.cpu().numpy()):
                # Pad doc_id to match filenames if needed
                filename = f"{int(doc_id):010d}.tif"
                image_path = os.path.join(IMG_ROOT, filename)

                confidentiality_level, pii_count, pii_entities = get_confidentiality_from_image(image_path)

                print(f"Document ID: {doc_id}")
                print(f"Predicted Class: {pred} (Prob: {prob:.2f})")
                print(f"Confidentiality Level: {confidentiality_level}")
                print(f"PII Count: {pii_count}")
                print(f"Detected PII Entities: {pii_entities}\n")

                log_to_csv(LOG_PATH, [doc_id, pred, confidentiality_level, round(prob * 100, 2)])
                results.append((doc_id, pred, confidentiality_level, round(prob * 100, 2)))

    # -----------------------
    # Compute metrics
    # -----------------------
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1],
        'Confusion_Matrix': [conf_matrix.tolist()]
    })
    os.makedirs('logs', exist_ok=True)
    metrics_df.to_csv('logs/performance_metrics.csv', index=False)

    print(f"\n=== Performance Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"\nPerformance metrics saved to logs/performance_metrics.csv")

    return results

if __name__ == '__main__':
    evaluate()
