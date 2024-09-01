import torch
from datasets.rainplace import RainPlaceDataset
from models.mssn import MSSN
from models.msfpn import MSFPN
from models.matcher import Matcher
from utils.logger import setup_logger

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def main():
    config = load_config('config/default.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu
