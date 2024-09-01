import torch
from torch.optim import AdamW
from datasets.rainplace import RainPlaceDataset
from models.mssn import MSSN
from models.msfpn import MSFPN
from models.matcher import Matcher
from utils.logger import setup_logger

def main():
    config = load_config('config/default.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    mssn = MSSN(**config['model']['MSSN']).to(device)
    msfpn = MSFPN(**config['model']['MSFPN']).to(device)
    matcher = Matcher().to(device)

    # Load dataset
    train_dataset = RainPlaceDataset(config['dataset']['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Optimizer
    optimizer = AdamW(list(mssn.parameters()) + list(msfpn.parameters()) + list(matcher.parameters()), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        for batch in train_loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            
            # Forward pass
            rain_free_images = mssn(images)
            features = msfpn(rain_free_images)
            loss = matcher(features, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {loss.item()}")

if __name__ == "__main__":
    main()
