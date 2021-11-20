import pandas as pd
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_predictions(model, test_loader):
    model = model.eval()
    model = model.to(device)
    patient_id = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for key in batch:
                if key != 'patient_id':
                    batch[key] = batch[key].to(device)
            batch_scores = model(batch)
            batch_probabilities = torch.sigmoid(batch_scores)
            batch_predictions = (batch_scores >= 0.5).int().tolist()
            predictions.extend(batch_predictions)
            patient_id.extend(batch['patient_id'])
    df = pd.DataFrame({'Id': patient_id, 'Predicted': predictions})
    return df